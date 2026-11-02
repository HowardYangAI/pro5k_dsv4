"""Shared credential redaction and deterministic archive copying.

No credential values are logged or written to the destination.
"""
from pathlib import Path
import collections
import gzip
import hashlib
import json
import re

COUNTS = collections.Counter()
KNOWN = set()
SECRET_FIELD = re.compile(r'(?i)(?:api[_-]?key|access[_-]?token|refresh[_-]?token|id[_-]?token|client[_-]?secret|password|passwd|authorization|private[_-]?key|aws_secret_access_key|github_token)')

def _collect(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if SECRET_FIELD.search(key) and isinstance(value, str) and len(value) >= 8:
                KNOWN.add(value)
            elif isinstance(value, (dict, list)):
                _collect(value)
    elif isinstance(obj, list):
        for value in obj:
            _collect(value)

try:
    _collect(json.loads(Path('/root/.codex/auth.json').read_text()))
except (OSError, ValueError):
    pass

PATTERNS = [
    ('private_key', re.compile(r'-----BEGIN (?:OPENSSH |RSA |EC |DSA |ENCRYPTED )?PRIVATE KEY-----(?:\r?\n|\\n).*?-----END (?:OPENSSH |RSA |EC |DSA |ENCRYPTED )?PRIVATE KEY-----', re.S)),
    ('github_token', re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b')),
    ('api_token', re.compile(r'\b(?:sk-(?:proj-|svcacct-|ant-)?[A-Za-z0-9_-]{16,}|hf_[A-Za-z0-9]{20,}|xox[baprs]-[A-Za-z0-9-]{16,})\b')),
    ('aws_access_key', re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b')),
    ('jwt', re.compile(r'\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b')),
]
ASSIGN = re.compile(r'''(?i)((?<![A-Za-z0-9_])["']?(?:[A-Za-z0-9_]{0,60}(?:api[_-]?key|access[_-]?token|refresh[_-]?token|client[_-]?secret|password|passwd|aws_secret_access_key|github[_-]?token)[A-Za-z0-9_]{0,60}|authorization)["']?\s*[:=]\s*["']?)([A-Za-z0-9_/.+=:@-]{8,})''')
BEARER = re.compile(r'(?i)(\b(?:Bearer|Basic)\s+)([A-Za-z0-9_.+/=-]{12,})')
URL_AUTH = re.compile(r'(?i)(https?://)([^\s/<>"\x27]+)@')
SIGNED_QUERY = re.compile(r'(?i)([?&](?:token|access_token|api_key|key|signature|sig|x-amz-signature|x-amz-credential)=)([^\s&"\x27<>]+)')
PLACEHOLDER = re.compile(r'(?i)^(?:none|null|true|false|undefined|optional|str|changeme|your.*|example.*|placeholder.*|redacted.*|replace.*|dummy.*|test.*|token.*|api_key.*)$')

def is_symbolic_assignment(match):
    value = match[2]
    quoted = match[1].endswith(('"', "'"))
    if not quoted:
        if match.string[match.end():match.end()+1] in {'(', '['}:
            return True
        if re.fullmatch(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+', value):
            return True
        if re.fullmatch(r'[A-Z][A-Z0-9_]*(?:_KEY|_TOKEN|_PASSWORD)', value):
            return True
    return bool(PLACEHOLDER.match(value))

def sanitize_text(value):
    if not isinstance(value, str):
        return value
    for secret in sorted(KNOWN, key=len, reverse=True):
        n = value.count(secret)
        if n:
            value = value.replace(secret, '[REDACTED_CREDENTIAL]')
            COUNTS['known_credential'] += n
    for name, regex in PATTERNS:
        value, n = regex.subn('[REDACTED_' + name.upper() + ']', value)
        COUNTS[name] += n
    def assignment(m):
        if is_symbolic_assignment(m) or '[' in m[2]:
            return m[0]
        COUNTS['credential_assignment'] += 1
        return m[1] + '[REDACTED_CREDENTIAL]'
    value = ASSIGN.sub(assignment, value)
    def replacement(name, suffix=''):
        def inner(m):
            COUNTS[name] += 1
            return m[1] + '[REDACTED_CREDENTIAL]' + suffix
        return inner
    value = BEARER.sub(replacement('authorization'), value)
    value = URL_AUTH.sub(replacement('url_credentials', '@'), value)
    value = SIGNED_QUERY.sub(replacement('signed_url'), value)
    return value

def sanitize_obj(value):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            cleaned = sanitize_obj(item)
            if (isinstance(item, str) and SECRET_FIELD.search(key)
                    and 8 <= len(item) <= 4096 and not re.search(r'\s', item)
                    and not PLACEHOLDER.match(item)
                    and not re.search(r'(?i)(?:_path|_name|_env|_type|_description|_file)$', key)
                    and cleaned == item):
                cleaned = '[REDACTED_CREDENTIAL]'
                COUNTS['structured_credential'] += 1
            result[sanitize_text(key)] = cleaned
        return result
    if isinstance(value, list):
        return [sanitize_obj(v) for v in value]
    if isinstance(value, tuple):
        return [sanitize_obj(v) for v in value]
    if isinstance(value, bytes):
        return {'encoding': 'hex', 'data': value.hex()}
    return sanitize_text(value) if isinstance(value, str) else value

def sha256(data):
    return hashlib.sha256(data).hexdigest()

def copy_archive(src, dst):
    """Copy one file, redact text, gzip large text, split >90 MiB binaries.

    Returns a manifest dictionary with source/destination(s), sizes, hashes,
    compression, and redaction counts. Destination may gain .gz or .partNNN.
    Symlinks are represented as JSON rather than dereferenced.
    """
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    before = COUNTS.copy()
    if src.is_symlink():
        import os
        data = json.dumps({'symlink_target': os.readlink(src)}, ensure_ascii=False).encode()
        dst = Path(str(dst) + '.symlink.json')
        original = data
        compression = None
    else:
        original = src.read_bytes()
        data = original
        compression = None
        compressed = data[:2] == b'\x1f\x8b'
        if compressed:
            try:
                data = gzip.decompress(data)
                compression = 'gzip'
            except (OSError, EOFError):
                compressed = False
        sample = data[:8192]
        text_file = b'\x00' not in sample
        if text_file:
            try:
                value = data.decode('utf-8')
            except UnicodeDecodeError:
                text_file = False
            else:
                logical_name = src.name[:-3] if src.name.endswith('.gz') else src.name
                suffix = Path(logical_name).suffix.lower()
                if suffix in {'.jsonl', '.ndjson'}:
                    lines = []
                    for line in value.splitlines(keepends=True):
                        if not line.strip():
                            lines.append(line)
                            continue
                        try:
                            original_obj = json.loads(line)
                        except ValueError:
                            # Preserve originally malformed fixtures; validation
                            # must identify and document source-format exceptions.
                            lines.append(sanitize_text(line))
                            continue
                        cleaned_obj = sanitize_obj(original_obj)
                        if cleaned_obj == original_obj:
                            lines.append(line)
                        else:
                            ending = '\n' if line.endswith(('\n', '\r')) else ''
                            lines.append(json.dumps(cleaned_obj, ensure_ascii=False, separators=(',', ':')) + ending)
                    data = ''.join(lines).encode('utf-8')
                elif suffix == '.json':
                    try:
                        original_obj = json.loads(value)
                    except ValueError:
                        data = sanitize_text(value).encode('utf-8')
                    else:
                        cleaned_obj = sanitize_obj(original_obj)
                        data = (value if cleaned_obj == original_obj else json.dumps(cleaned_obj, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
                else:
                    data = sanitize_text(value).encode('utf-8')
        if compressed:
            data = gzip.compress(data, compresslevel=6, mtime=0)
        elif text_file and len(data) > 2 * 1024 * 1024 and dst.suffix.lower() in {'.log', '.json', '.jsonl', '.csv', '.txt', '.out', '.err', '.patch', '.tsv'}:
            dst = Path(str(dst) + '.gz')
            data = gzip.compress(data, compresslevel=6, mtime=0)
            compression = 'gzip'
    outputs = []
    max_part = 90 * 1024 * 1024
    for i in range(max(1, (len(data) + max_part - 1) // max_part)):
        part = data[i * max_part:(i + 1) * max_part]
        target = dst if len(data) <= max_part else Path(str(dst) + '.part%03d' % i)
        target.write_bytes(part)
        outputs.append({'path': str(target), 'bytes': len(part), 'sha256': sha256(part)})
    result = {'source': str(src), 'source_bytes': len(original), 'source_sha256': sha256(original), 'outputs': outputs, 'compression': compression, 'redactions': dict(COUNTS - before)}
    if len(outputs) == 1:
        result.update(destination=outputs[0]['path'], bytes=outputs[0]['bytes'], sha256=outputs[0]['sha256'])
    return result
