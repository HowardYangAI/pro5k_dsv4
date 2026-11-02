#!/usr/bin/env python3
"""Verify archive formats, gzip CRC, output manifests, sizes and file names.

Usage: python3 tools/verify_archive.py [--root ARCHIVE] [--report /tmp/report.json]
The archive is read only. Reports contain paths and diagnostics, never file data.
Known invalid source fixtures require an exact output SHA-256 and a reason in
metadata/workspace/known_source_format_exceptions.json or --exceptions FILE.
"""
from pathlib import Path
import argparse
import collections
import gzip
import hashlib
import json
import os
import re
import sys
import time
import zlib

MAX_FILE = 100 * 1024 * 1024
SHA = re.compile(r'^[0-9a-f]{64}$')
CREDENTIAL_NAMES = {
    'auth.json', 'credentials', 'credentials.json', 'credentials.yaml',
    'credentials.yml', 'token.json', 'tokens.json', 'service-account.json',
    '.git-credentials', '.netrc', '.npmrc', '.pypirc', '.env',
}
PRIVATE_KEY_NAME = re.compile(r'^(?:id_(?:rsa|dsa|ecdsa|ed25519)(?:_sk)?)(?!\.pub$)(?:$|\.)', re.I)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--exceptions', type=Path, action='append', default=[])
    parser.add_argument('--report', type=Path, help='Optional report outside the archive; default prints JSON to stdout.')
    args = parser.parse_args()
    root = args.root.resolve()
    started = time.monotonic()
    issues, accepted = [], []
    counts = collections.Counter()
    digest_cache, json_cache, line_counts = {}, {}, {}

    def relative(path):
        try:
            return str(path.relative_to(root))
        except ValueError:
            return str(path)

    def issue(code, path, detail=None):
        item = {'code': code, 'path': relative(Path(path))}
        if detail is not None:
            item['detail'] = detail
        issues.append(item)

    def digest(path):
        if path not in digest_cache:
            h = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    h.update(chunk)
            digest_cache[path] = h.hexdigest()
        return digest_cache[path]

    def opener(path, mode='rt'):
        if path.suffix.lower() == '.gz':
            return gzip.open(path, mode, encoding='utf-8') if 't' in mode else gzip.open(path, mode)
        return path.open(mode, encoding='utf-8') if 't' in mode else path.open(mode)

    def resolve_output(value, field):
        path = Path(value)
        if not path.is_absolute():
            # Every conversation manifest uses a path relative to conversations/,
            # including the manifests nested in databases/ and attachments/.
            path = root / ('conversations' if field == 'exported_path' else '') / path
        elif not path.is_relative_to(root):
            # Absolute original export paths remain portable after git clone.
            marker = '/pro5k_dsv4/'
            if marker in str(path):
                path = root / str(path).split(marker, 1)[1]
        path = path.resolve()
        if not path.is_relative_to(root):
            raise ValueError('Output path points outside archive')
        return path

    exception_paths = [root / 'metadata/workspace/known_source_format_exceptions.json'] + args.exceptions
    exceptions = {}
    for path in exception_paths:
        if not path.exists():
            continue
        try:
            obj = json.loads(path.read_text())
            entries = obj.get('exceptions', obj.get('records', [])) if isinstance(obj, dict) else obj
            if not isinstance(entries, list):
                raise ValueError('Exception entries must be an array')
            for entry in entries:
                if not isinstance(entry, dict) or not entry.get('reason') or not SHA.fullmatch(entry.get('sha256', '')):
                    issue('invalid_exception_schema', path)
                    continue
                output = entry.get('path') or entry.get('exported_path')
                if not isinstance(output, str) or entry.get('kind') not in {'json', 'jsonl'}:
                    issue('invalid_exception_schema', path)
                    continue
                target = resolve_output(output, 'path')
                exceptions[(target, entry['kind'])] = entry
        except (OSError, ValueError, TypeError):
            issue('unreadable_exception_list', path)

    def allow_invalid(path, kind, lines=None):
        entry = exceptions.get((path, kind))
        if not entry or digest(path) != entry['sha256']:
            return False
        if kind == 'jsonl' and set(lines or []) != set(entry.get('lines', [])):
            return False
        accepted.append({'path': relative(path), 'kind': kind, 'lines': lines or [],
                         'sha256': entry['sha256'], 'reason': entry['reason']})
        return True

    files = []
    for directory, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(x for x in dirs if x != '.git')
        for name in sorted(names):
            path = Path(directory) / name
            if path.is_symlink():
                issue('unexpected_symlink', path)
                continue
            if path.is_file():
                files.append(path)
    counts['files'] = len(files)
    for index, path in enumerate(files, 1):
        try:
            size = path.stat().st_size
            counts['stored_bytes'] += size
            if size > MAX_FILE:
                issue('file_exceeds_100_mib', path, {'bytes': size})
            if path.name.lower() in CREDENTIAL_NAMES or PRIVATE_KEY_NAME.match(path.name):
                issue('credential_filename', path)
            if path.suffix.lower() == '.gz':
                counts['gzip_files'] += 1
                try:
                    with gzip.open(path, 'rb') as stream:
                        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                            counts['gzip_uncompressed_bytes'] += len(chunk)
                except (OSError, EOFError, zlib.error):
                    issue('gzip_integrity_error', path)
                    continue
            lower = path.name.lower()
            kind = 'jsonl' if lower.endswith(('.jsonl', '.jsonl.gz')) else 'json' if lower.endswith(('.json', '.json.gz')) else None
            if kind == 'json':
                counts['json_files'] += 1
                try:
                    with opener(path) as stream:
                        obj = json.load(stream)
                    if path.is_relative_to(root / 'metadata') or 'manifest' in path.name:
                        json_cache[path] = obj
                except json.JSONDecodeError as exc:
                    if not allow_invalid(path, kind):
                        issue('invalid_json', path, {'line': exc.lineno, 'column': exc.colno, 'reason': exc.msg})
                except UnicodeError:
                    if not allow_invalid(path, kind):
                        issue('invalid_json_encoding', path)
            elif kind == 'jsonl':
                counts['jsonl_files'] += 1
                invalid, records = [], 0
                with opener(path) as stream:
                    for line_number, line in enumerate(stream, 1):
                        if not line.strip():
                            continue
                        try:
                            json.loads(line)
                            records += 1
                        except json.JSONDecodeError:
                            invalid.append(line_number)
                line_counts[path] = records
                counts['jsonl_records'] += records
                if invalid and not allow_invalid(path, kind, invalid):
                    issue('invalid_jsonl_lines', path, {'lines': invalid})
        except UnicodeError:
            issue('invalid_jsonl_encoding', path)
        except (OSError, EOFError, zlib.error) as exc:
            issue('file_read_error', path, {'type': type(exc).__name__})
        if index % 2000 == 0:
            print('Checked %d/%d archive files' % (index, len(files)), file=sys.stderr, flush=True)

    specifications = []
    manifest_files = []

    def spec(node, source, field):
        value, size, expected = node.get(field), node.get('bytes'), node.get('sha256')
        if not isinstance(value, str) or not isinstance(size, int) or isinstance(size, bool) or size < 0 or not isinstance(expected, str) or not SHA.fullmatch(expected):
            issue('invalid_manifest_output_schema', source, {'field': field})
            return
        try:
            target = resolve_output(value, field)
        except ValueError:
            issue('manifest_path_outside_archive', source)
            return
        specifications.append((target, size, expected, node.get('records'), source))

    def walk(node, source):
        if isinstance(node, dict):
            if 'outputs' in node:
                if not isinstance(node['outputs'], list):
                    issue('invalid_manifest_outputs_array', source)
                else:
                    for output in node['outputs']:
                        if isinstance(output, dict):
                            spec(output, source, 'path')
                        else:
                            issue('invalid_manifest_output_schema', source)
            if 'exported_path' in node:
                spec(node, source, 'exported_path')
            elif 'destination' in node and 'bytes' in node and 'sha256' in node:
                spec(node, source, 'destination')
            # Alternative generic output records use path/bytes/sha256.
            elif 'path' in node and 'bytes' in node and 'sha256' in node:
                spec(node, source, 'path')
            for key, value in node.items():
                if key != 'outputs':
                    walk(value, source)
        elif isinstance(node, list):
            for item in node:
                walk(item, source)

    for path, obj in json_cache.items():
        if (('manifest' in path.name and (path.is_relative_to(root / 'metadata') or path.is_relative_to(root / 'conversations'))) or path == root / 'metadata/artifacts/summary.json' or path.name == 'offline_file_index.json.gz'):
            manifest_files.append(relative(path))
            walk(obj, path)
    seen, conflicting = {}, set()
    for target, size, expected, records, source in specifications:
        signature = (size, expected)
        if target in seen:
            if signature != seen[target] and target not in conflicting:
                issue('conflicting_manifests', target)
                conflicting.add(target)
            continue
        seen[target] = signature
        if not target.is_file():
            issue('manifest_output_missing', target, {'manifest': relative(source)})
            continue
        try:
            actual = target.stat().st_size
            if actual != size:
                issue('manifest_size_mismatch', target, {'expected': size, 'actual': actual, 'manifest': relative(source)})
            if digest(target) != expected:
                issue('manifest_sha256_mismatch', target, {'manifest': relative(source)})
            if isinstance(records, int) and target in line_counts and line_counts[target] != records:
                issue('manifest_record_count_mismatch', target, {'expected': records, 'actual': line_counts[target]})
        except OSError:
            issue('manifest_output_unreadable', target)
    if not manifest_files:
        issue('no_output_manifests_found', root)
    counts['manifest_documents'] = len(manifest_files)
    counts['manifest_output_references'] = len(specifications)
    counts['unique_manifest_outputs'] = len(seen)
    counts['accepted_source_format_exceptions'] = len(accepted)
    counts['issues'] = len(issues)
    report = {'ok': not issues, 'archive_root': str(root), 'elapsed_seconds': round(time.monotonic() - started, 3),
              'counts': dict(counts), 'manifest_documents': manifest_files,
              'accepted_source_format_exceptions': accepted, 'issues': issues}
    text = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if args.report:
        target = args.report.resolve()
        if target.is_relative_to(root):
            parser.error('--report must be outside the archive so verification remains read only')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        print(json.dumps({'ok': report['ok'], 'counts': report['counts'], 'report': str(target)}, ensure_ascii=False, indent=2))
    else:
        print(text, end='')
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
