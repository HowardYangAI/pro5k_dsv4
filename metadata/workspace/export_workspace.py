#!/usr/bin/env python3
"""Export workspace experiments and source deltas without modifying originals.

Usage: python /tmp/pro5k_export_workspace.py DEST
Destination files and manifest use the shared redaction/large-artifact policy.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import gzip
import json
import os
import sqlite3
import subprocess
import sys
import tempfile

from pro5k_archive_common import sanitize_text, sanitize_obj, copy_archive, COUNTS

ROOT = Path('/sgl-workspace')
OFFLINE_DIRS = {'test', 'tests', 'docs', 'docs_new', 'benchmark', 'benchmarks', 'examples', 'scripts'}
SQLITE_SIDECARS = ('-wal', '-shm')
EXCLUDED_DIRS = {'.git', '__pycache__', '.pytest_cache', '.ruff_cache', 'target', 'build', 'deep_ep.egg-info', '.github-auth', '.aws', '.codex', '.agents', 'node_modules', '.venv', 'venv'}
BASES = {
    'sglang-dsv4-fixed': '5c6e59a2c83ce00c52c64442497f9ced66b2e62e',
    'sglang-swa-evict-fixed': 'd07dd66fb75e9382e8455d1e40adc6e06a4feb89',
    'sglang-swa-evict-fixed-temp0': 'd07dd66fb75e9382e8455d1e40adc6e06a4feb89',
    'sglang-upstream-main': '3ec22948c122f99f742b1c1c0921893b6c7de958',
    'sglang-official-pr36185': 'e51a3ae65e3401b21de860c90a64102133d2d6a6',
    'sglang-pr32320-inspect': '48886803b94596884d737132a381bd5124ea1cfd',
}


def git(repo, *args, binary=False):
    result = subprocess.run(['git', '-c', 'safe.directory=*', '-C', str(repo), *args], capture_output=True, text=not binary)
    if result.returncode:
        raise RuntimeError(f'Git read operation failed in {repo}: {args[0]}')
    return result.stdout


def paths_below(root):
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDED_DIRS)
        for filename in sorted(files):
            if filename.endswith(SQLITE_SIDECARS):
                continue
            yield Path(directory) / filename
        # Preserve directory symlinks as metadata as well.
        for directory_name in dirs[:]:
            path = Path(directory) / directory_name
            if path.is_symlink():
                yield path
                dirs.remove(directory_name)


def write_generated(text, destination, records, source_label):
    with tempfile.TemporaryDirectory(prefix='pro5k-generated-') as tmp:
        temporary = Path(tmp) / destination.name
        temporary.write_text(text if destination.suffix.lower() == '.json' else sanitize_text(text), encoding='utf-8')
        record = copy_archive(temporary, destination)
        record['source'] = source_label
        record['generated'] = True
        records.append(record)
        return record


def write_json(value, destination, records, source_label):
    return write_generated(json.dumps(sanitize_obj(value), ensure_ascii=False, indent=2) + '\n', destination, records, source_label)


def relative_outputs(record, destination_root):
    for output in record.get('outputs', []):
        output['path'] = str(Path(output['path']).relative_to(destination_root))
    if 'destination' in record:
        record['destination'] = str(Path(record['destination']).relative_to(destination_root))


def export_tree(source, target, records):
    for file in paths_below(source):
        rel = file.relative_to(source)
        destination = target / rel
        if file.suffix == '.db' and file.is_file():
            # Preserve benchmark tasks/results/configuration in a searchable,
            # restorable text representation that can be safely redacted.
            connection = sqlite3.connect(file.resolve().as_uri() + '?mode=ro', uri=True)
            try:
                dump = '\n'.join(connection.iterdump()) + '\n'
            finally:
                connection.close()
            record = write_generated(dump, Path(str(destination) + '.sql'), records, str(file))
            record['conversion'] = 'SQLite read-only SQL dump; restore with sqlite3 TARGET.db < dump.sql'
        else:
            record = copy_archive(file, destination)
            record['source_mode'] = oct(file.lstat().st_mode & 0o777)
            if '.bundle' in file.name:
                record['opaque_git_bundle'] = True
                record['note'] = 'Git pack payload; requires separate historical-object credential review.'
            annotate_original_format(file, record)
            records.append(record)


def changed_paths(repo):
    staged = set(filter(None, git(repo, 'diff', '--cached', '--name-only', '-z').split('\0')))
    unstaged = set(filter(None, git(repo, 'diff', '--name-only', '-z').split('\0')))
    untracked = set(filter(None, git(repo, 'ls-files', '--others', '--exclude-standard', '-z').split('\0')))
    return tuple({path for path in paths if not path.endswith(SQLITE_SIDECARS)} for paths in (staged, unstaged, untracked))


def export_source_state(repo, destination, records):
    head = git(repo, 'rev-parse', 'HEAD').strip()
    name = repo.name
    base = BASES.get(name, head)
    target = destination / 'source' / 'worktrees' / name
    staged, unstaged, untracked = changed_paths(repo)
    committed = set(filter(None, git(repo, 'diff', '--name-only', '-z', base, head).split('\0'))) if base != head else set()
    status = git(repo, 'status', '--porcelain=v1', '--untracked-files=all')
    common = git(repo, 'rev-parse', '--git-common-dir').strip()
    common = str((repo / common).resolve()) if not os.path.isabs(common) else common
    branch = git(repo, 'branch', '--show-current').strip() or None
    refs = git(repo, 'for-each-ref', '--format=%(refname) %(objectname)', 'refs/heads', 'refs/stash').splitlines()
    remote_names = git(repo, 'remote').splitlines()
    remotes = {}
    for remote in remote_names:
        remotes[remote] = git(repo, 'remote', 'get-url', remote).strip()
    metadata = {
        'original_path': str(repo), 'head_commit': head, 'base_commit': base,
        'branch': branch, 'git_common_directory': common, 'local_refs': refs,
        'is_shallow': git(repo, 'rev-parse', '--is-shallow-repository').strip() == 'true',
        'remotes': remotes, 'status_porcelain': status,
        'staged_paths': sorted(staged), 'unstaged_paths': sorted(unstaged),
        'untracked_paths': sorted(untracked), 'committed_delta_paths': sorted(committed),
        'recent_commits': git(repo, 'log', '-8', '--format=%H %ad %s', '--date=iso-strict').splitlines(),
        'restore': 'Check out base_commit; apply committed.patch; preserve HEAD changes via staged.patch then unstaged.patch. files/ contains current relevant source and tests; untracked source paths are copied to files/. Patches contain credential redaction when necessary.',
    }
    write_json(metadata, target / 'metadata.json', records, f'{repo}:git-read-only-metadata')
    patch_specs = [('staged.patch', ['diff', '--cached', '--binary']), ('unstaged.patch', ['diff', '--binary']), ('head_to_worktree.patch', ['diff', 'HEAD', '--binary'])]
    if base != head:
        patch_specs.append(('committed.patch', ['diff', base, head, '--binary']))
        # Include individual mail patches so reviewers retain commit messages.
        commits = git(repo, 'format-patch', '--stdout', '--binary', f'{base}..{head}')
        if commits:
            write_generated(commits, target / 'committed_history.mbox', records, f'{repo}:format-patch:{base}..{head}')
    for filename, args in patch_specs:
        patch = git(repo, *args)
        if patch:
            write_generated(patch, target / filename, records, f'{repo}:git-{"-".join(args)}')
    snapshots = staged | unstaged | untracked | committed
    for rel in sorted(snapshots):
        if rel.endswith(SQLITE_SIDECARS):
            continue
        path = repo / rel
        if path.is_file() or path.is_symlink():
            records.append(copy_archive(path, target / 'files' / rel))
    if name == 'sglang-dsv4-fixed':
        path = repo / 'sglang-oai-chat_0825_1024_8192_2048.jsonl'
        if path.exists():
            records.append(copy_archive(path, target / 'ignored_artifacts' / path.name))
    return metadata





def export_baseline_deltas(destination, records):
    repository = ROOT / 'sglang-dsv4-fixed'
    archived_base = '48886803b94596884d737132a381bd5124ea1cfd'
    heads = ['5c6e59a2c83ce00c52c64442497f9ced66b2e62e', 'd07dd66fb75e9382e8455d1e40adc6e06a4feb89']
    descriptions = []
    for head in heads:
        target = destination / 'source/baseline_deltas' / head
        changed = list(filter(None, git(repository, 'diff', '--name-only', '-z', archived_base, head).split('\0')))
        metadata = {'archived_base_commit': archived_base, 'target_base_commit': head, 'original_repository': str(repository), 'commits': git(repository, 'log', '--format=%H %ad %s', '--date=iso-strict', f'{archived_base}..{head}').splitlines(), 'changed_paths': changed, 'restore': 'Restore legacy SGLang bundle then check out archived_base_commit. Apply committed.patch for identical target tree, or apply history.mbox with git am to retain per-commit descriptions. files/ are snapshots from target_base_commit.'}
        write_json(metadata, target / 'metadata.json', records, f'{repository}:private-baseline-bridge:{head}')
        write_generated(git(repository, 'diff', '--binary', archived_base, head), target / 'committed.patch', records, f'{repository}:private-baseline-diff:{archived_base}..{head}')
        write_generated(git(repository, 'format-patch', '--stdout', '--binary', f'{archived_base}..{head}'), target / 'history.mbox', records, f'{repository}:private-baseline-format-patch:{archived_base}..{head}')
        for relative_path in changed:
            result = subprocess.run(['git', '-C', str(repository), 'show', f'{head}:{relative_path}'], capture_output=True)
            if result.returncode:
                continue
            with tempfile.TemporaryDirectory(prefix='pro5k-baseline-') as tmp:
                file = Path(tmp) / Path(relative_path).name
                file.write_bytes(result.stdout)
                record = copy_archive(file, target / 'files' / relative_path)
                record['source'] = f'{repository}:git-object:{head}:{relative_path}'
                record['generated'] = True
                records.append(record)
        descriptions.append(metadata)
    return descriptions


def annotate_original_format(path, record):
    """Fixtures may deliberately contain invalid JSON; preserve that fact."""
    effective = path.with_suffix('') if path.suffix == '.gz' else path
    kind = effective.suffix.lower()[1:]
    if kind not in {'json', 'jsonl'} or path.is_symlink():
        return
    raw = path.read_bytes()
    if path.suffix == '.gz':
        raw = gzip.decompress(raw)
    lines = []
    if kind == 'jsonl':
        for line_number, line in enumerate(raw.splitlines(), 1):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except (ValueError, UnicodeDecodeError):
                lines.append(line_number)
        invalid = bool(lines)
    else:
        try:
            json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            invalid = True
        else:
            invalid = False
    if invalid:
        record['source_format_exception'] = {'format': kind, 'lines': lines, 'reason': 'Original source fixture is not valid strict JSON/JSONL; preserved as supplied.'}


def export_offline_references(destination, records):
    """Preserve all test/doc/script trees and conversation-referenced source once."""
    candidates = {}
    repositories = []
    for path in sorted(ROOT.iterdir()):
        if path.name in EXCLUDED_DIRS or path.name in {'pro5k_dsv4', 'deepseek-v4-flash'} or path.resolve() == destination:
            continue
        if path.is_dir() and ((path / '.git').exists() or path.name == 'DeepEP'):
            repositories.append(path)
    for repository in repositories:
        for path in paths_below(repository):
            rel = path.relative_to(repository)
            parts = rel.parts
            in_offline_tree = bool(OFFLINE_DIRS.intersection(parts[:-1]))
            root_document = path.suffix.lower() in {'.md', '.rst', '.adoc', '.sh'}
            if not in_offline_tree and not root_document:
                continue
            candidates[str(path)] = {'reason': 'test_document_benchmark_example_script', 'repository': str(repository)}
    audit_path = Path('/tmp/pro5k_conversation_audit.json')
    references = json.loads(audit_path.read_text()).get('referenced_paths', []) if audit_path.exists() else []
    repository_roots = tuple(str(repository) + '/' for repository in repositories)
    binary_cache_suffixes = {'.pyc', '.pyo', '.so', '.o', '.a', '.pt', '.pth', '.safetensors', '.db', '.sqlite', '.sqlite3'}
    for reference in references:
        raw = reference['path']
        if not raw.startswith(repository_roots) or not reference.get('exists'):
            continue
        path = Path(raw)
        if any(part in EXCLUDED_DIRS for part in path.parts) or path.suffix.lower() in binary_cache_suffixes:
            continue
        if '.bundle' in path.name or not (path.is_file() or path.is_symlink()):
            continue
        info = candidates.setdefault(raw, {'reason': 'conversation_referenced_source'})
        info['conversation_occurrences'] = reference.get('occurrences', 0)
        info['session_ids'] = reference.get('session_ids', [])
    seen = {}
    entries = []
    total_source_bytes = 0
    for raw, info in sorted(candidates.items()):
        path = Path(raw)
        if path.is_symlink():
            payload = ('symlink:' + os.readlink(path)).encode()
        else:
            payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        total_source_bytes += len(payload)
        if digest not in seen:
            suffix = ''.join(path.suffixes[-2:]) if path.suffix.lower() == '.gz' else path.suffix
            record = copy_archive(path, destination / 'source/referenced_files/blobs' / digest[:2] / (digest + suffix))
            annotate_original_format(path, record)
            records.append(record)
            seen[digest] = {'outputs': [{'path': str(Path(output['path']).relative_to(destination)), 'bytes': output['bytes'], 'sha256': output['sha256']} for output in record['outputs']], 'source_bytes': len(payload), 'compression': record.get('compression')}
        entries.append(dict(info, source_path=raw, source_sha256=digest, source_mode=oct(path.lstat().st_mode & 0o777), **seen[digest]))
    index = {'mapping': 'source_path -> archived deduplicated blob outputs', 'path_count': len(entries), 'unique_blob_count': len(seen), 'total_source_bytes': total_source_bytes, 'unique_source_bytes': sum(item['source_bytes'] for item in seen.values()), 'entries': entries}
    write_json(index, destination / 'metadata/workspace/offline_file_index.json', records, 'workspace-tests-documents-and-conversation-source:content-deduplicated-index')
    write_generated('Original absolute paths map to content-addressed blobs in metadata/workspace/offline_file_index.json (possibly .gz). Blobs retain source suffixes for searching. Restore a source path by reading its output, decompressing gzip if requested, and applying source_mode. Credentials are redacted in text; source_sha256 describes original content and output SHA256 describes archived content.\n', destination / 'source/referenced_files/README.txt', records, 'generated:offline-reference-restore-instructions')
    return {key: index[key] for key in ['path_count', 'unique_blob_count', 'total_source_bytes', 'unique_source_bytes']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--refresh-source', action='store_true', help='Refresh source snapshots/index only using current sanitizer, preserving completed workspace/legacy records.')
    args = parser.parse_args()
    destination = args.destination.resolve()
    if destination == ROOT:
        raise SystemExit('Destination must be a dedicated archive directory.')
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    manifest_path = destination / 'metadata/workspace/export_manifest.json'
    if args.refresh_source:
        previous = json.loads(manifest_path.read_text())
        for record in previous['records']:
            outputs = [o['path'] for o in record.get('outputs', [])]
            if any(path.startswith('source/') or path.startswith('metadata/workspace/') or path.startswith('workspace/llm-benchmark-online/') for path in outputs):
                continue
            if any(record.get('redactions', {}).values()) and not record.get('generated') and len(outputs) == 1 and Path(record['source']).is_file():
                target = destination / outputs[0]
                if record.get('compression') == 'gzip' and not record['source'].endswith('.gz') and target.name.endswith('.gz'):
                    target = Path(str(target)[:-3])
                record = copy_archive(Path(record['source']), target)
            if not record.get('generated') and Path(record['source']).is_file() and ('/session/' in record['source'] or record['source'].endswith('.jsonl')):
                annotate_original_format(Path(record['source']), record)
            records.append(record)
    before = COUNTS.copy()
    trees = [
        ('workspace/document', 'workspace/document'),
        ('workspace/graph', 'workspace/graph'),
        ('pd-dcl-6565881-launch', 'workspace/launch/pd-dcl-6565881-launch'),
        ('deepseek-v4-flash', 'legacy_archive'),
        ('llm-benchmark-online', 'workspace/llm-benchmark-online'),
        ('DeepEP', 'source/dependencies/DeepEP'),
    ]
    if not args.refresh_source:
        for src, dst in trees:
            export_tree(ROOT / src, destination / dst, records)
        records.append(copy_archive(ROOT / 'constraints.txt', destination / 'workspace/constraints.txt'))
    else:
        export_tree(ROOT / 'DeepEP', destination / 'source/dependencies/DeepEP', records)
        export_tree(ROOT / 'llm-benchmark-online', destination / 'workspace/llm-benchmark-online', records)
    source_states = []
    for repo in sorted(ROOT.iterdir()):
        if repo.name in EXCLUDED_DIRS or repo.name == 'pro5k_dsv4' or repo.resolve() == destination or not repo.is_dir() or not (repo / '.git').exists():
            continue
        try:
            source_states.append(export_source_state(repo, destination, records))
        except RuntimeError as error:
            raise RuntimeError(f'Source state export failed for {repo.name}') from error
    baseline_deltas = export_baseline_deltas(destination, records)
    offline_statistics = export_offline_references(destination, records)
    audit = Path('/tmp/pro5k_workspace_audit.json')
    if audit.exists():
        write_json(json.loads(audit.read_text()), destination / 'metadata/workspace/audit.json', records, str(audit))
    records.append(copy_archive(Path(__file__), destination / 'metadata/workspace/export_workspace.py'))
    records.append(copy_archive(Path('/tmp/pro5k_archive_common.py'), destination / 'metadata/workspace/pro5k_archive_common.py'))
    if Path('/tmp/pro5k_verify_source_roundtrip.py').exists():
        records.append(copy_archive(Path('/tmp/pro5k_verify_source_roundtrip.py'), destination / 'metadata/workspace/verify_source_roundtrip.py'))
    format_exceptions = []
    for record in records:
        if record.get('source_format_exception'):
            for output in record['outputs']:
                path = Path(output['path'])
                output_path = str(path.relative_to(destination)) if path.is_absolute() else str(path)
                info = record['source_format_exception']
                format_exceptions.append({'path': output_path, 'sha256': output['sha256'], 'kind': info['format'], 'lines': info.get('lines', []), 'reason': info['reason'], 'source': record['source']})
    write_json(format_exceptions, destination / 'metadata/workspace/known_source_format_exceptions.json', records, 'generated:original-fixture-format-exceptions')
    for record in records:
        if any(Path(output['path']).is_absolute() for output in record.get('outputs', [])):
            relative_outputs(record, destination)
    manifest = {
        'generated_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'workspace_root': str(ROOT), 'records': records,
        'source_state_count': len(source_states), 'private_baseline_deltas': [{'from': x['archived_base_commit'], 'to': x['target_base_commit']} for x in baseline_deltas], 'offline_reference_statistics': offline_statistics,
        'redactions': dict(COUNTS - before),
        'opaque_git_bundles': [r['source'] for r in records if r.get('opaque_git_bundle')],
        'known_source_format_exceptions': [{'source': r['source'], 'outputs': r['outputs'], **r['source_format_exception']} for r in records if r.get('source_format_exception')],
        'exclusions': sorted(EXCLUDED_DIRS) + ['models/weights', 'runtime packages', 'credentials/authentication files', 'SQLite temporary sidecars: *-wal, *-shm'],
    }
    manifest_path = destination / 'metadata/workspace/export_manifest.json'
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(sanitize_obj(manifest), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'manifest': str(manifest_path), 'files': len(records), 'repositories': len(source_states), 'redactions': dict(COUNTS - before)}, ensure_ascii=False))

if __name__ == '__main__':
    main()
