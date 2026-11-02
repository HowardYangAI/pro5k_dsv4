#!/usr/bin/env python3
"""Restore one repository's archived tests/docs/scripts into a NEW directory."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path


def read_index(root):
    path = root / 'metadata/workspace/offline_file_index.json'
    if path.exists():
        return json.loads(path.read_text())['entries']
    path = Path(str(path) + '.gz')
    with gzip.open(path, 'rt') as handle:
        return json.load(handle)['entries']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--list', action='store_true', help='List repository names')
    parser.add_argument('--repo', help='Repository name shown by --list')
    parser.add_argument('--dest', type=Path, help='New destination directory (must not exist)')
    args = parser.parse_args()
    root = args.archive_root.resolve()
    entries = read_index(root)
    prefix = '/sgl-workspace/'
    repositories = sorted({e['source_path'][len(prefix):].split('/')[0] for e in entries if e['source_path'].startswith(prefix)})
    if args.list:
        print('\n'.join(repositories))
        return
    if args.repo not in repositories or args.dest is None:
        parser.error('Use --list, or provide a listed --repo and a new --dest.')
    if args.dest.exists() or args.dest.is_symlink():
        parser.error('--dest must be a new directory; existing paths are never overwritten.')
    destination = args.dest.resolve()
    source_prefix = prefix + args.repo + '/'
    selected = [e for e in entries if e['source_path'].startswith(source_prefix)]
    planned = []
    for entry in selected:
        relative = Path(entry['source_path'][len(source_prefix):])
        target = destination / relative
        if relative.is_absolute() or '..' in relative.parts or not target.is_relative_to(destination):
            raise ValueError('Unsafe source path in archive index')
        for output in entry['outputs']:
            blob = (root / output['path']).resolve()
            if not blob.is_relative_to(root):
                raise ValueError('Unsafe output path in archive index')
            data = blob.read_bytes()
            if len(data) != output['bytes'] or hashlib.sha256(data).hexdigest() != output['sha256']:
                raise ValueError('Archive content hash mismatch: ' + output['path'])
        planned.append((entry, target))
    destination.mkdir(parents=True)
    restored = 0
    symlink_records = 0
    for entry, target in planned:
        data = b''.join((root / output['path']).read_bytes() for output in entry['outputs'])
        if entry.get('compression') == 'gzip':
            data = gzip.decompress(data)
        if entry['outputs'][0]['path'].endswith('.symlink.json'):
            target = Path(str(target) + '.symlink.json')
            symlink_records += 1
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as handle:
            handle.write(data)
        if entry.get('source_mode'):
            os.chmod(target, int(entry['source_mode'], 8) & 0o777)
        restored += 1
    print(json.dumps({'repository': args.repo, 'destination': str(destination), 'restored_files': restored,
                      'symlinks_preserved_as_metadata': symlink_records}, ensure_ascii=False))


if __name__ == '__main__':
    main()
