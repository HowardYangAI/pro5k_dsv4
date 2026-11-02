#!/usr/bin/env python3
"""Archive temporary research/test artifacts and audit conversation references.

Run after pro5k_artifact_audit.json and pro5k_conversation_audit.json exist.
Only archival destinations are changed; original files remain untouched.
"""
from pathlib import Path
import argparse
import collections
import hashlib
import json
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pro5k_archive_common import copy_archive, sanitize_obj, sanitize_text, COUNTS

CLEAN_CLONES = {
    '/tmp/sglang-dcl-check': '48886803b94596884d737132a381bd5124ea1cfd',
    '/tmp/tmp.muvLHigfAG/fa': 'ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820',
    '/tmp/tmp.iX6EANZMPS/fla': '35dceaee5408e69a555fec34cb215c93c375dabe',
    '/tmp/tmp.KHuQydmdSb/kda': '7afb9f454f160a6c4bbc0999beca0a8c40a38934',
}
INSTALL_PREFIXES = (
    '/root/.cache', '/root/.vscode-server', '/root/.local/bin',
    '/root/.codex/skills', '/root/.codex/vendor_imports',
    '/root/.codex/.tmp', '/root/.claude', '/root/.copilot',
)
CONFIG_PREFIXES = ('/root/.ssh', '/root/.aws', '/root/.config', '/root/.cc-switch')
IMPORTANT_MISSING = {
    '/tmp/ShareGPT_V3_unfiltered_cleaned_split.json': 'Benchmark input dataset referenced in PD testing; not present at archive time.',
    '/tmp/multiturn_run_130.jsonl': 'Referenced accuracy/multiturn run output; not present at archive time.',
    '/tmp/multiturn_run_131.jsonl': 'Referenced accuracy/multiturn run output; not present at archive time.',
    '/tmp/multiturn_run_132.jsonl': 'Referenced accuracy/multiturn run output; not present at archive time.',
    '/tmp/h3-dp-0.mp4': 'Referenced MiniMax H3 video output; not present at archive time.',
    '/tmp/h3-dp-1.mp4': 'Referenced MiniMax H3 video output; not present at archive time.',
    '/tmp/decode_health.out': 'Referenced decode health response; not present at archive time.',
    '/tmp/decode_health_body': 'Referenced decode health response; not present at archive time.',
    '/tmp/fetch.log': 'Referenced download log; not present at archive time.',
}


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize_obj(obj), ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-root', default='/sgl-workspace/pro5k_dsv4')
    parser.add_argument('--audit', default='/tmp/pro5k_artifact_audit.json')
    parser.add_argument('--audit-report', default='/tmp/pro5k_artifact_audit_report.json')
    parser.add_argument('--conversation-audit', default='/tmp/pro5k_conversation_audit.json')
    args = parser.parse_args()
    root = Path(args.archive_root)
    meta = root / 'metadata/artifacts'
    manifest = []
    candidates = json.loads(Path(args.audit).read_text())
    selected = {x['source']: x for x in candidates if x['source'].startswith('/tmp/')}
    supplements = []
    # These empty output files were omitted from the first size-based audit,
    # but their presence is evidence that the probe produced no response.
    for source in ['/tmp/router_help.err', '/tmp/prefill_health.out']:
        p = Path(source)
        if p.is_file() and source not in selected:
            entry = {'source': source, 'size': p.stat().st_size, 'category': 'test_result',
                     'reason': 'Conversation-referenced empty health/error output retained as test evidence',
                     'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
            selected[source] = entry
            supplements.append(entry)
    missing = []
    for source, info in sorted(selected.items()):
        src = Path(source)
        if not src.is_file() and not src.is_symlink():
            missing.append({'source': source, 'reason': 'Selected artifact disappeared before copy'})
            continue
        target = root / 'artifacts/tmp' / src.relative_to('/tmp')
        result = copy_archive(src, target)
        result.update(category=info['category'], reason=info['reason'])
        manifest.append(result)
    referenced = []
    reference_file = Path(args.conversation_audit)
    if reference_file.exists():
        audit = json.loads(reference_file.read_text())
        for item in audit['referenced_paths']:
            original_path = item['path']
            if original_path.startswith('/sgl-workspace'):
                continue  # Other exporter owns all workspace documents/tests.
            path = re.sub(r':\d+(?::.*)?$', '', original_path)
            entry = {'source': original_path, 'normalized_source': path,
                     'occurrences': item['occurrences'], 'session_ids': item['session_ids']}
            p = Path(path)
            clone = next((prefix for prefix in CLEAN_CLONES if path == prefix or path.startswith(prefix + '/')), None)
            if path in selected:
                entry['status'] = 'archived'
            elif clone:
                entry.update(status='excluded_clean_public_reference_checkout', commit=CLEAN_CLONES[clone],
                             reason='Clean public checkout; referenced source/tests/docs are reproducible from the recorded commit.')
            elif path.startswith('/root/.codex/attachments') or path.startswith('/root/.codex/sessions') or path.startswith('/root/.codex/archived_sessions'):
                entry['status'] = 'owned_by_conversation_exporter'
            elif path.startswith(CONFIG_PREFIXES) or path.startswith('/root/.codex') or path in {'/root/.bash_history', '/root/.gitconfig'}:
                entry.update(status='excluded_configuration_or_credential_material', reason='Configuration/history is outside the requested document/test archive.')
            elif path.startswith(INSTALL_PREFIXES) or any(s in path for s in ['/pip-', '/pip_', '/cg_extract/', '/playwright_']):
                entry['status'] = 'excluded_installation_or_runtime_cache'
            elif p.is_dir():
                entry['status'] = 'directory_reference_only'
            elif p.exists():
                entry['status'] = 'excluded_binary_tool_or_instruction_material'
                if p.is_file():
                    entry['bytes'] = p.stat().st_size
                if path.endswith('.bundle'):
                    entry['reason'] = 'Duplicate Git bundle; source checkouts/patches and commit metadata are archived separately.'
            elif path in IMPORTANT_MISSING:
                entry.update(status='missing_important_artifact', reason=IMPORTANT_MISSING[path])
                if not any(x['source'] == path for x in missing):
                    missing.append({'source': path, 'reason': IMPORTANT_MISSING[path], 'occurrences': item['occurrences']})
            else:
                entry['status'] = 'missing_ephemeral_or_pattern_reference'
            referenced.append(entry)
    report = json.loads(Path(args.audit_report).read_text())
    dump(meta / 'tmp_manifest.json', manifest)
    dump(meta / 'supplemental_candidates.json', supplements)
    dump(meta / 'referenced_path_coverage.json', referenced)
    dump(meta / 'missing_artifacts.json', missing)
    dump(meta / 'exclusions_and_risks.json', report)
    metadata_copies = []
    for source, name in [
        (args.audit, 'initial_candidates.json'),
        (args.audit_report, 'initial_audit_report.json'),
        (str(Path(__file__)), 'pro5k_export_artifacts.py'),
        (str(Path(__file__).with_name('pro5k_archive_common.py')), 'pro5k_archive_common.py'),
    ]:
        metadata_copies.append(copy_archive(source, meta / name))
    summary = {
        'selected_initial_tmp_files': len([x for x in candidates if x['source'].startswith('/tmp/')]),
        'supplemental_files': len(supplements), 'archived_files': len(manifest),
        'source_bytes': sum(x['source_bytes'] for x in manifest),
        'stored_bytes': sum(y['bytes'] for x in manifest for y in x['outputs']),
        'gzip_files': sum(x['compression'] == 'gzip' for x in manifest),
        'redactions': dict(COUNTS), 'referenced_path_statuses': dict(collections.Counter(x['status'] for x in referenced)),
        'important_missing_artifacts': len(missing), 'metadata_copies': metadata_copies,
    }
    dump(meta / 'summary.json', summary)
    print(json.dumps(sanitize_obj({k: v for k, v in summary.items() if k != 'metadata_copies'}), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
