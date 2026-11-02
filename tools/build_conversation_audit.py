#!/usr/bin/env python3
"""Read-only inventory of Codex history; never reads authentication files."""
import collections
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3

SNAPSHOT = Path('/tmp/pro5k_conversation_snapshot.json')
OUT = Path('/tmp/pro5k_conversation_audit.json')
snapshot = json.loads(SNAPSHOT.read_text())
cutoff = snapshot['cutoff_utc']
path_re = re.compile(r'(?<![A-Za-z0-9])/(?:sgl-workspace|tmp|root)(?:/[^\s\x00"\'`<>|;,()\\\[\]{}]*)?')
secret_patterns = {
    'github_credential': re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b'),
    'aws_access_key_id': re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    'openai_style_key': re.compile(r'\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}\b'),
    'huggingface_token': re.compile(r'\bhf_[A-Za-z0-9]{20,}\b'),
    'private_key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'credential_assignment': re.compile(r'(?im)\b(?:AWS_SECRET_ACCESS_KEY|GITHUB_TOKEN|GH_TOKEN|OPENAI_API_KEY|ANTHROPIC_API_KEY|HF_TOKEN|access_token|refresh_token)\s*[=:]\s*["\']?([^\s"\',;]{12,})'),
}

def safe_excerpt(text, size=260):
    for pat in secret_patterns.values():
        text = pat.sub('[REDACTED]', text)
    return text[:size]

def strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from strings(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from strings(value)

def text_content(payload):
    result = []
    for block in payload.get('content', []):
        if isinstance(block, dict) and isinstance(block.get('text'), str):
            result.append(block['text'])
    return '\n'.join(result)

def substantive(text):
    stripped = text.strip()
    return stripped and not stripped.startswith(('<environment_context>', '<permissions instructions>', '<turn_aborted>', '<collaboration_mode>', '<skills_instructions>'))

def normalized_path(value):
    value = value.rstrip('.:。；，！？')
    value = re.sub(r':(?:L)?\d+(?::\d+)?(?:-L?\d+)?$', '', value)
    return value.rstrip('/') or '/'

session_files = [f for f in snapshot['files'] if '/sessions/' in f['path'] or '/archived_sessions/' in f['path']]
sessions = []
path_counts = collections.Counter()
path_sessions = collections.defaultdict(set)
path_user_counts = collections.Counter()
topics = {
    'DeepSeek V4 / DSV4 / Flash': r'deepseek|dsv4|v4.flash',
    'split_paged / kernel optimization': r'split.?paged|kernel|cuda|triton|tilelang',
    'SGLang / correctness / regression': r'sglang|regression|accuracy|正确|回归|精度',
    'CP / MLA / DCP / prefill / decode': r'\bmla\b|\bdcp\b|\bcp\b|prefill|decode|吸收',
    'SWA / eviction / cache': r'\bswa\b|evict|paged|kv.?cache',
    'benchmark / performance / profiling': r'benchmark|performance|profile|latency|throughput|性能|测试',
    'MoE / MegaMoE / EP': r'megamoe|moe|expert|\bep\b',
    'GLM model weights': r'glm|权重',
    'Codex / skills / archive': r'codex|skill|同步|对话',
}
compiled_topics = {k:re.compile(v,re.I) for k,v in topics.items()}
all_secret_counts = collections.Counter()
for file in session_files:
    path = Path(file['path'])
    with path.open('rb') as handle:
        data = handle.read(file['size'])
    types = collections.Counter()
    roles = collections.Counter()
    user_messages = []
    records = []
    meta = None
    timestamps = []
    errors = 0
    local_paths = collections.Counter()
    local_secret_counts = collections.Counter()
    topic_counts = collections.Counter()
    for lineno, line in enumerate(data.splitlines(), 1):
        try:
            record = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            errors += 1
            continue
        payload = record.get('payload', {})
        kind = record.get('type', '')
        types[kind] += 1
        if record.get('timestamp'):
            timestamps.append(record['timestamp'])
        if kind == 'session_meta' and meta is None:
            meta = {k:payload.get(k) for k in ['id','session_id','timestamp','cwd','originator','cli_version','source','thread_source','forked_from_id','parent_thread_id','agent_nickname','agent_path','subagent_history_start_ordinal','history_mode','history_base'] if k in payload}
        if kind == 'response_item' and isinstance(payload,dict) and payload.get('type')=='message':
            role = payload.get('role','')
            roles[role] += 1
            content = text_content(payload)
            if role == 'user' and substantive(content):
                user_messages.append({'line':lineno,'timestamp':record.get('timestamp'),'excerpt':safe_excerpt(content)})
                for topic, pattern in compiled_topics.items():
                    if pattern.search(content):
                        topic_counts[topic] += 1
                for matched in path_re.findall(content):
                    path_user_counts[normalized_path(matched)] += 1
        for text in strings(payload):
            for matched in path_re.findall(text):
                local_paths[normalized_path(matched)] += 1
            for label, pattern in secret_patterns.items():
                local_secret_counts[label] += len(pattern.findall(text))
    sid = (meta or {}).get('id',path.stem)
    for value,n in local_paths.items():
        path_counts[value] += n
        path_sessions[value].add(sid)
    all_secret_counts.update(local_secret_counts)
    sessions.append({
        'source_path':str(path),'snapshot_bytes':file['size'],'snapshot_mtime_ns':file['mtime_ns'],
        'sha256_snapshot_bytes':hashlib.sha256(data).hexdigest(),'metadata':meta,
        'record_count':sum(types.values()),'record_types':dict(types),'message_roles':dict(roles),
        'first_record_timestamp':min(timestamps) if timestamps else None,
        'last_record_timestamp':max(timestamps) if timestamps else None,
        'invalid_or_incomplete_lines':errors,'ends_with_newline':data.endswith(b'\n'),
        'substantive_user_message_count':len(user_messages),'representative_user_messages':user_messages[:12]+user_messages[-8:] if len(user_messages)>20 else user_messages,
        'topic_message_counts':dict(topic_counts),'referenced_path_count':len(local_paths),
        'top_referenced_paths':local_paths.most_common(50),'potential_secret_pattern_counts':dict(local_secret_counts),
        'archive_task_thread':str(path).startswith('/root/.codex/sessions/2026/11/02/'),
    })

dbs=[]
thread_metadata=[]
for dbfile in [f for f in snapshot['files'] if f['path'].endswith('.sqlite')]:
    path=dbfile['path']
    con=sqlite3.connect('file:'+path+'?mode=ro',uri=True)
    con.row_factory=sqlite3.Row
    con.execute('BEGIN')
    at=datetime.datetime.now(datetime.timezone.utc).isoformat()
    tables=[]
    for row in con.execute("SELECT name,sql FROM sqlite_master WHERE type='table' ORDER BY name"):
        name=row['name']
        quoted='"'+name.replace('"','""')+'"'
        count=con.execute('SELECT COUNT(*) FROM '+quoted).fetchone()[0]
        tables.append({'name':name,'create_sql':row['sql'],'row_count':count})
    extra={}
    if path.endswith('state_5.sqlite'):
        for row in con.execute('SELECT id,title,rollout_path,created_at,updated_at,cwd,archived,source,agent_path FROM threads ORDER BY created_at'):
            value=dict(row)
            value['rollout_exists_now']=Path(value['rollout_path']).is_file()
            value['rollout_in_byte_snapshot']=any(f['path']==value['rollout_path'] for f in session_files)
            value['title']=safe_excerpt(value['title'],1000)
            thread_metadata.append(value)
    if path.endswith('thread_history_1.sqlite'):
        extra['turns_by_thread']=[dict(row) for row in con.execute('SELECT thread_id,COUNT(*) AS turn_count FROM thread_turns GROUP BY thread_id ORDER BY thread_id')]
        extra['items_by_thread']=[dict(row) for row in con.execute('SELECT thread_id,COUNT(*) AS item_count FROM thread_items GROUP BY thread_id ORDER BY thread_id')]
    dbs.append({'path':path,'read_transaction_utc':at,'tables':tables,**extra})
    con.rollback();con.close()

paths=[]
for path,count in path_counts.most_common():
    try:
        p=Path(path)
        exists=p.exists()
        isfile=p.is_file() if exists else False
        size=p.stat().st_size if isfile else None
        kind='file' if isfile else 'directory' if exists else 'missing_or_pattern'
    except (OSError,ValueError):
        exists=False;size=None;kind='unreadable_or_invalid'
    paths.append({'path':path,'occurrences':count,'user_message_occurrences':path_user_counts[path],
                  'session_ids':sorted(path_sessions[path]),'exists':exists,'kind':kind,'bytes':size,
                  'likely_document_test_result':bool(re.search(r'(?i)(test|bench|result|report|docs?|notes?|learn|profile|analysis|\.md$|\.rst$|\.txt$|\.json$|\.csv$|\.log$|\.patch$|\.diff$|\.png$|\.svg$|\.pdf$)',path))})

attachments=[]
for p in Path('/root/.codex/attachments').rglob('*'):
    if p.is_file():
        s=p.stat();attachments.append({'path':str(p),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns})
indexfiles=[f for f in snapshot['files'] if f['path'].endswith('/session_index.jsonl')]
index_records=[]
for f in indexfiles:
    with open(f['path'],'rb') as h:data=h.read(f['size'])
    for line in data.splitlines():
        try: index_records.append(json.loads(line))
        except ValueError:pass

result={
    'audit_version':1,'generated_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'jsonl_cutoff_utc':cutoff,'byte_snapshot_file':str(SNAPSHOT),'snapshot':snapshot,
    'source_discovery':{
        'user_roots_checked':['/root','/home/ubuntu','/nonexistent'],
        'other_home_codex_locations':'Escalated read-only inspection of user homes, .config, .local/share found no other Codex directories; no authentication files read.',
        'empty_codex_roots':['/tmp/.codex','/sgl-workspace/.codex'],
        'non_conversation_fixture_jsonl_excluded':[f['path'] for f in snapshot['files'] if '.tmp/plugins/' in f['path'] and f['path'].endswith('.jsonl')],
    },
    'summary':{'rollout_file_count':len(sessions),'unique_rollout_thread_ids':len({s['metadata']['id'] for s in sessions if s['metadata']}),
               'archived_rollout_files':sum('/archived_sessions/' in s['source_path'] for s in sessions),
               'rollout_snapshot_bytes':sum(s['snapshot_bytes'] for s in sessions),'rollout_records':sum(s['record_count'] for s in sessions),
               'referenced_paths':len(paths),'existing_referenced_paths':sum(p['exists'] for p in paths),
               'attachment_files':len(attachments),'attachment_bytes':sum(p['bytes'] for p in attachments),
               'session_index_records':len(index_records),'potential_secret_pattern_counts':dict(all_secret_counts)},
    'sessions':sessions,'session_index_records':index_records,'database_inventories':dbs,'database_thread_metadata':thread_metadata,
    'referenced_paths':paths,'attachments':attachments,
    'method_notes':[
        'JSONL files read only up to per-file snapshot byte limits. Invalid final partial lines are counted and should be excluded from JSONL export.',
        'Path counts include tool outputs, instructions, compaction and inherited subagent history; counts are raw occurrences and are not independent task counts.',
        'First session_meta identifies the physical thread. Later inherited session_meta records may reference its parent and must not replace that identity.',
        'Multiple rollout files can share one thread ID, including resumed September 8/9 history. Export every segment, not only threads.rollout_path.',
        'SQLite DBs are derived conversation stores and are inspected in read-only transactions at separately recorded times. Raw DB files contain unrelated runtime tables and should not be uploaded wholesale.',
        'Secret-pattern counts are potential matches, including placeholders and quoted examples. Actual credentials must be redacted before writing archive artifacts.',
    ],
    'export_recommendation':{
        'include':['sessions/**/*.jsonl within byte limits','archived_sessions/*.jsonl within byte limits','session_index.jsonl within byte limits',
                   'per-thread readable user/assistant Markdown','read-only sanitized JSONL projections of conversation SQLite tables','referenced non-secret attachments'],
        'exclude':['auth.json','config.toml','installation_id','logs_2.sqlite','queue_1.sqlite','shell_snapshots','ipc sockets','skills/plugin caches'],
        'database_tables':{'state_5.sqlite':['threads','thread_spawn_edges','thread_sections','projects','project_roots','thread_attachments'],
                           'thread_history_1.sqlite':['thread_turns','thread_items','thread_history_projection_state','thread_realtime_items'],
                           'goals_1.sqlite':['thread_goals','thread_goal_continuation_deferrals'],
                           'memories_1.sqlite':['stage1_outputs','consolidation_progress']},
        'risks':['Active session continues after snapshot; never repeatedly reread or append its growing archive output.',
                 'Session tool outputs can embed credentials even though auth/config files are excluded.',
                 'Missing historical artifact paths may be temporary files already deleted; retain a missing-path manifest.',
                 'Large model weights, caches, virtual environments, binaries and upstream checkouts should be inventoried rather than indiscriminately copied.']
    }
}
OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'output':str(OUT),'summary':result['summary']},ensure_ascii=False))
