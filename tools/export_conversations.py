#!/usr/bin/env python3
"""Export a bounded, sanitized Codex conversation archive.

Usage:
  python /tmp/pro5k_export_conversations.py --dest /path/to/archive/conversations \
    --snapshot /tmp/pro5k_conversation_snapshot.json \
    --common /tmp/pro5k_archive_common.py

The shared module must expose sanitize_text(str)->str and sanitize_obj(obj)->obj.
Authentication/configuration files are excluded from export. The shared redactor
may load local credentials into memory solely to recognize and remove them. No
unsanitized source content is ever written to the destination.
"""
import argparse
import base64
import collections
import contextlib
import datetime
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile

DB_TABLES = {
    'state_5.sqlite': ['threads','thread_spawn_edges','thread_sections','thread_dynamic_tools',
                       'projects','project_roots','thread_attachments'],
    'thread_history_1.sqlite': ['thread_turns','thread_items','thread_history_projection_state','thread_realtime_items'],
    'goals_1.sqlite': ['thread_goals','thread_goal_continuation_deferrals'],
    'memories_1.sqlite': ['stage1_outputs','consolidation_progress'],
}

def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def iso_ms(value):
    return int(datetime.datetime.fromisoformat(value.replace('Z','+00:00')).timestamp()*1000)

def import_common(path):
    spec=importlib.util.spec_from_file_location('pro5k_archive_common',path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    if not callable(getattr(module,'sanitize_text',None)) or not callable(getattr(module,'sanitize_obj',None)):
        raise TypeError('Shared redactor must export sanitize_text and sanitize_obj')
    return module

def hash_file(path):
    h=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()

def jsonable(obj):
    if isinstance(obj,bytes):
        return {'__type__':'bytes','base64':base64.b64encode(obj).decode('ascii')}
    if isinstance(obj,dict):
        return {key:jsonable(value) for key,value in obj.items()}
    if isinstance(obj,(tuple,list)):
        return [jsonable(value) for value in obj]
    return obj

def emit_json(path,obj,common):
    path.parent.mkdir(parents=True,exist_ok=True)
    clean=common.sanitize_obj(jsonable(obj))
    path.write_text(json.dumps(clean,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def gzip_rows(path,rows,common):
    path.parent.mkdir(parents=True,exist_ok=True)
    count=0
    with path.open('wb') as raw:
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0,compresslevel=9) as zipped:
            for row in rows:
                clean=common.sanitize_obj(jsonable(row))
                encoded=json.dumps(clean,ensure_ascii=False,separators=(',',':'))
                zipped.write((encoded+'\n').encode('utf-8'))
                count+=1
    return {'records':count,'bytes':path.stat().st_size,'sha256':hash_file(path)}

def message_text(payload):
    values=[]
    for block in payload.get('content',[]):
        if isinstance(block,str):
            values.append(block)
        elif isinstance(block,dict):
            if isinstance(block.get('text'),str):
                values.append(block['text'])
            elif block.get('type') in ('image','input_image','image_url'):
                values.append('[图片内容见原始 JSONL 记录及附件归档。]')
            elif block.get('type') in ('audio','input_audio','output_audio'):
                values.append('[音频内容见原始 JSONL 记录。]')
    return '\n\n'.join(values)

def title_text(text):
    text=' '.join(text.split())
    return text[:160]

def select_rows(con,table,allowed_thread_ids,cutoff_ms):
    quoted='"'+table.replace('"','""')+'"'
    for row in con.execute('SELECT * FROM '+quoted):
        value=dict(row)
        tid=value.get('thread_id')
        if table=='threads' and value.get('id') not in allowed_thread_ids:
            continue
        if tid is not None and tid not in allowed_thread_ids:
            continue
        if table=='thread_spawn_edges' and (value['parent_thread_id'] not in allowed_thread_ids or value['child_thread_id'] not in allowed_thread_ids):
            continue
        # Bound newly created DB records to the initial JSONL cutoff. Existing
        # rows can have later updates, explicitly recorded in the manifest.
        created=value.get('created_at_ms')
        if created is not None and created>cutoff_ms:
            continue
        if table=='thread_turns' and value.get('started_at') is not None and value['started_at']*1000>cutoff_ms:
            continue
        yield value

@contextlib.contextmanager
def readonly_database(source):
    temporary=None
    con=None
    method='direct read-only transaction; WAL included'
    try:
        try:
            con=sqlite3.connect(source.as_uri()+'?mode=ro',uri=True)
            con.row_factory=sqlite3.Row
            con.execute('PRAGMA query_only=ON')
            con.execute('BEGIN')
            con.execute('SELECT name FROM sqlite_master LIMIT 1').fetchall()
        except sqlite3.OperationalError:
            if con is not None:
                con.close()
            # A read-only source directory cannot create a missing SQLite SHM
            # sidecar. Copy stable DB/WAL/SHM bytes to a private /tmp directory;
            # no raw database bytes are ever copied to the archive destination.
            temporary=tempfile.TemporaryDirectory(prefix='pro5k_sqlite_snapshot_')
            copied=Path(temporary.name)/source.name
            siblings=[source,Path(str(source)+'-wal'),Path(str(source)+'-shm')]
            def signature():
                return [(str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in siblings if p.exists()]
            stable=False
            for _ in range(4):
                before=signature()
                for p,_,_ in before:
                    shutil.copyfile(p,Path(temporary.name)/Path(p).name)
                if signature()==before:
                    stable=True;break
            if not stable:
                raise RuntimeError('Could not capture stable SQLite source bytes: '+str(source))
            con=sqlite3.connect(copied.as_uri()+'?mode=ro',uri=True)
            con.row_factory=sqlite3.Row
            con.execute('PRAGMA query_only=ON')
            con.execute('BEGIN')
            con.execute('SELECT name FROM sqlite_master LIMIT 1').fetchall()
            method='stable DB/WAL/SHM byte copy into private /tmp, source size+mtime verified before/after; read-only transaction on copy'
        yield con,method
    finally:
        if con is not None:
            con.rollback();con.close()
        if temporary is not None:
            temporary.cleanup()

def export_databases(dest,source_root,common,cutoff_ms,known_ids):
    inventories=[]
    threads=[]
    statepath=source_root/'state_5.sqlite'
    allowed=set(known_ids)
    # Read all state metadata once inside a read-only transaction to establish
    # eligible thread identities, including a metadata-only Guardian thread.
    if statepath.is_file():
        with readonly_database(statepath) as (con,_):
            for row in con.execute('SELECT * FROM threads'):
                value=dict(row)
                created=value.get('created_at_ms') or value.get('created_at',0)*1000
                if created<=cutoff_ms:
                    allowed.add(value['id'])
    for dbname,wanted in DB_TABLES.items():
        source=source_root/dbname
        if not source.is_file():
            inventories.append({'source_path':str(source),'missing':True})
            continue
        with readonly_database(source) as (con,method):
            all_tables={r['name']:r['sql'] for r in con.execute("SELECT name,sql FROM sqlite_master WHERE type='table'")}
            record={'source_path':str(source),'read_transaction_utc':utc_now(),'snapshot_method':method,
                    'creation_cutoff_ms':cutoff_ms,'tables':[],
                    'skipped_tables':sorted(set(all_tables)-set(wanted)),
                    'note':'Read-only SQLite transaction includes WAL. Rows created after initial cutoff are omitted. Existing rows may reflect later updates at transaction time.'}
            for table in wanted:
                if table not in all_tables:
                    record['tables'].append({'name':table,'missing':True})
                    continue
                values=list(select_rows(con,table,allowed,cutoff_ms))
                if table=='threads':
                    threads=values
                target=dest/'databases'/dbname.replace('.sqlite','')/(table+'.jsonl.gz')
                stats=gzip_rows(target,values,common)
                total=con.execute('SELECT COUNT(*) FROM "'+table.replace('"','""')+'"').fetchone()[0]
                record['tables'].append({'name':table,'create_sql':all_tables[table],
                                         'source_row_count_at_transaction':total,
                                         'exported_path':str(target.relative_to(dest)),**stats})
        inventories.append(record)
    emit_json(dest/'databases'/'manifest.json',inventories,common)
    return inventories,threads

def export_attachments(dest,source_root,common,audit):
    records=[]
    for item in audit.get('attachments',[]):
        source=Path(item['path'])
        relative=source.relative_to(source_root/'attachments')
        target=dest/'attachments'/relative
        with source.open('rb') as handle:
            original=handle.read(item['bytes'])
        if target.suffix.lower()=='.json':
            try:
                clean=common.sanitize_obj(json.loads(original))
                data=(json.dumps(clean,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
            except (ValueError,UnicodeDecodeError):
                data=common.sanitize_text(original.decode('utf-8')).encode('utf-8')
        else:
            try:
                data=common.sanitize_text(original.decode('utf-8')).encode('utf-8')
            except UnicodeDecodeError:
                data=original
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(data)
        records.append({'source_path':str(source),'snapshot_bytes_limit':item['bytes'],
                        'source_snapshot_sha256':hashlib.sha256(original).hexdigest(),
                        'exported_path':str(target.relative_to(dest)),'bytes':len(data),
                        'sha256':hashlib.sha256(data).hexdigest()})
    emit_json(dest/'attachments'/'manifest.json',records,common)
    return records

def write_total_index(dest,index,common,cutoff):
    emit_json(dest/'index.json',index,common)
    lines=['# Codex 对话归档','',f'JSONL 截止时间（UTC）: {cutoff}',
           '',f'共 {len(index)} 个线程，含本机 Codex 数据及工作区此前保存的历史会话。',
           '','消息 Markdown 用于阅读；`raw/` 保留本机 Codex 脱敏事件，`databases/` 保留会话投影；更早的原始事件位于 [`../legacy_archive/session/`](../legacy_archive/session/)。',
           '','| 线程 | 分段 | 消息 | 说明 |','| --- | ---: | ---: | --- |']
    for item in index:
        title=item['title'].replace('|','\\|').replace('[','\\[').replace(']','\\]')
        note='本次归档快照' if item.get('archive_task_thread') else ''
        if item.get('legacy_workspace_history'):
            note='工作区早期历史'
        if item.get('rollout_missing_from_snapshot'):
            note='仅数据库元数据'
        lines.append(f'| [{title}]({item["readable_path"]}) | {item["rollout_segments"]} | {item["readable_messages"]} | {note} |')
    (dest/'README.md').write_text(common.sanitize_text('\n'.join(lines)+'\n'),encoding='utf-8')

def append_legacy(dest,legacy_root,common,cutoff):
    """Index legacy source JSONL without duplicating workspace agent's raw copies."""
    if not legacy_root.is_dir():
        return []
    index=json.loads((dest/'index.json').read_text())
    by_id={item['thread_id']:item for item in index}
    coverage=[]
    archive_root=dest.parent
    for source in sorted(legacy_root.rglob('rollout-*.jsonl')):
        size=source.stat().st_size
        with source.open('rb') as handle:
            data=handle.read(size)
        meta=None
        messages=[]
        invalid=[]
        count=0
        for lineno,line in enumerate(data.splitlines(),1):
            try:
                record=common.sanitize_obj(json.loads(line))
            except (ValueError,UnicodeDecodeError):
                invalid.append(lineno);continue
            count+=1
            payload=record.get('payload',{})
            if record.get('type')=='session_meta' and meta is None:
                meta=payload
            if record.get('type')=='response_item' and isinstance(payload,dict) and payload.get('type')=='message' and payload.get('role') in ('user','assistant'):
                text=message_text(payload)
                if text.strip():
                    messages.append({'role':payload['role'],'text':common.sanitize_text(text),'line':lineno,'timestamp':record.get('timestamp')})
        meta=meta or {}
        sid=meta.get('id') or meta.get('session_id') or source.stem
        title=sid
        for message in messages:
            if message['role']=='user' and not message['text'].lstrip().startswith('<environment_context>'):
                title=title_text(message['text']);break
        raw_target=archive_root/'legacy_archive'/'session'/'all-sessions'/source.relative_to(legacy_root)
        if size>2*1024*1024:
            raw_target=Path(str(raw_target)+'.gz')
        raw_relative=str(raw_target.relative_to(archive_root))
        thread_path=dest/'threads'/(sid+'.md')
        if sid not in by_id:
            lines=['# '+title,'',f'- Thread ID: `{sid}`','- 来源: 工作区此前保存的 Codex 会话',
                   f'- 源记录数: {count}',f'- 可阅读消息: {len(messages)}','',
                   f'[脱敏原始记录](../../{raw_relative})','','## 对话','']
            for n,message in enumerate(messages,1):
                lines.extend([f'### {n}. {message["role"]} · {message.get("timestamp") or "时间未记录"}',
                              '',f'来源: `{source.name}:{message["line"]}`','',message['text'],''])
            thread_path.write_text(common.sanitize_text('\n'.join(lines)),encoding='utf-8')
            item={'thread_id':sid,'title':title,'readable_path':str(thread_path.relative_to(dest)),
                  'rollout_segments':1,'readable_messages':len(messages),
                  'role_counts':dict(collections.Counter(m['role'] for m in messages)),
                  'archive_task_thread':False,'legacy_workspace_history':True,'rollout_missing_from_snapshot':False,
                  'metadata':{key:meta.get(key) for key in ('id','session_id','timestamp','cwd','originator','source','thread_source','forked_from_id','parent_thread_id','agent_path') if key in meta}}
            index.append(item);by_id[sid]=item
        coverage.append({'source_path':str(source),'category':'legacy_workspace_rollout','thread_id':sid,
                         'snapshot_bytes_limit':size,'snapshot_mtime_ns':source.stat().st_mtime_ns,
                         'source_snapshot_sha256':hashlib.sha256(data).hexdigest(),'record_count':count,
                         'invalid_or_partial_lines_omitted':invalid,
                         'raw_destination':raw_relative,'readable_destination':'conversations/'+by_id[sid]['readable_path'],
                         'raw_copy_owned_by':'workspace agent; no duplicate raw JSONL copied by conversation exporter'})
    primary=legacy_root.parent/'primary-session.jsonl'
    if primary.is_file():
        sha=hash_file(primary)
        duplicate=next((item for item in coverage if item['source_snapshot_sha256']==sha),None)
        coverage.append({'source_path':str(primary),'category':'legacy_primary_duplicate',
                         'source_snapshot_sha256':sha,'bytes':primary.stat().st_size,
                         'duplicate_of':duplicate['source_path'] if duplicate else None,
                         'thread_id':duplicate['thread_id'] if duplicate else None,
                         'readable_destination':duplicate['readable_destination'] if duplicate else None,
                         'raw_destination':'legacy_archive/session/primary-session.jsonl'+('.gz' if primary.stat().st_size>2*1024*1024 else '')})
    write_total_index(dest,sorted(index,key=lambda item:item['thread_id']),common,cutoff)
    manifest=json.loads((dest/'manifest.json').read_text())
    manifest['thread_count']=len(index)
    manifest['readable_message_count']=sum(item['readable_messages'] for item in index)
    manifest['legacy_workspace_rollout_count']=sum(item['category']=='legacy_workspace_rollout' for item in coverage)
    manifest['legacy_primary_duplicate_count']=sum(item['category']=='legacy_primary_duplicate' for item in coverage)
    manifest['source_coverage']='../metadata/conversations/source_coverage.json'
    emit_json(dest/'manifest.json',manifest,common)
    current=[]
    audit_path=Path('/tmp/pro5k_conversation_audit.json')
    audit=json.loads(audit_path.read_text()) if audit_path.is_file() else {}
    sid_by_source={item['source_path']:item['metadata'].get('id') for item in audit.get('sessions',[]) if item.get('metadata')}
    for item in manifest['files']:
        if item['source_path'].endswith('session_index.jsonl'):
            continue
        sid=item.get('thread_id') or sid_by_source.get(item['source_path'])
        current.append({'source_path':item['source_path'],'category':'current_codex_rollout','thread_id':sid,
                        'snapshot_bytes_limit':item['snapshot_bytes_limit'],
                        'source_snapshot_sha256':item['source_snapshot_sha256'],
                        'raw_destination':'conversations/'+item['exported_path'],
                        'readable_destination':'conversations/'+by_id[sid]['readable_path'] if sid in by_id else None})
    all_sources=current+coverage
    report={'generated_at_utc':utc_now(),'jsonl_cutoff_utc':cutoff,'sources':all_sources,
            'summary':{'source_paths':len(all_sources),'current_codex_rollouts':len(current),
                       'legacy_workspace_rollouts':sum(x['category']=='legacy_workspace_rollout' for x in coverage),
                       'duplicate_primary_copies':sum(x['category']=='legacy_primary_duplicate' for x in coverage),
                       'unique_thread_ids_in_rollouts':len({x['thread_id'] for x in all_sources if x.get('thread_id')}),
                       'unique_rollout_snapshot_hashes':len({x['source_snapshot_sha256'] for x in all_sources}),
                       'indexed_threads_including_metadata_only':len(index)},
            'notes':['Workspace legacy raw files are copied by the workspace export; this exporter only adds their readable messages and unified index.',
                     'Legacy primary-session.jsonl is a byte-identical extra copy of one all-sessions rollout.',
                     'Metadata-only DB threads are retained in conversations/index.json and can have no rollout in the initial byte snapshot.']}
    emit_json(archive_root/'metadata'/'conversations'/'source_coverage.json',report,common)
    return report['summary']

def export(snapshot,dest,source_root,common,audit):
    dest.mkdir(parents=True,exist_ok=True)
    cutoff_ms=iso_ms(snapshot['cutoff_utc'])
    grouped=collections.defaultdict(lambda:{'segments':[],'messages':[],'metadata':None})
    manifest={'archive_version':1,'started_at_utc':utc_now(),'jsonl_cutoff_utc':snapshot['cutoff_utc'],
              'files':[],'notes':[
                  'Only pre-enumerated JSONL bytes are read; ongoing archive work is not appended recursively.',
                  'Source JSONL payloads are sanitized before serialization. Files are gzip-compressed with deterministic headers.',
                  'Readable Markdown includes user/assistant messages; tool calls, tool outputs, compaction records, instructions and other events remain in sanitized raw JSONL.',
                  'Some subagent rollouts contain inherited parent history, preserved intentionally.',
                  'Multiple rollout segments sharing one thread ID are grouped without dropping segments.',
                  'Authentication, configuration, runtime logs, queues, shell snapshots and plugin caches are excluded.',
              ]}
    relevant=[f for f in snapshot['files'] if '/sessions/' in f['path'] or '/archived_sessions/' in f['path'] or f['path'].endswith('/session_index.jsonl')]
    for file in sorted(relevant,key=lambda item:item['path']):
        source=Path(file['path'])
        relative=source.relative_to(source_root)
        with source.open('rb') as handle:
            data=handle.read(file['size'])
        records=[]
        invalid=[]
        meta=None
        messages=[]
        for lineno,line in enumerate(data.splitlines(),1):
            try:
                record=json.loads(line)
            except (ValueError,UnicodeDecodeError):
                invalid.append(lineno)
                continue
            clean=common.sanitize_obj(record)
            records.append(clean)
            payload=clean.get('payload',{})
            if clean.get('type')=='session_meta' and meta is None:
                meta=payload
            if clean.get('type')=='response_item' and isinstance(payload,dict) and payload.get('type')=='message' and payload.get('role') in ('user','assistant'):
                content=message_text(payload)
                if content.strip():
                    messages.append({'role':payload['role'],'text':common.sanitize_text(content),
                                     'timestamp':clean.get('timestamp'),'source_segment':str(relative),
                                     'source_line':lineno,'ordinal':clean.get('ordinal')})
        target=dest/'raw'/relative.parent/(relative.name+'.gz')
        stats=gzip_rows(target,records,common)
        entry={'source_path':str(source),'snapshot_bytes_limit':file['size'],
               'bytes_read':len(data),'snapshot_mtime_ns':file.get('mtime_ns'),
               'source_snapshot_sha256':hashlib.sha256(data).hexdigest(),
               'exported_path':str(target.relative_to(dest)),'invalid_or_partial_lines_omitted':invalid,
               **stats}
        if meta:
            entry['thread_id']=meta.get('id') or meta.get('session_id') or source.stem
        manifest['files'].append(entry)
        if meta:
            sid=meta.get('id') or meta.get('session_id') or source.stem
            group=grouped[sid]
            if group['metadata'] is None:
                group['metadata']={key:meta.get(key) for key in (
                    'id','session_id','timestamp','cwd','originator','cli_version','source','thread_source',
                    'forked_from_id','parent_thread_id','agent_nickname','agent_path','history_mode','history_base'
                ) if key in meta}
            group['segments'].append(entry)
            group['messages'].extend(messages)
    db_manifest,dbthreads=export_databases(dest,source_root,common,cutoff_ms,set(grouped))
    db_by_id={r['id']:r for r in dbthreads}
    # Retain metadata-only conversations and clearly explain missing rollouts.
    for sid,row in db_by_id.items():
        if sid not in grouped:
            grouped[sid]['metadata']={k:row.get(k) for k in ('id','cwd','source','agent_path')}
    index=[]
    for sid,group in sorted(grouped.items()):
        row=db_by_id.get(sid,{})
        title=title_text(common.sanitize_text(row.get('title') or ''))
        if not title:
            for message in group['messages']:
                if message['role']=='user' and not message['text'].lstrip().startswith('<environment_context>'):
                    title=title_text(message['text']);break
        title=title or sid
        thread_path=dest/'threads'/(sid+'.md')
        thread_path.parent.mkdir(parents=True,exist_ok=True)
        task_thread=any('/2026/11/02/' in entry['source_path'] for entry in group['segments'])
        lines=['# '+title,'',f'- Thread ID: `{sid}`',f'- JSONL 截止时间（UTC）: {snapshot["cutoff_utc"]}',
               f'- 分段文件: {len(group["segments"])}',f'- 可阅读消息: {len(group["messages"])}']
        if task_thread:
            lines.append('- 本线程属于此次归档过程；只包含首次快照时已有记录。')
        if not group['segments']:
            lines.append('- 仅数据库元数据可用；首次 JSONL 快照未包含对应 rollout 文件。')
        lines.extend(['','## 来源分段',''])
        for entry in group['segments']:
            link='../'+entry['exported_path']
            lines.append(f'- [{Path(entry["source_path"]).name}]({link})')
        lines.extend(['','## 对话',''])
        for n,message in enumerate(group['messages'],1):
            lines.extend([f'### {n}. {message["role"]} · {message.get("timestamp") or "时间未记录"}',
                          '',f'来源: `{message["source_segment"]}:{message["source_line"]}`','',message['text'],''])
        thread_path.write_text(common.sanitize_text('\n'.join(lines)),encoding='utf-8')
        index.append({'thread_id':sid,'title':title,'readable_path':str(thread_path.relative_to(dest)),
                      'rollout_segments':len(group['segments']),'readable_messages':len(group['messages']),
                      'role_counts':dict(collections.Counter(m['role'] for m in group['messages'])),
                      'archive_task_thread':task_thread,'rollout_missing_from_snapshot':not bool(group['segments']),
                      'metadata':common.sanitize_obj(group['metadata'])})
    emit_json(dest/'index.json',index,common)
    lines=['# Codex 对话归档','',f'JSONL 截止时间（UTC）: {snapshot["cutoff_utc"]}',
           '',f'共 {len(index)} 个线程，{len(manifest["files"])} 个 JSONL 来源文件。',
           '','消息 Markdown 用于阅读；`raw/` 保留脱敏后原始事件，`databases/` 保留只读数据库中的会话投影。',
           '','| 线程 | 分段 | 消息 | 说明 |','| --- | ---: | ---: | --- |']
    for item in index:
        title=item['title'].replace('|','\\|').replace('[','\\[').replace(']','\\]')
        note='本次归档快照' if item['archive_task_thread'] else ''
        if item['rollout_missing_from_snapshot']:
            note='仅数据库元数据'
        lines.append(f'| [{title}]({item["readable_path"]}) | {item["rollout_segments"]} | {item["readable_messages"]} | {note} |')
    (dest/'README.md').write_text(common.sanitize_text('\n'.join(lines)+'\n'),encoding='utf-8')
    manifest['thread_count']=len(index)
    manifest['readable_message_count']=sum(item['readable_messages'] for item in index)
    manifest['database_manifest']='databases/manifest.json'
    attachment_records=export_attachments(dest,source_root,common,audit)
    manifest['attachment_manifest']='attachments/manifest.json'
    manifest['attachment_count']=len(attachment_records)
    manifest['redaction_counts']=dict(getattr(common,'COUNTS',{}))
    manifest['completed_at_utc']=utc_now()
    emit_json(dest/'manifest.json',manifest,common)
    return {'destination':str(dest),'threads':len(index),'readable_messages':manifest['readable_message_count'],
            'jsonl_source_files':len(manifest['files']),'attachments':len(attachment_records),
            'jsonl_cutoff_utc':snapshot['cutoff_utc']}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dest',type=Path,required=True)
    parser.add_argument('--snapshot',type=Path,default=Path('/tmp/pro5k_conversation_snapshot.json'))
    parser.add_argument('--common',type=Path,default=Path('/tmp/pro5k_archive_common.py'))
    parser.add_argument('--audit',type=Path,default=Path('/tmp/pro5k_conversation_audit.json'))
    parser.add_argument('--source-root',type=Path,default=Path('/root/.codex'))
    parser.add_argument('--legacy-root',type=Path,default=Path('/sgl-workspace/deepseek-v4-flash/session/all-sessions'))
    parser.add_argument('--legacy-only',action='store_true',help='Append legacy readable history to an already completed main export')
    args=parser.parse_args()
    if not args.common.is_file():
        parser.error('Shared redactor does not exist: '+str(args.common))
    snapshot=json.loads(args.snapshot.read_text())
    audit=json.loads(args.audit.read_text()) if args.audit.is_file() else {}
    common=import_common(args.common)
    result={} if args.legacy_only else export(snapshot,args.dest,args.source_root,common,audit)
    result['source_coverage']=append_legacy(args.dest,args.legacy_root,common,snapshot['cutoff_utc'])
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':
    main()
