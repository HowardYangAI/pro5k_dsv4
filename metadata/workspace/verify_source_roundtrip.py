#!/usr/bin/env python3
from pathlib import Path
import ast,gzip,hashlib,json,sys
sys.path.insert(0,'/tmp')
from pro5k_archive_common import sanitize_text,sanitize_obj
root=Path(sys.argv[1])
indexpath=root/'metadata/workspace/offline_file_index.json'
if not indexpath.exists():indexpath=Path(str(indexpath)+'.gz')
raw=indexpath.read_bytes()
index=json.loads(gzip.decompress(raw) if indexpath.suffix=='.gz' else raw)
seen=set();checked=0;new_errors=[];roundtrip_changes=[];source_py_valid=0
for entry in index['entries']:
 if not entry['source_path'].endswith('.py'):continue
 key=entry['outputs'][0]['path']
 if key.endswith('.symlink.json'):continue
 if key in seen:continue
 seen.add(key)
 out=root/key
 archived=out.read_bytes()
 if entry.get('compression')=='gzip':archived=gzip.decompress(archived)
 try:source=Path(entry['source_path']).read_text();target=archived.decode('utf-8')
 except (OSError,UnicodeDecodeError):continue
 checked+=1
 if sanitize_text(target)!=target:roundtrip_changes.append(key)
 try:ast.parse(source)
 except (SyntaxError,ValueError):continue
 source_py_valid+=1
 try:ast.parse(target)
 except (SyntaxError,ValueError):new_errors.append({'source':entry['source_path'],'archived_path':key})
manifestpath=root/'metadata/workspace/export_manifest.json'
manifest=json.loads(manifestpath.read_text())
result={'python_unique_files_checked':checked,'original_python_syntax_valid':source_py_valid,'archived_new_python_syntax_errors':new_errors,'latest_sanitizer_roundtrip_changes':roundtrip_changes,'manifest_sanitizer_idempotent':sanitize_obj(manifest)==manifest}
output=root/'metadata/workspace/source_roundtrip_verification.json'
output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
if new_errors or roundtrip_changes or not result['manifest_sanitizer_idempotent']:raise SystemExit(1)
