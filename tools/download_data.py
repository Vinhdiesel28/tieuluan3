"""Download public inputs; verify exact files used in the submitted experiments."""
from pathlib import Path
import hashlib, json, urllib.request

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
DATA.mkdir(exist_ok=True)
missing = []
for item in json.loads((ROOT / 'docs/data_manifest.json').read_text()):
    path = DATA / item['file']
    if not path.exists():
        if not item['url']:
            missing.append(item)
            continue
        print('Downloading', item['file'], flush=True)
        temporary = path.with_suffix(path.suffix + '.download')
        urllib.request.urlretrieve(item['url'], temporary)
        with temporary.open('rb') as f:
            checksum = hashlib.file_digest(f, 'sha256').hexdigest()
        if checksum != item['sha256']:
            raise RuntimeError(f"Checksum mismatch: {item['file']}; source may have changed.")
        temporary.replace(path)
    with path.open('rb') as f:
        checksum = hashlib.file_digest(f, 'sha256').hexdigest()
    if checksum != item['sha256']:
        raise RuntimeError(f"Different input file: {path}. See docs/data_manifest.json.")
    print('Verified', item['file'])
if missing:
    for item in missing:
        print('Download manually:', item['source'])
        print('Place the original CSV at:', DATA / item['file'])
    raise SystemExit('Housing CSV is required. The submitted run used the file supplied for Assignment 3.')
