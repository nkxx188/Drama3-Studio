"""Download the October 29 checkpoint to a fresh D-drive directory, with hashes."""

import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / '.cache/drama3-update-20261031'
WORK.mkdir(parents=True, exist_ok=True)
for key in ('TEMP', 'TMP', 'TMPDIR', 'HF_HOME', 'HUGGINGFACE_HUB_CACHE', 'XDG_CACHE_HOME'):
    os.environ[key] = str(WORK)
sys.path.insert(0, str(ROOT / 'src'))
import requests
from drama_low_vram.model_download import Downloader

REVISION = '78e5ab79e49aa9a25abe8799eff2c4d73a1c3216'
DESTINATION = ROOT / 'checkpoints/drama-3-20261029-fp8'


def main():
    session = requests.Session()
    response = session.get(f'https://huggingface.co/api/models/fishaudio/drama-3/revision/{REVISION}?blobs=true', timeout=60)
    response.raise_for_status()
    files = []
    for row in response.json()['siblings']:
        if row['rfilename'] == '.gitattributes':
            continue
        digest = (row.get('lfs') or {}).get('sha256')
        if digest is None:
            content = session.get(f'https://huggingface.co/fishaudio/drama-3/resolve/{REVISION}/{row["rfilename"]}', timeout=60)
            content.raise_for_status()
            digest = hashlib.sha256(content.content).hexdigest()
        files.append(dict(name=row['rfilename'], size=row['size'], sha256=digest, huggingface_compatible=True))
    # Pin the ModelScope snapshot rather than following a mutable branch.
    response = session.get('https://modelscope.cn/api/v1/models/fishaudio/drama-3/repo/files?Revision=master&Recursive=true', timeout=60)
    response.raise_for_status()
    ms = {row['Name']: row for row in response.json()['Data']['Files']}
    if any(ms.get(row['name'], {}).get('Sha256') != row['sha256'] for row in files):
        raise RuntimeError('ModelScope and Hugging Face checkpoint hashes differ')
    # Per-file immutable revisions are accepted by the existing downloader.
    for row in files:
        row['revision'] = ms[row['name']]['Revision']
    manifest = dict(model='fishaudio/drama-3', hf_revision=REVISION, files=files)
    (WORK / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    downloader = Downloader(DESTINATION, files, 12, 32 * 1024**2,
                            source='auto', revision=REVISION, hf_revision=REVISION)
    sample = next(row for row in files if row['name'].endswith('.safetensors'))
    def probe(source):
        started = time.monotonic()
        try:
            count = 16 * 1024**2
            with requests.get(downloader.direct_url(sample, source),
                              headers={'Range': f'bytes=0-{count - 1}', 'Accept-Encoding': 'identity'},
                              stream=True, timeout=35) as response:
                if response.status_code != 206:
                    raise RuntimeError(f'HTTP {response.status_code}')
                data = response.raw.read(count + 1)
                if len(data) != count:
                    raise RuntimeError('Range length mismatch')
            seconds = time.monotonic() - started
            return dict(source=source, seconds=seconds, mib_per_second=16 / seconds,
                        sha256=hashlib.sha256(data).hexdigest())
        except Exception as error:
            return dict(source=source, error=str(error))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(probe, ['modelscope', 'huggingface']))
    good = [row for row in results if 'seconds' in row]
    if not good:
        raise RuntimeError(results)
    if len(good) == 2 and good[0]['sha256'] != good[1]['sha256']:
        raise RuntimeError('Speed probes returned different content')
    chosen = min(good, key=lambda row: row['seconds'])['source']
    (WORK / 'speed-probes.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(dict(event='speed_test', results=results, chosen=chosen)), flush=True)
    downloader.source = chosen
    downloader.selected_source = chosen
    last = [0.0]
    def progress(event):
        now = time.monotonic()
        if now - last[0] > 15 or event['stage'] not in ('downloading', 'verifying'):
            print(json.dumps(event), flush=True)
            last[0] = now
    downloader.progress = progress
    downloader.run()
    print(json.dumps(dict(event='complete', directory=str(DESTINATION))), flush=True)


if __name__ == '__main__':
    main()
