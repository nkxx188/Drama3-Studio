"""Bounded real-worker checks on the freshly converted checkpoint packages."""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / '.cache/drama3-update-20261031'
for key in ('TEMP', 'TMP', 'TMPDIR', 'HF_HOME', 'XDG_CACHE_HOME', 'TORCH_HOME'):
    os.environ[key] = str(WORK)
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'vendor/fish-speech')]
import numpy as np
import soundfile as sf
from drama_low_vram.gguf_runtime import GGUFEngine
from drama_low_vram.gguf_resources import inspect_gguf
from drama_low_vram.gguf_workflows import GGUFWorkflows
from drama_low_vram.gguf_takes import load_native_take
from drama_low_vram.protocols import VoiceReference, InfillSpan


def main():
    reports = {}
    reference = ROOT / 'ui_data/voices/f627a00e97-Test Voice/reference.wav'
    for label in ('q8', 'q4'):
        model = next((ROOT / f'checkpoints/drama-3-20261029-gguf-{label}').glob('*.gguf'))
        report = reports[label] = dict(model=str(model), inspection=inspect_gguf(model), tasks={})
        engine = GGUFEngine(model, root=ROOT)
        workflows = GGUFWorkflows(engine)
        def save():
            (WORK / 'inference-acceptance.json').write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding='utf-8')
        def run(name, text, mode='tts', **kwargs):
            print('RUN', label, name, flush=True)
            begin = time.monotonic()
            result = workflows.run(mode, text, WORK / f'{label}-{name}.wav',
                                   max_new_tokens=256, seed=123, top_k=0,
                                   silence_retry=True, **kwargs)
            report['tasks'][name] = result
            save()
            if result['status'] != 'complete':
                raise RuntimeError(f'{label} {name}: {result["status"]}')
            audio, rate = sf.read(result['output'])
            if rate != 44100 or not np.isfinite(audio).all() or np.max(np.abs(audio)) < .001:
                raise RuntimeError('Invalid or silent audio')
            print('PASS', label, name, round(time.monotonic()-begin, 2), flush=True)
            return result
        try:
            run('plain', '你好，今天心情很好。')
            clone = run('clone', '[开心]你好，今天心情很好。',
                        references=[VoiceReference(str(reference), '你好，这是测试。', 0)])
            other = next((ROOT / 'ui_data/voices').glob('*-林 voice/voice.json'))
            voice = json.loads(other.read_text(encoding='utf-8'))
            run('dialogue', '<|speaker:0|>你好，你到了吗？\n<|speaker:1|>我到了，出发吧。',
                references=[VoiceReference(str(reference), '你好，这是测试。', 0),
                            VoiceReference(str(other.parent / voice['audio']), voice['transcript'], 1)],
                single_chunk=True)
            take, _, _ = load_native_take(clone['take_file'])
            if take.codes.shape[1] < 16:
                raise RuntimeError('Clone is unexpectedly short')
            edit = run('infill', '你好，今天心情不错。', mode='infill',
                       infill=InfillSpan(take.codes, 4, min(20, take.codes.shape[1]-4)))
            if not edit['infill']['prefix_codes_equal'] or not edit['infill']['suffix_codes_equal']:
                raise RuntimeError('Infill changed preserved source codes')
            report['passed'] = True
            save()
        except Exception as error:
            report['passed'] = False
            report['error'] = f'{type(error).__name__}: {error}'
            save()
            raise
        finally:
            engine.close()
    print('Both new packages passed real generation, reference cloning, dialogue and infill integrity.', flush=True)


if __name__ == '__main__':
    main()
