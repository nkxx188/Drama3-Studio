"""Link the existing pinned ggml reference quantizers into a small CPU DLL."""
import os
from pathlib import Path
import subprocess
from build_native import windows_tools

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / '.cache/drama3-update-20261031'
WORK.mkdir(parents=True, exist_ok=True)
os.environ.update(TEMP=str(WORK), TMP=str(WORK))
env, _, _, installation = windows_tools()
env.update(TEMP=str(WORK), TMP=str(WORK))
ggml = ROOT / '.cache/audio-cpp-phase19/external/ggml'
lib = ROOT / '.cache/audio-cpp-phase19-sm89-build/ggml/src'
compiler = sorted((Path(installation) / 'VC/Tools/MSVC').glob('*/bin/Hostx64/x64/cl.exe'))[-1]
subprocess.run([
    str(compiler), '/nologo', '/O2', '/MD', '/LD', '/openmp', '/std:c++17',
    '/I' + str(ggml / 'include'), '/I' + str(ggml / 'src'),
    str(ROOT / 'tools/gguf_quantizer_bridge.cpp'),
    '/Fo' + str(WORK / 'quantizer.obj'), '/link',
    '/OUT:' + str(WORK / 'quantizer.dll'), str(lib / 'ggml-base.lib'),
], cwd=WORK, env=env, check=True)
