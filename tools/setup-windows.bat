@echo off
setlocal
cd /d "%~dp0.."

set "CACHE_ROOT=%CD%\.cache"
set "UV_CACHE_DIR=%CACHE_ROOT%\uv"
set "PIP_CACHE_DIR=%CACHE_ROOT%\pip"
set "HF_HOME=%CACHE_ROOT%\huggingface"
set "HF_HUB_CACHE=%HF_HOME%\hub"
set "MODELSCOPE_CACHE=%CACHE_ROOT%\modelscope"
set "TRITON_CACHE_DIR=%CACHE_ROOT%\triton"
set "TORCHINDUCTOR_CACHE_DIR=%CACHE_ROOT%\torchinductor"
set "TORCH_EXTENSIONS_DIR=%CACHE_ROOT%\torch_extensions"
set "TEMP=%CACHE_ROOT%\tmp"
set "TMP=%CACHE_ROOT%\tmp"

for %%D in (
  "%UV_CACHE_DIR%"
  "%PIP_CACHE_DIR%"
  "%HF_HUB_CACHE%"
  "%MODELSCOPE_CACHE%"
  "%TRITON_CACHE_DIR%"
  "%TORCHINDUCTOR_CACHE_DIR%"
  "%TORCH_EXTENSIONS_DIR%"
  "%TEMP%"
) do if not exist "%%~D" mkdir "%%~D"

where uv >nul 2>&1
if errorlevel 1 goto setup_with_python
if not exist ".venv\Scripts\python.exe" uv venv .venv --python 3.12
if errorlevel 1 goto failed
uv pip install --python .venv\Scripts\python.exe -r requirements-windows.lock.txt
if errorlevel 1 goto failed
goto prepare_sources

:setup_with_python
if exist ".venv\Scripts\python.exe" goto install_with_python
py -3.12 -m venv .venv
if errorlevel 1 goto failed
:install_with_python
".venv\Scripts\python.exe" -m pip install -r requirements-windows.lock.txt
if errorlevel 1 goto failed

:prepare_sources
".venv\Scripts\python.exe" tools\prepare_sources.py
if errorlevel 1 goto failed

echo.
echo Setup complete. Cache root: %CACHE_ROOT%
echo Run start.bat in the project folder.
echo Native worker builds are described in docs\development.md.
endlocal
exit /b 0

:failed
echo.
echo Setup failed. Check the error above before starting Studio.
endlocal
exit /b 1
