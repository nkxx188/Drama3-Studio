@echo off
setlocal
title Drama 3 Studio
pushd "%~dp0"
if errorlevel 1 exit /b 1
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "DRAMA_PYTHON=%~dp0runtime\python\python.exe"
if not exist "%DRAMA_PYTHON%" set "DRAMA_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%DRAMA_PYTHON%" goto missing_runtime
"%DRAMA_PYTHON%" -I "%~dp0tools\windows_launcher.py" %*
set "DRAMA_EXIT=%errorlevel%"
if not "%DRAMA_EXIT%"=="0" goto failed
popd
exit /b 0

:missing_runtime
echo.
echo The Python runtime is missing. Extract the complete Windows package first.
echo Expected: "%~dp0runtime\python\python.exe"
if not exist "%~dp0diagnostics" mkdir "%~dp0diagnostics" >nul 2>&1
> "%~dp0diagnostics\launcher-error.log" echo The Python runtime is missing.
>> "%~dp0diagnostics\launcher-error.log" echo Expected: "%~dp0runtime\python\python.exe"
set "DRAMA_EXIT=1"

:failed
echo.
echo Drama 3 could not start. The error above explains why.
echo Error log: "%~dp0diagnostics\launcher-error.log"
if not defined DRAMA_NO_PAUSE pause
popd
exit /b %DRAMA_EXIT%
