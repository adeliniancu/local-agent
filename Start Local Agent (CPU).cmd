@echo off
setlocal
cd /d "%~dp0"
set "PYTHONHOME="
set "PYTHONPATH="
set "LOCAL_AGENT_GPU_LAYERS=0"
if not exist "%~dp0runtime\python\pythonw.exe" goto missing
if not exist "%~dp0runtime\python\python.exe" goto missing
if not exist "%~dp0runtime\ollama\ollama.exe" goto missing
if not exist "%~dp0bootstrap.py" goto missing
if not exist "%~dp0app.py" goto missing
start "Local Agent - CPU" "%~dp0runtime\python\pythonw.exe" -I "%~dp0bootstrap.py"
exit /b 0

:missing
echo.
echo Local Agent could not find all of its files.
echo Extract the ENTIRE ZIP into a writable folder first.
echo Keep the runtime and models folders beside this launcher.
echo See START HERE.txt for help.
echo.
pause
exit /b 1
