@echo off
setlocal
cd /d "%~dp0"
"%~dp0runtime\python\python.exe" -I "%~dp0source_sync.py" --stop
