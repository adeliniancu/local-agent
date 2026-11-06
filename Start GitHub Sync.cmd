@echo off
setlocal
cd /d "%~dp0"
start "" "%~dp0runtime\python\pythonw.exe" -I "%~dp0source_sync.py"
