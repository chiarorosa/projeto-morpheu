@echo off
rem Morpheu launcher. Usage: morpheu.cmd [--browser | --server] [--port N]
setlocal
cd /d "%~dp0backend"
"%~dp0.venv\Scripts\python.exe" -m morpheu %*
