@echo off
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel% equ 0 (
    py -m pip install -r requirements.txt
) else (
    python -m pip install -r requirements.txt
)
if errorlevel 1 pause
