@echo off
title ALERT Driver Fatigue Detection System
cd /d "%~dp0"
set PYTHONPATH=src
echo ========================================================
echo   ALERT Driver Fatigue Detection System (Production Model)
echo ========================================================
echo Starting camera stream...
.venv\Scripts\python.exe src\run_alert.py %*
if errorlevel 1 (
    echo.
    echo An error occurred while running ALERT.
    pause
)
