@echo off
title APEX Trading Engine

echo.
echo  ============================================
echo   APEX Trading Engine — Phase 1 Setup
echo  ============================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo  ERROR: Python not found.
    echo  Please install Python from https://python.org
    echo  Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)

echo  [1/3] Python found. Installing dependencies...
pip install -r requirements.txt --quiet

if errorlevel 1 (
    echo  ERROR: Failed to install dependencies.
    echo  Try running: pip install flask flask-cors requests
    pause
    exit /b 1
)

echo  [2/3] Dependencies installed.
echo  [3/3] Starting APEX Engine...
echo.
echo  ============================================
echo   Server running at: http://localhost:5000
echo   Open apex_dashboard.html in your browser
echo  ============================================
echo.
echo  Add your API keys in the dashboard Setup menu (top right)
echo  or edit config.json directly.
echo.
echo  Press Ctrl+C to stop the server.
echo.

python server.py

pause
