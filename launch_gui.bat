@echo off
REM Windows launcher for LiquidUI

echo ============================================================
echo  LiquidUI - Quantitative Trading Platform
echo ============================================================
echo.
echo Starting GUI...
echo.

REM Try to run with virtual environment if it exists
if exist venv\Scripts\python.exe (
    echo Using virtual environment...
    venv\Scripts\python.exe launch_gui.py
) else (
    echo Virtual environment not found, using system Python...
    python launch_gui.py
)

if errorlevel 1 (
    echo.
    echo Error: Failed to launch GUI
    echo Make sure you have installed all requirements:
    echo   pip install -r requirements-windows.txt
    echo.
    pause
)
