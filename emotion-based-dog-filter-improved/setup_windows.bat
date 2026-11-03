@echo off
setlocal
cd /d "%~dp0"
echo ================================================
echo Emotion-Based Dog Filter - first-time setup
echo ================================================
where py >nul 2>nul
if %errorlevel%==0 (
    py -3.12 -m venv venv
) else (
    python -m venv venv
)
if not exist "venv\Scripts\python.exe" (
    echo.
    echo Could not create the virtual environment. Install Python 3.12 and try again.
    pause
    exit /b 1
)
call "venv\Scripts\activate.bat"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo Dependency installation failed. Check your internet connection and Python version.
    pause
    exit /b 1
)
echo.
echo Setup complete! Double-click run_filter.bat to start the filter.
pause
