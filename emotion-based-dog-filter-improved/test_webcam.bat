@echo off
setlocal
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
    echo First run setup_windows.bat.
    pause
    exit /b 1
)
call "venv\Scripts\activate.bat"
cd src
python step_3_camera_test.py
pause
