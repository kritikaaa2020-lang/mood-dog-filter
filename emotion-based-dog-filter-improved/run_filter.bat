@echo off
setlocal
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
    echo First run setup_windows.bat, then run this file again.
    pause
    exit /b 1
)
call "venv\Scripts\activate.bat"
cd src
python step_8_dog_emotion_mask.py
if errorlevel 1 echo The program ended with an error. Read the message above.
pause
