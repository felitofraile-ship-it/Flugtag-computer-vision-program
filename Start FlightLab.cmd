@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  py -3 -m venv .venv
  if errorlevel 1 goto fail
)
".venv\Scripts\python.exe" -c "import cv2, numpy, PIL, matplotlib, imageio_ffmpeg" >nul 2>&1
if errorlevel 1 (
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 goto fail
)
".venv\Scripts\python.exe" app.py %*
if errorlevel 1 pause
exit /b
:fail
echo Setup failed. Install Python 3.11, 3.12, or 3.13 with tkinter, then retry.
pause
