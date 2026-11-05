@echo off
cd /d "%~dp0"
if not exist "media\flight.mp4" (
  echo Place the original matching Flugtag clip at media\flight.mp4 first.
  echo For another clip, use Start FlightLab.cmd and select Open video.
  pause
  exit /b 1
)
call "%~dp0Start FlightLab.cmd" "media\flight.mp4" --setup "%~dp0flugtag_setup.json"
