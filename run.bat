@echo off
cd /d "%~dp0"
title Bangla Sign Language Detector

:: Check install was run first
if not exist ".venv\Scripts\python.exe" (
    echo.
    echo  [ERROR] Virtual environment not found.
    echo  Please run install.bat first.
    echo.
    pause
    exit /b 1
)

:: Check MediaPipe model
if not exist "models\hand_landmarker.task" (
    echo.
    echo  [ERROR] Missing: models\hand_landmarker.task
    echo  Please run install.bat first.
    echo.
    pause
    exit /b 1
)

:: Launch
echo.
echo  =====================================================
echo   Bangla Sign Language Detector - Starting...
echo  =====================================================
echo.
echo   Hold a sign for ~1 second  =  word added to sentence
echo   Press V  =  speak sentence aloud
echo   Press C  =  clear sentence
echo   Press B  =  remove last word
echo   Press Q  =  quit
echo.

.venv\Scripts\python main.py

if errorlevel 1 (
    echo.
    echo  [ERROR] Application crashed. See error above.
    pause
)
