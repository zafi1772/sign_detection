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

:: Check model file
if not exist "models\best50_50.h5" (
    echo.
    echo  [ERROR] Missing: models\best50_50.h5
    echo  Make sure all model files are present.
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
