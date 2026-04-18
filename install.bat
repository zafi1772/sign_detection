@echo off
cd /d "%~dp0"
title Bangla Sign Language - Installation

echo.
echo  =====================================================
echo   Bangla Sign Language Detector - Installation
echo  =====================================================
echo.

:: Check Python
echo  [1/4] Checking Python...

py -3.12 --version >nul 2>&1
if not errorlevel 1 ( set PYTHON=py -3.12 & goto python_ok )

py -3.11 --version >nul 2>&1
if not errorlevel 1 ( set PYTHON=py -3.11 & goto python_ok )

py -3.10 --version >nul 2>&1
if not errorlevel 1 ( set PYTHON=py -3.10 & goto python_ok )

python --version >nul 2>&1
if not errorlevel 1 ( set PYTHON=python & goto python_ok )

echo.
echo  [ERROR] Python not found!
echo  Install Python 3.10, 3.11, or 3.12 from https://www.python.org/downloads/
echo.
pause
exit /b 1

:python_ok
%PYTHON% --version
echo  Python OK.
echo.

:: Create virtual environment
echo  [2/4] Creating virtual environment...
if exist ".venv\Scripts\python.exe" (
    echo  Already exists, skipping.
) else (
    %PYTHON% -m venv .venv
    if errorlevel 1 ( echo  [ERROR] Could not create venv. & pause & exit /b 1 )
    echo  Created .venv
)
echo.

:: Upgrade pip
echo  [3/4] Upgrading pip...
.venv\Scripts\python -m pip install --upgrade pip --quiet
echo.

:: Install packages
echo  [4/4] Installing packages (may take 5-10 minutes)...
echo        Downloading TensorFlow and dependencies, please wait...
echo.

.venv\Scripts\pip install --no-cache-dir --timeout 300 -r requirements.txt

if errorlevel 1 (
    echo.
    echo  [ERROR] Installation failed. Check internet and try again.
    echo.
    pause
    exit /b 1
)

:: Download MediaPipe hand model if missing
echo  [+] Checking MediaPipe hand model...
if not exist "models\hand_landmarker.task" (
    echo  Downloading hand_landmarker.task (~7 MB)...
    .venv\Scripts\python -c "import urllib.request; urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task', 'models/hand_landmarker.task'); print('  Downloaded.')"
) else (
    echo  Model already present.
)
echo.

echo.
echo  =====================================================
echo   Installation complete!
echo   Double-click run.bat to start the application.
echo  =====================================================
echo.
pause
