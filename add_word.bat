@echo off
cd /d "%~dp0"
title Add New Sign Word

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Run install.bat first.
    pause & exit /b 1
)

echo.
echo  =====================================================
echo   Add a New Bangla Sign Word
echo  =====================================================
echo.
set /p WORD="  Enter the Bangla word to add (e.g. ধন্যবাদ): "
if "%WORD%"=="" ( echo No word entered. & pause & exit /b 1 )

echo.
echo  Step 1 of 2 - Recording samples for: %WORD%
echo  A webcam window will open.
echo  Press SPACE to start recording, SPACE to pause, Q when done.
echo.
pause

.venv\Scripts\python collect_signs.py --word "%WORD%" --samples 120
if errorlevel 1 ( echo [ERROR] Collection failed. & pause & exit /b 1 )

echo.
echo  Step 2 of 2 - Training the new model...
echo.
.venv\Scripts\python train_landmark_model.py
if errorlevel 1 ( echo [ERROR] Training failed. & pause & exit /b 1 )

echo.
echo  Done! Restart run.bat to use the new word.
echo.
pause
