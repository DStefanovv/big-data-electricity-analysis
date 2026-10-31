@echo off
title Analiz na elektropotreblenieto
cd /d "%~dp0web"

if not exist "node_modules" (
  echo Installing dependencies...
  call npm install
  if errorlevel 1 (
    echo.
    echo Installation failed.
    pause
    exit /b 1
  )
)

echo Starting the local report...
start "" /b cmd /c "timeout /t 2 /nobreak >nul && start http://127.0.0.1:4173/"
call npm run preview -- --host 127.0.0.1 --strictPort

echo.
echo The local server has stopped.
pause
