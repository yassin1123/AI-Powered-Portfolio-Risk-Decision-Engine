@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo.
echo === 1/5  Folder: %CD% ===
echo.

echo === 2/5  Remove ANY old Git data (root + Main code) ===
if exist ".git" (
  rd /s /q ".git"
  echo Removed: .git
) else (
  echo No .git in project root - OK
)
if exist "Main code\.git" (
  rd /s /q "Main code\.git"
  echo Removed: Main code\.git
) else (
  echo No .git in Main code - OK
)

where git >nul 2>nul
if errorlevel 1 (
  echo.
  echo ERROR: Git is not installed or not in PATH.
  echo Install from: https://git-scm.com/download/win
  echo Then reopen Command Prompt and run this script again.
  pause
  exit /b 1
)

echo.
echo === 3/5  git init ===
git init
if errorlevel 1 pause & exit /b 1

echo.
echo === 4/5  git add + commit (venv/.zip/.env ignored by .gitignore) ===
git add .
if errorlevel 1 pause & exit /b 1

git commit -m "Initial commit"
if errorlevel 1 (
  echo If it says "nothing to commit", you may need: git config user.email "you@example.com"
  echo                           and: git config user.name "Your Name"
  pause
  exit /b 1
)

git branch -M main

echo.
echo === 5/5  DONE locally ===
echo.
echo NOW run these commands yourself (same Command Prompt window is fine):
echo.
echo   git remote add origin https://github.com/yassin1123/AI-Powered-Portfolio-Risk-Decision-Engine.git
echo.
echo If it says remote already exists, run this first:
echo   git remote remove origin
echo.
echo   git push -u origin main
echo.
echo First push to an existing remote with old junk may need:
echo   git push -u origin main --force
echo.
pause
