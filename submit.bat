@echo off
:: Auto Git Commit and Push Script (Only if changes exist)

:: Change directory to where the script itself is
cd /d "%~dp0"

:: Check for changes
for /f %%i in ('git status --porcelain') do set changes=true

if not defined changes (
    echo No changes detected. Exiting.
    pause
    exit /b
)

:: Stage changes
git add .

:: Commit with timestamp
set datetime=%date% %time%
git commit -m "commit on %datetime%"
git push origin main

echo loose? I dont loose, I win , that's my job that's what i do
pause
