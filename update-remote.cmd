@echo off
setlocal enabledelayedexpansion

REM -------------------------------------------------------------------
REM Script to update a Git remote URL from cruegge/itreg.git to regpy/regpy.git
REM while preserving the protocol (HTTPS vs SSH).
REM
REM This script was created by ChatGPT (OpenAI).
REM -------------------------------------------------------------------

REM Get current remote URL
for /f "tokens=*" %%i in ('git remote get-url origin') do set REMOTE_URL=%%i

set OLD_PATH=cruegge/itreg.git
set NEW_PATH=regpy/regpy.git

echo Current remote: %REMOTE_URL%

REM Check if SSH style (starts with git@)
echo %REMOTE_URL% | findstr /b "git@" >nul
if %errorlevel%==0 (
    echo Detected SSH remote
    echo %REMOTE_URL% | findstr /c:"%OLD_PATH%" >nul
    if %errorlevel%==0 (
        set NEW_URL=%REMOTE_URL:%OLD_PATH%=%NEW_PATH%%
        echo Updating SSH remote:
        echo   %REMOTE_URL% → %NEW_URL%
        git remote set-url origin %NEW_URL%
    ) else (
        echo Remote is SSH but does not match %OLD_PATH% — no change.
    )
    goto :eof
)

REM Check if HTTPS style (starts with https://)
echo %REMOTE_URL% | findstr /b "https://" >nul
if %errorlevel%==0 (
    echo Detected HTTPS remote
    echo %REMOTE_URL% | findstr /c:"%OLD_PATH%" >nul
    if %errorlevel%==0 (
        set NEW_URL=%REMOTE_URL:%OLD_PATH%=%NEW_PATH%%
        echo Updating HTTPS remote:
        echo   %REMOTE_URL% → %NEW_URL%
        git remote set-url origin %NEW_URL%
    ) else (
        echo Remote is HTTPS but does not match %OLD_PATH% — no change.
    )
    goto :eof
)

echo Unknown remote format: %REMOTE_URL%
exit /b 1

