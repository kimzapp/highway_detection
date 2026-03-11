@echo off
REM ============================================
REM Highway Detection - Windows Build Script
REM ============================================
setlocal enabledelayedexpansion

set "PROJECT_NAME=HighwayDetection"
set "VERSION=1.0.0"
set "BUILD_DIR=build"
set "DIST_DIR=dist"

echo.
echo ============================================
echo   Building %PROJECT_NAME% v%VERSION%
echo ============================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install dependencies
echo [INFO] Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt
pip install pyinstaller

REM Clean previous builds
echo [INFO] Cleaning previous builds...
if exist "%BUILD_DIR%" rmdir /s /q "%BUILD_DIR%"
if exist "%DIST_DIR%" rmdir /s /q "%DIST_DIR%"

REM Create assets directory if not exists
if not exist "assets" mkdir assets

REM Build with PyInstaller
echo [INFO] Building executable with PyInstaller...
pyinstaller highway_detection.spec --clean

if errorlevel 1 (
    echo [ERROR] Build failed!
    exit /b 1
)

REM Create release package
echo [INFO] Creating release package...
set "RELEASE_DIR=release\%PROJECT_NAME%-%VERSION%"
if exist "release" rmdir /s /q "release"
mkdir "%RELEASE_DIR%"

REM Copy built files
xcopy /s /e /i "%DIST_DIR%\%PROJECT_NAME%" "%RELEASE_DIR%"

REM Copy additional files
if exist "README.md" copy "README.md" "%RELEASE_DIR%\"
if exist "LICENSE" copy "LICENSE" "%RELEASE_DIR%\"

REM Create version info
echo %PROJECT_NAME% v%VERSION% > "%RELEASE_DIR%\VERSION.txt"
echo Build Date: %date% %time% >> "%RELEASE_DIR%\VERSION.txt"

REM Create ZIP archive
echo [INFO] Creating ZIP archive...
powershell -Command "Compress-Archive -Path '%RELEASE_DIR%\*' -DestinationPath 'release\%PROJECT_NAME%-%VERSION%-win64.zip' -Force"

echo.
echo ============================================
echo   Build completed successfully!
echo   Output: release\%PROJECT_NAME%-%VERSION%-win64.zip
echo ============================================
echo.

endlocal
