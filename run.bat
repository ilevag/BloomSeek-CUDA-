@echo off
REM Bitcoin Generator - Quick Launch Script
REM =========================================

echo.
echo ============================================
echo    Bitcoin Generator - GPU Accelerated
echo ============================================
echo.

REM Check if exe exists
if not exist "BitcoinGenerator.exe" (
    echo [ERROR] BitcoinGenerator.exe not found!
    echo Please compile the project first or copy the exe here.
    pause
    exit /b 1
)

REM Check if DLLs exist
if not exist "sqlite3.dll" (
    echo [WARNING] sqlite3.dll not found!
    echo The program may not work correctly.
    echo.
)

if not exist "cudart64_65.dll" (
    echo [WARNING] cudart64_65.dll not found!
    echo CUDA runtime is missing.
    echo.
)

REM Run the program
echo Starting Bitcoin Generator...
echo.
BitcoinGenerator.exe

pause

