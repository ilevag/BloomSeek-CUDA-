@echo off
REM Bitcoin Generator - Build Script
REM =================================

echo.
echo ============================================
echo    BloomSeek - Build Script
echo ============================================
echo.

REM Check if CMakeLists.txt exists
if not exist "CMakeLists.txt" (
    echo [ERROR] CMakeLists.txt not found!
    pause
    exit /b 1
)

REM Create build directory
if not exist "build" (
    echo [1/4] Creating build directory...
    mkdir build
) else (
    echo [1/4] Build directory exists, cleaning...
    rmdir /s /q build
    mkdir build
)

cd build

echo.
echo [2/4] Configuring project with CMake...
echo.
cmake .. -G "Visual Studio 17 2022" -A x64
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] CMake configuration failed!
    echo Make sure you have:
    echo   - Visual Studio 2022 with C++ tools
    echo   - CUDA Toolkit 11.5+
    echo   - CMake 3.18+
    cd ..
    pause
    exit /b 1
)

echo.
echo [3/4] Building project (Release)...
echo.
cmake --build . --config Release -j
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Build failed!
    cd ..
    pause
    exit /b 1
)

echo.
echo [4/4] Copying files...
copy Release\BloomSeek.exe ..\ >nul 2>&1
if exist Release\BloomSeek.exp copy Release\BloomSeek.exp ..\ >nul 2>&1
if exist Release\BloomSeek.lib copy Release\BloomSeek.lib ..\ >nul 2>&1

cd ..

echo.
echo ============================================
echo    BUILD SUCCESSFUL!
echo ============================================
echo.
echo Executable: BloomSeek.exe
echo.
echo To run the program:
echo   run.bat
echo   or
echo   .\BloomSeek.exe
echo.

pause

