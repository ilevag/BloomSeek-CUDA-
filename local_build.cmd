@echo off
setlocal

REM Configure Visual Studio build environment
call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat"
if errorlevel 1 goto :error

REM Ensure build folder exists
if not exist "build" mkdir build
cd build

set "CMAKE_GENERATOR=Visual Studio 18 2026"
set "CMAKE_GENERATOR_PLATFORM=x64"
set "CMAKE_GENERATOR_INSTANCE=C:\Program Files\Microsoft Visual Studio\18\Community"

REM Configure project with CMake (Visual Studio 2022 generator)
"C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" -S .. -B . -G "Visual Studio 18 2026" -A x64
if errorlevel 1 goto :error

REM Build Release configuration
"C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" --build . --config Release
if errorlevel 1 goto :error

exit /b 0

:error
exit /b 1

