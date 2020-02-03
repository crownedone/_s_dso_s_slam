@echo off

cd /d "%~dp0"
C:\CMake\bin\cmake.exe .. -G "Visual Studio 16 2019" -A x64 -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
cd /d "%~dp0"
DSO.sln

pause