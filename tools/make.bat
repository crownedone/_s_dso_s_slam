@echo off

cd /d "%~dp0"
C:\CMake\bin\cmake.exe .. -G "Visual Studio 15 2017 Win64" -DBUILD_SHARED_LIBS=ON -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
cd /d "%~dp0"
Pangolin.sln

pause