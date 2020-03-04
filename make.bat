@echo off

cd /d "%~dp0"
If Not Exist build (
mkdir build
)
cd build
If Exist C:\CMake\bin\cmake.exe  (
C:\CMake\bin\cmake.exe .. -G "Visual Studio 16 2019" -A x64 -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
) Else if Exist "C:\Program Files\CMake\bin\cmake.exe" (
"C:\Program Files\CMake\bin\cmake.exe .. -G "Visual Studio 16 2019" -A x64 -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake"
)
cd /d "%~dp0"
DSO_SLAM.sln

pause