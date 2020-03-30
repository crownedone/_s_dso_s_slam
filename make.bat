@echo off

cd /d "%~dp0"
If Not Exist build (
mkdir build
)
cd build

If Exist "C:\CMake\bin\cmake.exe"  (
C:\CMake\bin\cmake.exe .. -G "Visual Studio 16 2019" -A x64 -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
) Else If Exist 'C:\Program\ Files\CMake\bin\cmake.exe' (
"C:\Program\ Files\CMake\bin\cmake.exe" .. -G "Visual Studio 16 2019" -A x64 -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
) Else If Exist "D:\Program Files\CMake\bin\cmake.exe"  (
"D:\Program Files\CMake\bin\cmake.exe" .. -G "Visual Studio 16 2019" -A x64 -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
) Else If Exist 'D:\Program Files\CMake\bin\cmake.exe' (
"D:\Program Files\CMake\bin\cmake.exe" .. -G "Visual Studio 16 2019" -A x64 -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
)
DSO_SLAM.sln
cd /d "%~dp0

pause