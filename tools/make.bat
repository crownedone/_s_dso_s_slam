@echo off

cd /d "%~dp0"
C:\CMake\bin\cmake.exe .. -G "Visual Studio 16 2019" -A x64 -DBUILD_SHARED_LIBS=ON -DBUILD_EXTERN_GLEW=OFF -DBUILD_EXTERN_LIBPNG=OFF -DBUILD_EXTERN_LIBJPEG=OFF -DMSVC_USE_STATIC_CRT=OFF -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
cd /d "%~dp0"
Pangolin.sln

pause