@ECHO OFF

IF EXIST C:/CMake/bin/ (
echo Found CMake.
) ELSE (
echo Please install CMake in C:/CMake
)
IF EXIST C:/vcpkg/ (
echo Found VCPKG.
echo installing dependencies:
C:/vcpkg/vcpkg.exe install ^
boost:x64-windows ^
eigen3:x64-windows ^
clapack:x64-windows ^
ceres:x64-windows ^
freeglut:x64-windows ^
gflags:x64-windows ^
glog:x64-windows ^
gl3w:x64-windows ^
glew:x64-windows ^
glfw3:x64-windows ^
hdf5:x64-windows ^
libzip[bzip2]:x64-windows ^
libpng:x64-windows ^
libjpeg-turbo:x64-windows ^
sophus:x64-windows ^
suitesparse:x64-windows ^
zlib:x64-windows ^
opengl:x64-windows ^
catch2:x64-windows ^
opencv4[contrib,png,tiff,jpeg,opengl,world,cuda]:x64-windows

) ELSE (
echo Please install vcpkg in C:/vcpkg
)

mkdir thirdparty
cd thirdparty
git clone https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin
mkdir build
cd build

Echo Please copy CMakeLists from tools folder to thirdparty/Pangolin.
Echo Please copy make.bat from tools folder to thirdparty/Pangolin/build and run.
Echo Make sure you build both Debug and Release targets for pangolin.

pause