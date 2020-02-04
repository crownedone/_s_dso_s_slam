cmake_minimum_required(VERSION 3.12)
project(DSO_SLAM)

# later be merged into one ... or not
set(DSO DSO_LIB)
set(ORB_SLAM ORB_SLAM_LIB)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()
MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# flags
add_definitions("-D__SSE3__")
add_definitions("-DNOMINMAX")
add_definitions("-D_USE_MATH_DEFINES")

if (MSVC)
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /bigobj")
endif (MSVC)

# set boost dynamic multi-thread libraries
set (Boost_USE_MULTITHREADED ON)

# Required dependencies
find_package(gflags CONFIG REQUIRED)
find_package(glog CONFIG REQUIRED)

find_package(OpenCV CONFIG REQUIRED)
find_package(suitesparse CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(Boost REQUIRED COMPONENTS system thread filesystem) 

find_package(sophus CONFIG REQUIRED)
# Not set correctly
set(SOPHUS_SOURCE_DIR ${_VCPKG_INSTALLED_DIR}/x64-windows/include/)

# Testing
find_package(Catch2 CONFIG REQUIRED)

# Optional dependencies
find_package(Pangolin QUIET)


if (Pangolin_FOUND)
    include_directories(${PROJECT_SOURCE_DIR}/Thirdparty/Pangolin/include)
    include_directories(${PROJECT_SOURCE_DIR}/Thirdparty/Pangolin/build/src/include)
    include_directories(${PROJECT_SOURCE_DIR}/Thirdparty/Pangolin/build/external/glew/include)
    set(dso_pangolin_SOURCE_FILES 
      ${PROJECT_SOURCE_DIR}/src/IOWrapper/Pangolin/KeyFrameDisplay.cpp
      ${PROJECT_SOURCE_DIR}/src/IOWrapper/Pangolin/KeyFrameDisplay.hpp
      ${PROJECT_SOURCE_DIR}/src/IOWrapper/Pangolin/PangolinDSOViewer.cpp
      ${PROJECT_SOURCE_DIR}/src/IOWrapper/Pangolin/PangolinDSOViewer.hpp)
    source_group("Pangolin" FILES ${dso_pangolin_SOURCE_FILES})
    set(HAS_PANGOLIN 1)
else ()
    message("--- could not find PANGOLIN, not compiling dso_pangolin library.")
    message("    this means there will be no 3D display / GUI available.")
    set(dso_pangolin_SOURCE_FILES )
    set(HAS_PANGOLIN 0)
endif ()

# library setup
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

set(dso_UTIL
    ${PROJECT_SOURCE_DIR}/src/util/settings.cpp
    ${PROJECT_SOURCE_DIR}/src/util/settings.hpp
    ${PROJECT_SOURCE_DIR}/src/util/Undistort.hpp
    ${PROJECT_SOURCE_DIR}/src/util/Undistort.cpp
    ${PROJECT_SOURCE_DIR}/src/util/globalCalib.cpp
    ${PROJECT_SOURCE_DIR}/src/util/globalCalib.hpp
    ${PROJECT_SOURCE_DIR}/src/util/DatasetReader.hpp
    ${PROJECT_SOURCE_DIR}/src/util/Input.hpp
    ${PROJECT_SOURCE_DIR}/src/util/FrameShell.hpp
    ${PROJECT_SOURCE_DIR}/src/util/NumType.hpp
    ${PROJECT_SOURCE_DIR}/src/util/nanoflann.hpp
    ${PROJECT_SOURCE_DIR}/src/util/MinimalImage.hpp
    ${PROJECT_SOURCE_DIR}/src/util/IndexThreadReduce.hpp
    ${PROJECT_SOURCE_DIR}/src/util/ImageAndExposure.hpp
    ${PROJECT_SOURCE_DIR}/src/util/globalFuncs.hpp
) 
source_group("util" FILES ${dso_UTIL})
set(dso_SYS
    ${PROJECT_SOURCE_DIR}/src/StopWatch.cpp
    ${PROJECT_SOURCE_DIR}/src/StopWatch.hpp
    ${PROJECT_SOURCE_DIR}/src/Logging.hpp
) 
source_group("sys" FILES ${dso_SYS})
set(dso_FULLSYSTEM
    ${PROJECT_SOURCE_DIR}/src/DSO_System/FullSystem.cpp
    ${PROJECT_SOURCE_DIR}/src/DSO_System/FullSystem.hpp
    ${PROJECT_SOURCE_DIR}/src/DSO_System/FullSystemOptimize.cpp
    ${PROJECT_SOURCE_DIR}/src/DSO_System/FullSystemOptPoint.cpp
    ${PROJECT_SOURCE_DIR}/src/DSO_System/FullSystemDebugStuff.cpp
    ${PROJECT_SOURCE_DIR}/src/DSO_System/FullSystemMarginalize.cpp
    ${PROJECT_SOURCE_DIR}/src/DSO_System/Residuals.cpp
    ${PROJECT_SOURCE_DIR}/src/DSO_System/Residuals.hpp
    ${PROJECT_SOURCE_DIR}/src/DSO_System/ResidualProjections.hpp
    ${PROJECT_SOURCE_DIR}/src/DSO_System/CoarseTracker.cpp
    ${PROJECT_SOURCE_DIR}/src/DSO_System/CoarseTracker.hpp
    ${PROJECT_SOURCE_DIR}/src/DSO_System/CoarseInitializer.cpp
    ${PROJECT_SOURCE_DIR}/src/DSO_System/CoarseInitializer.hpp
    ${PROJECT_SOURCE_DIR}/src/DSO_System/ImmaturePoint.cpp
    ${PROJECT_SOURCE_DIR}/src/DSO_System/ImmaturePoint.hpp
    ${PROJECT_SOURCE_DIR}/src/DSO_System/HessianBlocks.cpp
    ${PROJECT_SOURCE_DIR}/src/DSO_System/HessianBlocks.hpp
    ${PROJECT_SOURCE_DIR}/src/DSO_System/PixelSelector.cpp
    ${PROJECT_SOURCE_DIR}/src/DSO_System/PixelSelector.hpp
) 
source_group("DSO_System" FILES ${dso_FULLSYSTEM})
set(dso_OPTBACK
    ${PROJECT_SOURCE_DIR}/src/OptimizationBackend/EnergyFunctional.cpp
    ${PROJECT_SOURCE_DIR}/src/OptimizationBackend/EnergyFunctional.hpp
    ${PROJECT_SOURCE_DIR}/src/OptimizationBackend/AccumulatedTopHessian.cpp
    ${PROJECT_SOURCE_DIR}/src/OptimizationBackend/AccumulatedTopHessian.hpp
    ${PROJECT_SOURCE_DIR}/src/OptimizationBackend/AccumulatedSCHessian.cpp
    ${PROJECT_SOURCE_DIR}/src/OptimizationBackend/AccumulatedSCHessian.hpp
    ${PROJECT_SOURCE_DIR}/src/OptimizationBackend/EnergyFunctionalStructs.cpp
    ${PROJECT_SOURCE_DIR}/src/OptimizationBackend/EnergyFunctionalStructs.hpp
    ${PROJECT_SOURCE_DIR}/src/OptimizationBackend/MatrixAccumulators.hpp
    ${PROJECT_SOURCE_DIR}/src/OptimizationBackend/RawResidualJacobian.hpp
)
source_group("OptimizationBackend" FILES ${dso_OPTBACK})
set(dso_IO
    ${PROJECT_SOURCE_DIR}/src/IOWrapper/OpenCV/ImageDisplay_OpenCV.cpp
    ${PROJECT_SOURCE_DIR}/src/IOWrapper/OpenCV/ImageRW_OpenCV.cpp
    ${PROJECT_SOURCE_DIR}/src/IOWrapper/ImageDisplay.hpp
    ${PROJECT_SOURCE_DIR}/src/IOWrapper/ImageRW.hpp
    ${PROJECT_SOURCE_DIR}/src/IOWrapper/Output3DWrapper.hpp
    #${PROJECT_SOURCE_DIR}/src/IOWrapper/OutputWrapper/SampleOutputWrapper.hpp
)
source_group("IO" FILES ${dso_IO})

# Sources files
set(dso_SOURCE_FILES
  ${dso_UTIL}
  ${dso_OPTBACK}
  ${dso_FULLSYSTEM}
  ${dso_SYS}
  ${dso_IO}
)

set(dso_TESTS
  ${PROJECT_SOURCE_DIR}/tests/SampleTest.cpp
)
source_group("tests" FILES ${dso_TESTS})



include_directories(
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/src
  ${SOPHUS_SOURCE_DIR}
  ${EIGEN3_INCLUDE_DIR}
  ${Pangolin_INCLUDE_DIRS}
) 

# DSO Library (will be shared when we got rid of all the static settings)
add_library(${DSO} ${dso_SOURCE_FILES})
target_link_libraries(${DSO}
glog::glog
${OpenCV_LIBS}
${EIGEN3_LIBS}
Boost::system Boost::thread Boost::filesystem
SuiteSparse::amd SuiteSparse::btf SuiteSparse::klu SuiteSparse::ldl
)

# Main Executable
add_executable(exe_DSO_SLAM ${PROJECT_SOURCE_DIR}/src/main.cpp ${dso_pangolin_SOURCE_FILES})
target_link_libraries(exe_DSO_SLAM ${DSO} ${Pangolin_LIBRARIES} gflags)
add_dependencies(exe_DSO_SLAM ${DSO})

# Tests
add_executable(unit_DSO_SLAM ${PROJECT_SOURCE_DIR}/tests/CatchMain.cpp ${dso_TESTS})
target_link_libraries(unit_DSO_SLAM ${DSO} gflags)
add_dependencies(unit_DSO_SLAM ${DSO})


#### APPENDIX #### Copy missing dll files (Only Windows)
if(WIN32)
# copy pangolin dlls
add_custom_command(TARGET exe_DSO_SLAM POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${PROJECT_SOURCE_DIR}/thirdparty/Pangolin/build/src/$<CONFIGURATION>"
    $<TARGET_FILE_DIR:exe_DSO_SLAM>)
# copy missing hdf5 and zlib1.dll
add_custom_command(TARGET exe_DSO_SLAM POST_BUILD 
    COMMAND ${CMAKE_COMMAND} -E copy_if_different 
    ${_VCPKG_INSTALLED_DIR}/x64-windows/bin/hdf5.dll
    $<TARGET_FILE_DIR:exe_DSO_SLAM>)
add_custom_command(TARGET exe_DSO_SLAM POST_BUILD 
    COMMAND ${CMAKE_COMMAND} -E copy_if_different 
    ${_VCPKG_INSTALLED_DIR}/x64-windows/bin/zstd.dll
    $<TARGET_FILE_DIR:exe_DSO_SLAM>)
add_custom_command(TARGET exe_DSO_SLAM POST_BUILD 
    COMMAND ${CMAKE_COMMAND} -E copy_if_different 
    ${_VCPKG_INSTALLED_DIR}/x64-windows/bin/zlib1.dll
    $<TARGET_FILE_DIR:exe_DSO_SLAM>)
add_custom_command(TARGET exe_DSO_SLAM POST_BUILD 
    COMMAND ${CMAKE_COMMAND} -E copy_if_different 
    ${_VCPKG_INSTALLED_DIR}/x64-windows/bin/gflags.dll
    $<TARGET_FILE_DIR:exe_DSO_SLAM>/gflags_debug.dll)

    # copy pangolin dlls
add_custom_command(TARGET unit_DSO_SLAM POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${PROJECT_SOURCE_DIR}/thirdparty/Pangolin/build/src/$<CONFIGURATION>"
    $<TARGET_FILE_DIR:unit_DSO_SLAM>)
# copy missing hdf5 and zlib1.dll
add_custom_command(TARGET unit_DSO_SLAM POST_BUILD 
    COMMAND ${CMAKE_COMMAND} -E copy_if_different 
    ${_VCPKG_INSTALLED_DIR}/x64-windows/bin/hdf5.dll
    $<TARGET_FILE_DIR:unit_DSO_SLAM>)
add_custom_command(TARGET unit_DSO_SLAM POST_BUILD 
    COMMAND ${CMAKE_COMMAND} -E copy_if_different 
    ${_VCPKG_INSTALLED_DIR}/x64-windows/bin/zstd.dll
    $<TARGET_FILE_DIR:unit_DSO_SLAM>)
add_custom_command(TARGET unit_DSO_SLAM POST_BUILD 
    COMMAND ${CMAKE_COMMAND} -E copy_if_different 
    ${_VCPKG_INSTALLED_DIR}/x64-windows/bin/zlib1.dll
    $<TARGET_FILE_DIR:unit_DSO_SLAM>)
endif()