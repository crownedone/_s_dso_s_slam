SET(PROJECT_NAME DSO)

PROJECT(${PROJECT_NAME})
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
	
# set boost dynamic multi-thread libraries
set (Boost_USE_MULTITHREADED ON)

# required libraries
find_package(suitesparse CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(Boost 1.71.0 REQUIRED COMPONENTS system thread filesystem) 
find_package(sophus CONFIG REQUIRED)

# Required dependencies
find_package(OpenCV CONFIG REQUIRED)
find_package(gflags CONFIG REQUIRED)
find_package(glog CONFIG REQUIRED)

# Optional dependencies
find_package(Pangolin CONFIG QUIET)
find_package(libzip CONFIG QUIET) # has no cmake windows support and can be found with pkg-config if needed.

# Testing
find_package(Catch2 CONFIG REQUIRED)

# flags
add_definitions("-D__SSE3__")
add_definitions("-DNOMINMAX")
add_definitions("-D_USE_MATH_DEFINES")

if (MSVC)
     set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /bigobj")
endif (MSVC)

set(dso_UTIL
	${PROJECT_SOURCE_DIR}/src/util/settings.cpp
	${PROJECT_SOURCE_DIR}/src/util/settings.h
	${PROJECT_SOURCE_DIR}/src/util/Undistort.h
	${PROJECT_SOURCE_DIR}/src/util/Undistort.cpp
	${PROJECT_SOURCE_DIR}/src/util/globalCalib.cpp
	${PROJECT_SOURCE_DIR}/src/util/globalCalib.h
	${PROJECT_SOURCE_DIR}/src/util/DatasetReader.h
	${PROJECT_SOURCE_DIR}/src/util/FrameShell.h
	${PROJECT_SOURCE_DIR}/src/util/NumType.h
	${PROJECT_SOURCE_DIR}/src/util/nanoflann.h
	${PROJECT_SOURCE_DIR}/src/util/MinimalImage.h
	${PROJECT_SOURCE_DIR}/src/util/IndexThreadReduce.h
	${PROJECT_SOURCE_DIR}/src/util/ImageAndExposure.h
	${PROJECT_SOURCE_DIR}/src/util/globalFuncs.h
) 
source_group("Util" FILES ${dso_UTIL})
set(dso_SYS
	${PROJECT_SOURCE_DIR}/src/sys/times.cpp
	${PROJECT_SOURCE_DIR}/src/sys/times.h
	${PROJECT_SOURCE_DIR}/src/sys/time.h
	${PROJECT_SOURCE_DIR}/src/sys/Logging.hpp
) 
source_group("Sys" FILES ${dso_SYS})
set(dso_FULLSYSTEM
	${PROJECT_SOURCE_DIR}/src/FullSystem/FullSystem.cpp
	${PROJECT_SOURCE_DIR}/src/FullSystem/FullSystem.h
	${PROJECT_SOURCE_DIR}/src/FullSystem/FullSystemOptimize.cpp
	${PROJECT_SOURCE_DIR}/src/FullSystem/FullSystemOptPoint.cpp
	${PROJECT_SOURCE_DIR}/src/FullSystem/FullSystemDebugStuff.cpp
	${PROJECT_SOURCE_DIR}/src/FullSystem/FullSystemMarginalize.cpp
	${PROJECT_SOURCE_DIR}/src/FullSystem/Residuals.cpp
	${PROJECT_SOURCE_DIR}/src/FullSystem/Residuals.h
	${PROJECT_SOURCE_DIR}/src/FullSystem/ResidualProjections.h
	${PROJECT_SOURCE_DIR}/src/FullSystem/CoarseTracker.cpp
	${PROJECT_SOURCE_DIR}/src/FullSystem/CoarseTracker.h
	${PROJECT_SOURCE_DIR}/src/FullSystem/CoarseInitializer.cpp
	${PROJECT_SOURCE_DIR}/src/FullSystem/CoarseInitializer.h
	${PROJECT_SOURCE_DIR}/src/FullSystem/ImmaturePoint.cpp
	${PROJECT_SOURCE_DIR}/src/FullSystem/ImmaturePoint.h
	${PROJECT_SOURCE_DIR}/src/FullSystem/HessianBlocks.cpp
	${PROJECT_SOURCE_DIR}/src/FullSystem/HessianBlocks.h
	${PROJECT_SOURCE_DIR}/src/FullSystem/PixelSelector.cpp
	${PROJECT_SOURCE_DIR}/src/FullSystem/PixelSelector.h
) 
source_group("FullSystem" FILES ${dso_FULLSYSTEM})
set(dso_OPTBACK
	${PROJECT_SOURCE_DIR}/src/OptimizationBackend/EnergyFunctional.cpp
	${PROJECT_SOURCE_DIR}/src/OptimizationBackend/EnergyFunctional.h
	${PROJECT_SOURCE_DIR}/src/OptimizationBackend/AccumulatedTopHessian.cpp
	${PROJECT_SOURCE_DIR}/src/OptimizationBackend/AccumulatedTopHessian.h
	${PROJECT_SOURCE_DIR}/src/OptimizationBackend/AccumulatedSCHessian.cpp
	${PROJECT_SOURCE_DIR}/src/OptimizationBackend/AccumulatedSCHessian.h
	${PROJECT_SOURCE_DIR}/src/OptimizationBackend/EnergyFunctionalStructs.cpp
	${PROJECT_SOURCE_DIR}/src/OptimizationBackend/EnergyFunctionalStructs.h
	${PROJECT_SOURCE_DIR}/src/OptimizationBackend/MatrixAccumulators.h
	${PROJECT_SOURCE_DIR}/src/OptimizationBackend/RawResidualJacobian.h
)
source_group("OptimizationBackend" FILES ${dso_OPTBACK})
set(dso_IO
	${PROJECT_SOURCE_DIR}/src/IOWrapper/OpenCV/ImageDisplay_OpenCV.cpp
	${PROJECT_SOURCE_DIR}/src/IOWrapper/OpenCV/ImageRW_OpenCV.cpp
	${PROJECT_SOURCE_DIR}/src/IOWrapper/ImageDisplay.h
	${PROJECT_SOURCE_DIR}/src/IOWrapper/ImageRW.h
	${PROJECT_SOURCE_DIR}/src/IOWrapper/Output3DWrapper.h
	#${PROJECT_SOURCE_DIR}/src/IOWrapper/OutputWrapper/SampleOutputWrapper.h
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

set(dso_TEST_CASES
  ${dso_SOURCE_FILES}
  ${dso_TESTS}
)

set(SOPHUS_SOURCE_DIR ${_VCPKG_INSTALLED_DIR}/x64-windows/include/)
include_directories(
  ${PROJECT_SOURCE_DIR}/src
  ${SOPHUS_SOURCE_DIR}
  ${EIGEN3_INCLUDE_DIR}
) 



# decide if we have pangolin
if (Pangolin_FOUND)
	include_directories(${PROJECT_SOURCE_DIR}/thirdparty/Pangolin/include)
	include_directories(${PROJECT_SOURCE_DIR}/thirdparty/Pangolin/build/src/include)
	include_directories(${PROJECT_SOURCE_DIR}/thirdparty/Pangolin/build/external/glew/include)
	set(dso_pangolin_SOURCE_FILES 
	  ${PROJECT_SOURCE_DIR}/src/IOWrapper/Pangolin/KeyFrameDisplay.cpp
	  ${PROJECT_SOURCE_DIR}/src/IOWrapper/Pangolin/KeyFrameDisplay.h
	  ${PROJECT_SOURCE_DIR}/src/IOWrapper/Pangolin/PangolinDSOViewer.cpp
	  ${PROJECT_SOURCE_DIR}/src/IOWrapper/Pangolin/PangolinDSOViewer.h)
	source_group("Pangolin" FILES ${dso_pangolin_SOURCE_FILES})
	set(HAS_PANGOLIN 1)
else ()
	message("--- could not find PANGOLIN, not compiling dso_pangolin library.")
	message("    this means there will be no 3D display / GUI available for dso_dataset.")
	set(dso_pangolin_SOURCE_FILES )
	set(HAS_PANGOLIN 0)
endif ()


# decide if we have ziplib.
if (LibZip_FOUND)
	message("--- found ziplib (${LIBZIP_VERSION}), compiling with zip capability.")
	add_definitions(-DHAS_ZIPLIB=1)
	include_directories( ${LIBZIP_INCLUDE_DIR_ZIP} ${LIBZIP_INCLUDE_DIR_ZIPCONF} ) 
else()
	message("--- not found ziplib (${LIBZIP_LIBRARY}), compiling without zip capability.")
	set(LIBZIP_LIBRARY "")
endif()

add_executable(dso ${PROJECT_SOURCE_DIR}/src/main_dso_pangolin.cpp ${dso_SOURCE_FILES} ${dso_pangolin_SOURCE_FILES})
add_executable(dsoTests ${PROJECT_SOURCE_DIR}/tests/CatchMain.cpp ${dso_TEST_CASES} ${dso_pangolin_SOURCE_FILES})

target_link_libraries(dso gflags)
target_link_libraries(dso glog::glog)
target_link_libraries(dso opencv_world opencv_img_hash ${__cvcomponent})
target_link_libraries(dso Boost::system Boost::thread Boost::filesystem)
target_link_libraries(dso SuiteSparse::amd SuiteSparse::btf SuiteSparse::klu SuiteSparse::ldl)

target_link_libraries(dsoTests gflags)
target_link_libraries(dsoTests glog::glog)
target_link_libraries(dsoTests opencv_world opencv_img_hash ${__cvcomponent})
target_link_libraries(dsoTests Boost::system Boost::thread Boost::filesystem)
target_link_libraries(dsoTests SuiteSparse::amd SuiteSparse::btf SuiteSparse::klu SuiteSparse::ldl)
target_link_libraries(dsoTests Catch2::Catch2)

if (Pangolin_FOUND)
target_link_libraries(dso pangolin)
target_include_directories(dso PRIVATE ${Pangolin_INCLUDE_DIRS})
# copy pangolin dlls
add_custom_command(TARGET dso POST_BUILD
COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${PROJECT_SOURCE_DIR}/thirdparty/Pangolin/build/src/$<CONFIGURATION>"
    $<TARGET_FILE_DIR:dso>)
# copy zstd.dll
add_custom_command(TARGET dso POST_BUILD 
COMMAND ${CMAKE_COMMAND} -E copy_if_different 
    "${PROJECT_SOURCE_DIR}/thirdparty/Pangolin/build/src/Release/zstd.dll"
    $<TARGET_FILE_DIR:dso>)


target_link_libraries(dsoTests pangolin)
target_include_directories(dsoTests PRIVATE ${Pangolin_INCLUDE_DIRS})
# copy pangolin dlls
add_custom_command(TARGET dsoTests POST_BUILD
COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${PROJECT_SOURCE_DIR}/thirdparty/Pangolin/build/src/$<CONFIGURATION>"
    $<TARGET_FILE_DIR:dsoTests>)
# copy zstd.dll
add_custom_command(TARGET dsoTests POST_BUILD 
COMMAND ${CMAKE_COMMAND} -E copy_if_different 
    "${PROJECT_SOURCE_DIR}/thirdparty/Pangolin/build/src/Release/zstd.dll"
    $<TARGET_FILE_DIR:dso>)
endif()


# copy missing hdf5 and zlib1.dll
add_custom_command(TARGET dso POST_BUILD 
COMMAND ${CMAKE_COMMAND} -E copy_if_different 
    ${_VCPKG_INSTALLED_DIR}/x64-windows/bin/hdf5.dll
    $<TARGET_FILE_DIR:dso>)
add_custom_command(TARGET dso POST_BUILD 
COMMAND ${CMAKE_COMMAND} -E copy_if_different 
    ${_VCPKG_INSTALLED_DIR}/x64-windows/bin/zlib1.dll
    $<TARGET_FILE_DIR:dso>)

	# copy missing hdf5 and zlib1.dll
add_custom_command(TARGET dsoTests POST_BUILD 
COMMAND ${CMAKE_COMMAND} -E copy_if_different 
    ${_VCPKG_INSTALLED_DIR}/x64-windows/bin/hdf5.dll
    $<TARGET_FILE_DIR:dsoTests>)
add_custom_command(TARGET dsoTests POST_BUILD 
COMMAND ${CMAKE_COMMAND} -E copy_if_different 
    ${_VCPKG_INSTALLED_DIR}/x64-windows/bin/zlib1.dll
    $<TARGET_FILE_DIR:dsoTests>)
