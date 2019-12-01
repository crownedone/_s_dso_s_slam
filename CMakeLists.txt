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

find_package(Pangolin CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)
find_package(gflags CONFIG REQUIRED)
find_package(glog CONFIG REQUIRED)
#find_package(libzip CONFIG REQUIRED)

# flags
add_definitions("-D__SSE3__")
add_definitions("-DNOMINMAX")
add_definitions("-D_USE_MATH_DEFINES")
set(CMAKE_CXX_FLAGS_RELEASE "/MT")

if (MSVC)
     set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /bigobj")
endif (MSVC)

# Sources files
set(dso_SOURCE_FILES
  ${PROJECT_SOURCE_DIR}/src/FullSystem/FullSystem.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/FullSystemOptimize.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/FullSystemOptPoint.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/FullSystemDebugStuff.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/FullSystemMarginalize.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/Residuals.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/CoarseTracker.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/CoarseInitializer.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/ImmaturePoint.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/HessianBlocks.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/PixelSelector.cpp
  ${PROJECT_SOURCE_DIR}/src/OptimizationBackend/EnergyFunctional.cpp
  ${PROJECT_SOURCE_DIR}/src/OptimizationBackend/AccumulatedTopHessian.cpp
  ${PROJECT_SOURCE_DIR}/src/OptimizationBackend/AccumulatedSCHessian.cpp
  ${PROJECT_SOURCE_DIR}/src/OptimizationBackend/EnergyFunctionalStructs.cpp
  ${PROJECT_SOURCE_DIR}/src/util/settings.cpp
  ${PROJECT_SOURCE_DIR}/src/util/Undistort.h
  ${PROJECT_SOURCE_DIR}/src/util/Undistort.cpp
  ${PROJECT_SOURCE_DIR}/src/util/globalCalib.cpp
  ${PROJECT_SOURCE_DIR}/src/sys/times.cpp
  ${PROJECT_SOURCE_DIR}/src/sys/Logging.hpp
  ${PROJECT_SOURCE_DIR}/src/util/DatasetReader.h
)
set(SOPHUS_SOURCE_DIR "C:/vcpkg/installed/x64-windows/include/")
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
	  ${PROJECT_SOURCE_DIR}/src/IOWrapper/Pangolin/PangolinDSOViewer.cpp)
	set(HAS_PANGOLIN 1)
else ()
	message("--- could not find PANGOLIN, not compiling dso_pangolin library.")
	message("    this means there will be no 3D display / GUI available for dso_dataset.")
	set(dso_pangolin_SOURCE_FILES )
	set(HAS_PANGOLIN 0)
endif ()

# decide if we have openCV
if (OpenCV_FOUND)
	set(dso_opencv_SOURCE_FILES 
	  ${PROJECT_SOURCE_DIR}/src/IOWrapper/OpenCV/ImageDisplay_OpenCV.cpp
	  ${PROJECT_SOURCE_DIR}/src/IOWrapper/OpenCV/ImageRW_OpenCV.cpp)
	set(HAS_OPENCV 1)
else ()
	message("--- could not find OpenCV, not compiling dso_opencv library.")
	message("    this means there will be no image display, and image read / load functionality.")
	set(dso_opencv_SOURCE_FILES 
	  ${PROJECT_SOURCE_DIR}/src/IOWrapper/ImageDisplay_dummy.cpp
	  ${PROJECT_SOURCE_DIR}/src/IOWrapper/ImageRW_dummy.cpp)
	set(HAS_OPENCV 0)
endif ()

# decide if we have Boost
if (Boost_FOUND)
	set(HAS_BOOST 1)
else ()
	set(HAS_BOOST 0)
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


include_directories(${CSPARSE_INCLUDE_DIR} ${CHOLMOD_INCLUDE_DIR}) 
	
# build main executable (only if we have both OpenCV Pangolin and Boost)
if (OpenCV_FOUND AND Boost_FOUND)
	add_executable(dso ${PROJECT_SOURCE_DIR}/src/main_dso_pangolin.cpp ${dso_SOURCE_FILES} ${dso_opencv_SOURCE_FILES} ${dso_pangolin_SOURCE_FILES})

	target_link_libraries(dso gflags)
	target_link_libraries(dso glog::glog)
	target_link_libraries(dso pangolin)
	target_link_libraries(dso opencv_world opencv_img_hash ${__cvcomponent})
	target_link_libraries(dso Boost::system Boost::thread Boost::filesystem)
	target_link_libraries(dso SuiteSparse::amd SuiteSparse::btf SuiteSparse::klu SuiteSparse::ldl)
	target_include_directories(dso PRIVATE ${Pangolin_INCLUDE_DIRS})

	# copy pangolin dlls
	add_custom_command(TARGET dso POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${PROJECT_SOURCE_DIR}/thirdparty/Pangolin/build/src/$<CONFIGURATION>"
        $<TARGET_FILE_DIR:dso>)

	# copy missing hdf5 and zlib1.dll
	add_custom_command(TARGET dso POST_BUILD 
    COMMAND ${CMAKE_COMMAND} -E copy_if_different 
        "C:/vcpkg/installed/x64-windows/bin/hdf5.dll"
        $<TARGET_FILE_DIR:dso>)
	add_custom_command(TARGET dso POST_BUILD 
    COMMAND ${CMAKE_COMMAND} -E copy_if_different 
        "C:/vcpkg/installed/x64-windows/bin/zlib1.dll"
        $<TARGET_FILE_DIR:dso>)

	# copy zstd.dll
	add_custom_command(TARGET dso POST_BUILD 
    COMMAND ${CMAKE_COMMAND} -E copy_if_different 
        "${PROJECT_SOURCE_DIR}/thirdparty/Pangolin/build/src/Release/zstd.dll"
        $<TARGET_FILE_DIR:dso>)


else()
	message("--- not building dso_dataset, since either don't have openCV or Pangolin.")
endif()

