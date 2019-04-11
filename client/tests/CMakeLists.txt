cmake_minimum_required(VERSION 3.1)

set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Yupch.h /FIpch.h")

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/deps)
set(Matlab_INCLUDE_DIRS "")

cmake_policy(SET CMP0074 NEW)

find_package( Eigen3 REQUIRED )
find_package( Boost REQUIRED )
find_package( Threads REQUIRED )
find_package( MATLAB )

include_directories(../include ${EIGEN3_INCLUDE_DIR} ${Boost_INCLUDE_DIR} ${Ads_INCLUDE_DIR} ${Matlab_INCLUDE_DIRS})

add_executable(tests 
	test.cpp
	pch.h
	pch.cpp
	../lib/controller.cpp
	../lib/link.cpp
	../lib/ethercat_link.cpp
	../lib/geometry.cpp
	../lib/gain.cpp
	../lib/matlabgain.cpp
	../lib/holo_gain.cpp
	../lib/modulation.cpp
	../lib/groupedgain.cpp
	../lib/link.hpp
	../lib/timer.hpp
	../lib/timer.cpp
	../lib/ethercat_link.hpp
	../include/privdef.hpp
	../include/autd3.hpp
)

if (Matlab_FOUND AND Matlab_MX_LIBRARY)
    add_definitions(-DMATLAB_ENABLED)
    target_link_libraries(autd3 ${Matlab_MX_LIBRARY} ${Matlab_MAT_LIBRARY})
endif()

set_source_files_properties(test.cpp
    PROPERTIES
    COMPILE_FLAGS "/Ycpch.h"
)

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

find_package(GTest MODULE REQUIRED)
target_link_libraries(tests ${GTEST_BOTH_LIBRARIES} ${Boost_LIBRARIES} ${Ads_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4068")