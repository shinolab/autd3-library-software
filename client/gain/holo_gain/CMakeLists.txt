cmake_minimum_required(VERSION 3.10)

set (CMAKE_CXX_STANDARD 17)

set(EIGEN_PATH ../../deps/Eigen3)

include_directories(../../include ${EIGEN_PATH})

add_library(holo_gain STATIC
	holo_gain.cpp
	../../include/holo_gain.hpp
)

if(NOT WIN32)
  set_target_properties(holo_gain PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
