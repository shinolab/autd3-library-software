cmake_minimum_required(VERSION 3.10)

set (CMAKE_CXX_STANDARD 17)

set(EIGEN_PATH ../../deps/Eigen3)

include_directories(../../include ${EIGEN_PATH})

add_library(emulator_link STATIC
emulator_link.cpp
	../../include/emulator_link.hpp
)

if(NOT WIN32)
  set_target_properties(emulator_link PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
