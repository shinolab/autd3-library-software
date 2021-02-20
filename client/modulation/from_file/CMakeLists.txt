cmake_minimum_required(VERSION 3.10)

set (CMAKE_CXX_STANDARD 17)

set(EIGEN_PATH ../../deps/Eigen3)

include_directories(../../include ${EIGEN_PATH})

add_library(modulation_from_file STATIC
  from_file.cpp
  ../../include/modulation/from_file.hpp
)

if(NOT WIN32)
  set_target_properties(modulation_from_file PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
