cmake_minimum_required(VERSION 3.10)

set (CMAKE_CXX_STANDARD 17)

include_directories(../../include)

add_library(modulation_primitive STATIC
  primitive.cpp
  ../../include/modulation/primitive.hpp
)

if(NOT WIN32)
  set_target_properties(modulation_primitive PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
