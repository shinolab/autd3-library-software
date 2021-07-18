cmake_minimum_required(VERSION 3.10)

include_directories(${PROJECT_SOURCE_DIR}/include)

add_library(modulation_from_file STATIC
  from_file.cpp
  ${PROJECT_SOURCE_DIR}/include/autd3/modulation/from_file.hpp
)

set_target_properties(modulation_from_file
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

if(NOT WIN32)
  set_target_properties(modulation_from_file PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
