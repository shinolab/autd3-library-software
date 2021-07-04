cmake_minimum_required(VERSION 3.10)


include_directories(
  ${PROJECT_SOURCE_DIR}/include
)

add_library(twincat_link STATIC
  twincat_link.cpp
  ${PROJECT_SOURCE_DIR}/include/autd3/link/twincat.hpp
)

set_target_properties(twincat_link
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

if(NOT WIN32)
  set_target_properties(twincat_link PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
