cmake_minimum_required(VERSION 3.10)

set (CMAKE_CXX_STANDARD 17)

include_directories(${PROJECT_SOURCE_DIR}/include ${EIGEN_PATH})

add_library(link_debug STATIC
  debug.cpp
	${PROJECT_SOURCE_DIR}/include/link/debug.hpp
)

set_target_properties(link_debug
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

if(NOT WIN32)
  set_target_properties(link_debug PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
