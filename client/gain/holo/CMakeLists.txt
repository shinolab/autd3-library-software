cmake_minimum_required(VERSION 3.10)

set (CMAKE_CXX_STANDARD 17)

include_directories(${PROJECT_SOURCE_DIR}/include ${EIGEN_PATH})

add_library(gain_holo STATIC
	holo.cpp
	${PROJECT_SOURCE_DIR}/include/gain/holo.hpp
)

set_target_properties(gain_holo
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

if(NOT WIN32)
  set_target_properties(gain_holo PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
