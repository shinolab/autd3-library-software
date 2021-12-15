cmake_minimum_required(VERSION 3.16)

add_library(from_file_modulation
  from_file.cpp
  ${PROJECT_SOURCE_DIR}/include/autd3/modulation/from_file.hpp
)

target_include_directories(from_file_modulation PUBLIC ${PROJECT_SOURCE_DIR}/include ${EIGEN_PATH})

set_target_properties(from_file_modulation
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

if(WIN32)
  set_target_properties(from_file_modulation PROPERTIES FOLDER "modulation")
else()
  set_target_properties(from_file_modulation PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
