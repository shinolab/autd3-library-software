cmake_minimum_required(VERSION 3.20)

add_library(modulation_from_file
  from_file.cpp
  ${PROJECT_SOURCE_DIR}/include/autd3/modulation/from_file.hpp
)

target_include_directories(modulation_from_file PUBLIC ${PROJECT_SOURCE_DIR}/include)

set_target_properties(modulation_from_file
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

if(WIN32)
  set_target_properties(modulation_from_file PROPERTIES FOLDER "gain")
else()
  set_target_properties(modulation_from_file PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
