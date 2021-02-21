cmake_minimum_required(VERSION 3.12)

set (CMAKE_CXX_STANDARD 17)

set(INCLUDE_LIST
  ${PROJECT_SOURCE_DIR}/include
)

add_library(autd3capi-modulation-from-file SHARED
  c_api.cpp
  modulation_from_file.h
  ../base/wrapper_modulation.hpp
)
 
include_directories(${INCLUDE_LIST})

target_link_libraries(autd3capi-modulation-from-file modulation_from_file)
target_link_libraries(autd3capi-modulation-from-file autd3)

set_target_properties(autd3capi-modulation-from-file
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

if(NOT WIN32)
  set_target_properties(autd3capi-modulation-from-file PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
