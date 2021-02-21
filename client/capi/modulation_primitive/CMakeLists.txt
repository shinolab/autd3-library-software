cmake_minimum_required(VERSION 3.12)

set (CMAKE_CXX_STANDARD 17)

set(INCLUDE_LIST
  ${PROJECT_SOURCE_DIR}/include
)

add_library(autd3capi-modulation-primitive SHARED
  c_api.cpp
  modulation_primitive.h
  ../base/wrapper_modulation.hpp
)
 
include_directories(${INCLUDE_LIST})

target_link_libraries(autd3capi-modulation-primitive modulation_primitive)
target_link_libraries(autd3capi-modulation-primitive autd3)

set_target_properties(autd3capi-modulation-primitive
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

if(NOT WIN32)
  set_target_properties(autd3capi-modulation-primitive PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
