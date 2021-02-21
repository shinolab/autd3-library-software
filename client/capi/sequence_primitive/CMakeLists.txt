cmake_minimum_required(VERSION 3.12)

set (CMAKE_CXX_STANDARD 17)

add_library(autd3capi-sequence-primitive SHARED
  c_api.cpp
  sequence_primitive.h
  ../base/wrapper_sequence.hpp
)
 
include_directories(${PROJECT_SOURCE_DIR}/include ${EIGEN_PATH})

target_link_libraries(autd3capi-sequence-primitive sequence_primitive)
target_link_libraries(autd3capi-sequence-primitive autd3)
set_target_properties(autd3capi-sequence-primitive
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

if(NOT WIN32)
  set_target_properties(autd3capi-sequence-primitive PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
