cmake_minimum_required(VERSION 3.12)

set (CMAKE_CXX_STANDARD 17)

set(INCLUDE_LIST
  ${PROJECT_SOURCE_DIR}/include
  ${EIGEN_PATH}
)

add_library(autd3capi-gain-holo SHARED
  c_api.cpp
  gain_holo.h
  ../base/wrapper_gain.hpp
)
 
include_directories(${INCLUDE_LIST})

target_link_libraries(autd3capi-gain-holo gain_holo)
target_link_libraries(autd3capi-gain-holo autd3)

set_target_properties(autd3capi-gain-holo
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

if(NOT WIN32)
  set_target_properties(autd3capi-gain-holo PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
