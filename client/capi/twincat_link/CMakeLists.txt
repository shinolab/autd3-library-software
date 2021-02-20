cmake_minimum_required(VERSION 3.12)

set (CMAKE_CXX_STANDARD 17)

if (POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

if (POLICY CMP0079)
  cmake_policy(SET CMP0079 NEW)
endif()

set(INCLUDE_LIST
  ../../include
)

add_library(autd3capi-twincat-link SHARED
  c_api.cpp
  autd3_c_api_twincat_link.h
  ../base/wrapper_link.hpp
)

include_directories(${INCLUDE_LIST})

target_link_libraries(autd3capi-twincat-link twincat_link)

set_target_properties(autd3capi-twincat-link
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

if(NOT WIN32)
  set_target_properties(autd3capi-twincat-link PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
