cmake_minimum_required(VERSION 3.16)

add_library(autd3capi-twincat-link SHARED
  c_api.cpp
  twincat_link.h
  ../base/header.h
  ../base/wrapper_link.hpp
)

target_link_libraries(autd3capi-twincat-link twincat_link)

set_target_properties(autd3capi-twincat-link
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

set_target_properties(autd3capi-twincat-link PROPERTIES FOLDER "capi/link")
