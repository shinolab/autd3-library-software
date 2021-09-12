cmake_minimum_required(VERSION 3.16)

add_library(autd3capi-remote-twincat-link SHARED
  c_api.cpp
  remote_twincat_link.h
  ../base/header.h
  ../base/wrapper_link.hpp
)

target_link_libraries(autd3capi-remote-twincat-link remote_twincat_link)
if(WIN32)
elseif(APPLE)
else()
  target_link_libraries(autd3capi-remote-twincat-link -static-libgcc -static-libstdc++)
endif()

set_target_properties(autd3capi-remote-twincat-link
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

if(WIN32)
  set_target_properties(autd3capi-remote-twincat-link PROPERTIES FOLDER "capi")
else()
  set_target_properties(autd3capi-remote-twincat-link PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
