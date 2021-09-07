cmake_minimum_required(VERSION 3.16)

add_library(autd3capi-holo-gain SHARED
  c_api.cpp
  holo_gain.h
  ../base/header.h
  ../base/wrapper.hpp
)

target_link_libraries(autd3capi-holo-gain holo_gain)
if(WIN32)
elseif(APPLE)
else()
  target_link_libraries(autd3capi-holo-gain -static-libgcc -static-libstdc++)
endif()

set_target_properties(autd3capi-holo-gain
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

if(WIN32)
  set_target_properties(autd3capi-holo-gain PROPERTIES FOLDER "capi")
else()
  set_target_properties(autd3capi-holo-gain PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
