cmake_minimum_required(VERSION 3.12)

set (CMAKE_CXX_STANDARD 17)

set(INCLUDE_LIST
  ${PROJECT_SOURCE_DIR}/include
)

add_library(autd3capi-twincat-link SHARED
  c_api.cpp
  twincat_link.h
  ../base/header.h
  ../base/wrapper_link.hpp
)

include_directories(${INCLUDE_LIST})

target_link_libraries(autd3capi-twincat-link twincat_link)
if(WIN32)
  target_link_libraries(autd3capi-twincat-link ws2_32.lib)
elseif(APPLE)
else()
  target_link_libraries(autd3capi-twincat-link -static-libgcc -static-libstdc++)
endif()

set_target_properties(autd3capi-twincat-link
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

if(NOT WIN32)
  set_target_properties(autd3capi-twincat-link PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
