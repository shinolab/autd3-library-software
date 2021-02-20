cmake_minimum_required(VERSION 3.12)

set (CMAKE_CXX_STANDARD 17)

set(INCLUDE_LIST
  ../../include
)

add_library(autd3capi-link-soem SHARED
  c_api.cpp
  soem_link.h
  ../base/wrapper_link.hpp
)

include_directories(${INCLUDE_LIST})

target_link_libraries(autd3capi-link-soem link_soem)
if(WIN32)
  target_link_libraries(autd3capi-link-soem ws2_32.lib)
  target_link_libraries(autd3capi-link-soem winmm.lib)
elseif(APPLE)
  target_link_libraries(autd3capi-link-soem pcap)
else()
  target_link_libraries(autd3capi-link-soem rt)
endif()

set_target_properties(autd3capi-link-soem
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

if(NOT WIN32)
  set_target_properties(autd3capi-link-soem PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
