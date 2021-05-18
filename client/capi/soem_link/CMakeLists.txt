cmake_minimum_required(VERSION 3.12)

set (CMAKE_CXX_STANDARD 17)

set(INCLUDE_LIST
  ${PROJECT_SOURCE_DIR}/include
)

add_library(autd3capi-soem-link SHARED
  c_api.cpp
  soem_link.h
  ../base/header.h
  ../base/wrapper_link.hpp
)

include_directories(${INCLUDE_LIST})

target_link_libraries(autd3capi-soem-link soem_link)
if(WIN32)
  target_link_libraries(autd3capi-soem-link ws2_32.lib)
  target_link_libraries(autd3capi-soem-link winmm.lib)
elseif(APPLE)
  target_link_libraries(autd3capi-soem-link pcap)
else()
  target_link_libraries(autd3capi-soem-link rt)
  target_link_libraries(autd3capi-soem-link -static-libgcc -static-libstdc++)
endif()

set_target_properties(autd3capi-soem-link
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

if(NOT WIN32)
  set_target_properties(autd3capi-soem-link PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
