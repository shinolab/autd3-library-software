cmake_minimum_required(VERSION 3.12)

set (CMAKE_CXX_STANDARD 17)

set(EIGEN_PATH ../../deps/Eigen3)

set(INCLUDE_LIST
  ../../include
  ${EIGEN_PATH}
)

add_library(autd3capi-link-emulator SHARED
  c_api.cpp
  emulator_link.h
  ../base/wrapper_link.hpp
) 

include_directories(${INCLUDE_LIST})

target_link_libraries(autd3capi-link-emulator ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(autd3capi-link-emulator link_emulator)
if(WIN32)
  target_link_libraries(autd3capi-link-emulator ws2_32.lib)
  target_link_libraries(autd3capi-link-emulator winmm.lib)
endif()

set_target_properties(autd3capi-link-emulator
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

if(NOT WIN32)
  set_target_properties(autd3capi-link-emulator PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
