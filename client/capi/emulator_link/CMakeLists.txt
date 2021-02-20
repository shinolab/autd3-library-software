cmake_minimum_required(VERSION 3.12)

set (CMAKE_CXX_STANDARD 17)

set(EIGEN_PATH ../../deps/Eigen3)

if (POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

if (POLICY CMP0079)
  cmake_policy(SET CMP0079 NEW)
endif()

set(INCLUDE_LIST
  ../../include
  ${EIGEN_PATH}
)

add_library(autd3capi-emulator-link SHARED
  c_api.cpp
  autd3_c_api_emulator_link.h
  ../base/wrapper_link.hpp
) 

include_directories(${INCLUDE_LIST})

target_link_libraries(autd3capi-emulator-link ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(autd3capi-emulator-link emulator_link)
if(WIN32)
  target_link_libraries(autd3capi-emulator-link ws2_32.lib)
  target_link_libraries(autd3capi-emulator-link winmm.lib)
endif()

set_target_properties(autd3capi-emulator-link
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

if(NOT WIN32)
  set_target_properties(autd3capi-emulator-link PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
