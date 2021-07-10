cmake_minimum_required(VERSION 3.12)

set(INCLUDE_LIST
  ${PROJECT_SOURCE_DIR}/include
  ${EIGEN_PATH}
)

add_library(autd3capi-emulator-link SHARED
  c_api.cpp
  emulator_link.h
  ../base/header.h
  ../base/wrapper_link.hpp
)

include_directories(${INCLUDE_LIST})

target_link_libraries(autd3capi-emulator-link emulator_link autd3)
if(WIN32)
  target_link_libraries(autd3capi-emulator-link ws2_32.lib winmm.lib)
elseif(APPLE)
else()
  target_link_libraries(autd3capi-emulator-link -static-libgcc -static-libstdc++)
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
