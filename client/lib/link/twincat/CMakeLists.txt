cmake_minimum_required(VERSION 3.16)

add_library(twincat_link
  twincat.cpp
  ${PROJECT_SOURCE_DIR}/include/autd3/link/twincat.hpp
)
target_include_directories(twincat_link PUBLIC ${PROJECT_SOURCE_DIR}/include)

set_target_properties(twincat_link
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

if(WIN32)
  set_target_properties(twincat_link PROPERTIES FOLDER "link")
else()
  set_target_properties(twincat_link PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
