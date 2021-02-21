cmake_minimum_required(VERSION 3.10)

set (CMAKE_CXX_STANDARD 17)

include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${ADS_PATH}/AdsLib
  ${ADS_PATH}/AdsLib/standalone
)

file(GLOB ads_src_files ${ADS_PATH}/AdsLib/*.cpp ${ADS_PATH}/AdsLib/standalone/*.cpp)

add_library(link_twincat STATIC
  ${ads_src_files}  
  twincat.cpp
  ${PROJECT_SOURCE_DIR}/include/link/twincat.hpp
)

set_target_properties(link_twincat
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

if(WIN32)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4267 /wd4996")
endif()

if(NOT WIN32)
  set_target_properties(link_twincat PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
