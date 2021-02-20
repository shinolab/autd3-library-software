cmake_minimum_required(VERSION 3.12)

set (CMAKE_CXX_STANDARD 17)

set(EIGEN_PATH ../deps/Eigen3)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/deps)

if (POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

if (POLICY CMP0079)
  cmake_policy(SET CMP0079 NEW)
endif()

find_package( Threads REQUIRED )

set(INCLUDE_LIST
  ../include
  ${EIGEN_PATH}
  ${Ads_INCLUDE_DIR}
)

file(GLOB include_files  ../include/*.hpp)

add_library(autd3capi SHARED
  c_api.cpp
  autd3_c_api.h
  wrapper.hpp
  ${include_files}
)

include_directories(${INCLUDE_LIST})

target_link_libraries(autd3capi ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(autd3capi soem_link)
target_link_libraries(autd3capi twincat_link)
target_link_libraries(autd3capi autd3)
if(WIN32)
    target_link_libraries(autd3capi ws2_32.lib)
    target_link_libraries(autd3capi winmm.lib)
elseif(APPLE)
    target_link_libraries(autd3capi pcap)
else()
    target_link_libraries(autd3capi rt)
    target_link_libraries(autd3capi -static-libgcc -static-libstdc++)
endif()
set_target_properties(autd3capi
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

if (WIN32)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4068 /wd4819")
endif()

if(NOT WIN32)
  set_target_properties(autd3capi PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
