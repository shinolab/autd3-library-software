cmake_minimum_required(VERSION 3.10)

set (CMAKE_CXX_STANDARD 17)

file(GLOB cpp_files *.cpp)
file(GLOB hpp_files *.hpp)

set(SOURCE_LIST 
  ${cpp_files}
  ${hpp_files}
  ${OS}/timer.cpp
  ${include_files}
)

include_directories(${PROJECT_SOURCE_DIR}/include)
add_library(osal_timer STATIC ${SOURCE_LIST})

if(WIN32)
  target_link_libraries(osal_timer ws2_32.lib)
  target_link_libraries(osal_timer winmm.lib)
elseif(APPLE)
  target_link_libraries(osal_timer pcap)
else()
  target_link_libraries(osal_timer rt)
  target_link_libraries(osal_timer -static-libgcc -static-libstdc++)
endif()

set_target_properties(osal_timer
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

if (NOT WIN32)
  set_target_properties(osal_timer PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
