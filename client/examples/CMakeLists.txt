cmake_minimum_required(VERSION 3.1)

set (CMAKE_CXX_STANDARD 17)

include_directories(${PROJECT_SOURCE_DIR}/include ${EIGEN_PATH} examples)

file(GLOB hpp_files examples/*.hpp runner.hpp)

link_libraries(autd3)
if(BUILD_HOLO_GAIN)
  link_libraries(holo_gain)
  add_definitions(-DBUILD_HOLO_GAIN)
endif()
if(WIN32)
  link_libraries(ws2_32.lib)
  link_libraries(winmm.lib)
elseif(APPLE)
  link_libraries(pcap)
else()
  link_libraries(rt)
endif()
  
if(BUILD_SOEM_LINK)
  add_executable(example_soem soem.cpp ${hpp_files})
  target_link_libraries(example_soem soem_link)
endif()

if(WIN32)
  if(BUILD_TWINCAT_LINK)
    add_executable(example_twincat twincat.cpp ${hpp_files})
    target_link_libraries(example_twincat twincat_link)
  endif()  
endif()
