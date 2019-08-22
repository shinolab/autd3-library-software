cmake_minimum_required(VERSION 3.12)

set (CMAKE_CXX_STANDARD 11)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/deps)
set(Matlab_INCLUDE_DIRS "")

cmake_policy(SET CMP0074 NEW)

find_package( Eigen3 REQUIRED )
find_package( Boost REQUIRED )
find_package( Threads REQUIRED )
find_package( MATLAB )

include_directories(../include ${EIGEN3_INCLUDE_DIR} ${Boost_INCLUDE_DIR} ${Ads_INCLUDE_DIR} ${Matlab_INCLUDE_DIRS})

add_library(autd3 STATIC
  ethercat_link.cpp
  soem_link.cpp
 
  gain.cpp
  matlabgain.cpp
  holo_gain.cpp
  groupedgain.cpp
 
  modulation.cpp
 
  controller.cpp
  geometry.cpp
 
  timer.cpp
 
  link.hpp
  timer.hpp
  ethercat_link.hpp
  soem_link.hpp
  privdef.hpp
  
  ../include/core.hpp
  ../include/controller.hpp
  ../include/gain.hpp
  ../include/modulation.hpp
  ../include/geometry.hpp
  ../include/autd3.hpp
)

if (Matlab_FOUND AND Matlab_MX_LIBRARY)
    add_definitions(-DMATLAB_ENABLED)
    target_link_libraries(autd3 ${Matlab_MX_LIBRARY} ${Matlab_MAT_LIBRARY})
endif()

source_group("link" FILES
  link.cpp
  ethercat_link.cpp
  soem_link.cpp
)

source_group("gain" FILES
  gain.cpp
  matlabgain.cpp
  holo_gain.cpp
  groupedgain.cpp
)

source_group("modulation" FILES
  modulation.cpp
)

source_group("controller" FILES
  controller.cpp
  geometry.cpp
)

source_group("util" FILES
  timer.cpp
)

target_link_libraries(autd3 ${Boost_LIBRARIES} ${Ads_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4068")