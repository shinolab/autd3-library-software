cmake_minimum_required(VERSION 3.10)

set (CMAKE_CXX_STANDARD 17)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/deps)

if (POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

find_package( Eigen3 REQUIRED )
find_package( Boost REQUIRED )
find_package( Threads REQUIRED )

include_directories(../include ${EIGEN3_INCLUDE_DIR} ${Boost_INCLUDE_DIR})

add_library(autd3 STATIC
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
  soem_link.hpp
  privdef.hpp
  
  ../include/core.hpp
  ../include/controller.hpp
  ../include/gain.hpp
  ../include/modulation.hpp
  ../include/geometry.hpp
  ../include/autd3.hpp
)

if(WIN32)
  include_directories(${Ads_INCLUDE_DIR})
  add_library(autd3 STATIC
    ethercat_link.cpp
    ethercat_link.hpp
  )
endif()


if(WIN32)
  set(Matlab_INCLUDE_DIRS "")
  find_package( MATLAB )
  include_directories(${Matlab_INCLUDE_DIRS})
  if (Matlab_FOUND AND Matlab_MX_LIBRARY)
      add_definitions(-DMATLAB_ENABLED)
      target_link_libraries(autd3 ${Matlab_MX_LIBRARY} ${Matlab_MAT_LIBRARY})
  endif()
endif()

if(WIN32)
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
endif()

target_link_libraries(autd3 ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

if (WIN32)
  target_link_libraries(${Ads_LIBRARIES})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4068")
endif()
