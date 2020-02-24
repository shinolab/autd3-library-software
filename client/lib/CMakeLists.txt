cmake_minimum_required(VERSION 3.10)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/deps)

set (CMAKE_CXX_STANDARD 17)

if(WIN32)
	set(OS win32)
elseif(APPLE)
	set(OS macosx)
else()
	set(OS linux)
endif()

if (POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

if (POLICY CMP0079)
  cmake_policy(SET CMP0079 NEW)
endif()

find_package( Eigen3 REQUIRED )
find_package( Threads REQUIRED )

set(INCLUDE_LIST
  ../include
  ${EIGEN3_INCLUDE_DIR}
)

set(SOURCE_LIST 
  soem_link.cpp
 
  gain.cpp
  matlabgain.cpp
  holo_gain.cpp
  groupedgain.cpp
 
  modulation.cpp
 
  controller.cpp
  geometry.cpp
 
  ${OS}/timer.cpp
 
  link.hpp
  timer.hpp
  soem_link.hpp
  privdef.hpp
  ec_config.hpp
  
  ../include/core.hpp
  ../include/controller.hpp
  ../include/gain.hpp
  ../include/modulation.hpp
  ../include/geometry.hpp
  ../include/autd3.hpp
)

if(WIN32)
  list(APPEND INCLUDE_LIST
    ${Ads_INCLUDE_DIR}
  )
  
  list(APPEND SOURCE_LIST
  	ethercat_link.cpp
    ethercat_link.hpp
  )
endif()

add_library(autd3 STATIC ${SOURCE_LIST})

if(WIN32 AND (NOT DISABLE_MATLAB))
  set(Matlab_INCLUDE_DIRS "")
  find_package( MATLAB )
  list(APPEND INCLUDE_LIST
    ${Matlab_INCLUDE_DIRS}
  )
  if (Matlab_FOUND AND Matlab_MX_LIBRARY)
      add_definitions(-DMATLAB_ENABLED)
      target_link_libraries(autd3 ${Matlab_MX_LIBRARY} ${Matlab_MAT_LIBRARY})
  endif()
endif()

include_directories(${INCLUDE_LIST})

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
    ${OS}/timer.cpp
  )
endif()

target_link_libraries(autd3 ${CMAKE_THREAD_LIBS_INIT})

if (WIN32)
  target_link_libraries(autd3 ${Ads_LIBRARIES})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4068")
endif()
