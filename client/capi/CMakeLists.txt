cmake_minimum_required(VERSION 3.12)

set (CMAKE_CXX_STANDARD 17)

if(WIN32)
	set(OS win32)
elseif(APPLE)
	set(OS macosx)
else()
	set(OS linux)
endif()

# Eigen3
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/deps)

cmake_policy(SET CMP0074 NEW)
cmake_policy(SET CMP0079 NEW)

find_package( Eigen3 REQUIRED )
find_package( Threads REQUIRED )

set(INCLUDE_LIST
  ../include
  ${EIGEN3_INCLUDE_DIR}
  ${Ads_INCLUDE_DIR}
)

add_library(autd3capi SHARED
  c_api.cpp
  
  ../lib/ethercat_link.cpp
  ../lib/soem_link.cpp
  
  ../lib/gain.cpp
  ../lib/matlabgain.cpp
  ../lib/holo_gain.cpp
  ../lib/groupedgain.cpp
  
  ../lib/modulation.cpp
  
  ../lib/controller.cpp
  ../lib/geometry.cpp
  
  ../lib/${OS}/timer.cpp
    
  ../lib/link.hpp
  ../lib/timer.hpp
  ../lib/ethercat_link.hpp
  ../lib/soem_link.hpp
  ../lib/privdef.hpp
  ../lib/ec_config.hpp
  
  ../include/core.hpp
  ../include/controller.hpp
  ../include/gain.hpp
  ../include/modulation.hpp
  ../include/geometry.hpp
  ../include/autd3.hpp
  ../include/autd3_c_api.h
)

include_directories(${INCLUDE_LIST})

target_link_libraries(autd3capi ${Ads_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(autd3capi libsoem)
target_link_libraries(autd3capi ws2_32.lib)
target_link_libraries(autd3capi winmm.lib)

set_target_properties(autd3capi
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/dll"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/dll"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/dll"
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4068")
add_compile_definitions(DLL_FOR_CAPI)
