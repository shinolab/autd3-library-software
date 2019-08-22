cmake_minimum_required(VERSION 3.12)

set (CMAKE_CXX_STANDARD 11)

# Eigen3
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/deps)
set(Matlab_INCLUDE_DIRS "")

cmake_policy(SET CMP0074 NEW)

find_package( Eigen3 REQUIRED )
find_package( Boost REQUIRED )
find_package( Threads REQUIRED )
find_package( MATLAB )

include_directories(../include ${EIGEN3_INCLUDE_DIR} ${Boost_INCLUDE_DIR} ${Ads_INCLUDE_DIR} ${Matlab_INCLUDE_DIRS})

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
  
  ../lib/timer.cpp
    
  ../lib/link.hpp
  ../lib/timer.hpp
  ../lib/ethercat_link.hpp
  ../lib/soem_link.hpp
  ../lib/privdef.hpp
  
  ../include/core.hpp
  ../include/controller.hpp
  ../include/gain.hpp
  ../include/modulation.hpp
  ../include/geometry.hpp
  ../include/autd3.hpp
  ../include/autd3_c_api.h
)

if (Matlab_FOUND AND Matlab_MX_LIBRARY)
    add_definitions(-DMATLAB_ENABLED)
    target_link_libraries(autd3 ${Matlab_MX_LIBRARY} ${Matlab_MAT_LIBRARY})
endif()

if (USE_UNITY)
    add_custom_command(TARGET autd3capi
                 POST_BUILD
                 COMMAND copy /Y $(TargetPath) $(SolutionDir)autdunity\\Assets\\AUTD\\Plugins\\x86_64)
endif()

target_link_libraries(autd3capi ${Boost_LIBRARIES} ${Ads_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

set_target_properties(autd3capi
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/csharp_example"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/csharp_example"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/csharp_example"
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4068")
add_compile_definitions(DLL_FOR_CAPI)