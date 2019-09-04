cmake_minimum_required(VERSION 3.1)

set (CMAKE_CXX_STANDARD 11)

# Eigen3
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/deps)
find_package( Eigen3 REQUIRED )
include_directories( ${EIGEN3_INCLUDE_DIR} )

add_executable(simple_soem simple_soem.cpp)
include_directories(../include)
target_link_libraries(simple_soem autd3)
target_link_libraries(simple_soem libsoem)
if(WIN32)
    target_link_libraries(simple_soem ws2_32.lib)
    target_link_libraries(simple_soem winmm.lib)
else()
    target_link_libraries(simple_soem rt)
endif()
