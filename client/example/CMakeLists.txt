cmake_minimum_required(VERSION 3.1)

set (CMAKE_CXX_STANDARD 11)
set(EIGEN_PATH ../deps/Eigen3)

# Eigen3
include_directories( ${EIGEN_PATH} )

add_executable(simple simple.cpp)
include_directories(../include)
target_link_libraries(simple autd3)
target_link_libraries(simple libsoem)
if(WIN32)
    target_link_libraries(simple ws2_32.lib)
    target_link_libraries(simple winmm.lib)
elseif(APPLE)
    target_link_libraries(simple pcap)
else()
    target_link_libraries(simple rt)
endif()
