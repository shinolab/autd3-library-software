cmake_minimum_required(VERSION 3.1)

set (CMAKE_CXX_STANDARD 11)
set(EIGEN_PATH ../deps/Eigen3)

# Eigen3
include_directories( ${EIGEN_PATH} )

add_executable(simple_soem simple_soem.cpp)
include_directories(../include)
target_link_libraries(simple_soem autd3)
target_link_libraries(simple_soem autdsoem)
if(WIN32)
    target_link_libraries(simple_soem ws2_32.lib)
    target_link_libraries(simple_soem winmm.lib)
elseif(APPLE)
    target_link_libraries(simple_soem pcap)
else()
    target_link_libraries(simple_soem rt)
endif()
