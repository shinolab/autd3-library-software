cmake_minimum_required(VERSION 3.1)

set (CMAKE_CXX_STANDARD 11)

# Eigen3
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/deps)
find_package( Eigen3 REQUIRED )
include_directories( ${EIGEN3_INCLUDE_DIR} )

add_executable(simple simple.cpp)
include_directories(../include)
target_link_libraries(simple autd3)