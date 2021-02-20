cmake_minimum_required(VERSION 3.10)

set (CMAKE_CXX_STANDARD 17)

set(EIGEN_PATH ../../deps/Eigen3)

include_directories(../../include ${EIGEN_PATH})

add_library(gain_matlab STATIC
  matlab.cpp
  ../../include/gain/matlab.hpp
)

find_package(Matlab)
if (Matlab_FOUND AND Matlab_MX_LIBRARY)
  list(APPEND INCLUDE_LIST
  	${Matlab_INCLUDE_DIRS}
  )
  target_link_libraries(gain_matlab ${Matlab_MX_LIBRARY} ${Matlab_MAT_LIBRARY})
  add_definitions(-DMATLAB_ENABLED)
endif()

include_directories(${INCLUDE_LIST})

if(NOT WIN32)
  set_target_properties(gain_matlab PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
