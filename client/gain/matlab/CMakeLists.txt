cmake_minimum_required(VERSION 3.10)

set (CMAKE_CXX_STANDARD 17)

include_directories(${PROJECT_SOURCE_DIR}/include ${EIGEN_PATH})

add_library(gain_matlab STATIC
  matlab.cpp
  ${PROJECT_SOURCE_DIR}/include/gain/matlab.hpp
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

set_target_properties(gain_matlab
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

if(NOT WIN32)
  set_target_properties(gain_matlab PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
