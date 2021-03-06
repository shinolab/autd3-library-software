cmake_minimum_required(VERSION 3.10)

set (CMAKE_CXX_STANDARD 17)

include_directories(${PROJECT_SOURCE_DIR}/include)

add_library(gain_holo STATIC
linalg_backend.cpp
${PROJECT_SOURCE_DIR}/include/gain/linalg_backend.hpp
${PROJECT_SOURCE_DIR}/include/gain/holo.hpp
)

if(ENABLE_EIGEN)
  include_directories(${EIGEN_PATH})
endif()

if(ENABLE_BLAS)
  include_directories(${BLAS_INCLUDE_DIR})
  target_link_libraries(gain_holo ${BLAS_LIBRARIES})
endif()

set_target_properties(gain_holo
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

target_link_directories(gain_holo PUBLIC ${BLAS_DEPEND_LIB_DIR})
if (WIN32)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4190")
else()
  set_target_properties(gain_holo PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
