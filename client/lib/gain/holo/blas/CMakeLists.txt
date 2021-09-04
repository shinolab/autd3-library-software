cmake_minimum_required(VERSION 3.10)

include_directories(${PROJECT_SOURCE_DIR}/include ${EIGEN_PATH} ${BLAS_INCLUDE_DIR})

add_library(backend_blas STATIC
    blas_backend.cpp
    ${PROJECT_SOURCE_DIR}/include/autd3/gain/linalg_backend.hpp
    ${PROJECT_SOURCE_DIR}/include/autd3/gain/blas_backend.hpp
)

target_link_directories(backend_blas INTERFACE ${BLAS_DEPEND_LIB_DIR})
target_link_libraries(backend_blas ${BLAS_LIBRARIES})
if(NOT WIN32)
  target_link_libraries(backend_blas gfortran)
endif()

set_target_properties(backend_blas
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

if(WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4190")
else()
    set_target_properties(backend_blas PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
