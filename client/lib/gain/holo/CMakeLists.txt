cmake_minimum_required(VERSION 3.10)

add_subdirectory(eigen)

# BLAS backend
if(BUILD_BLAS_BACKEND)
  set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${BLAS_LIB_DIR}")
  set(BLA_STATIC ON)
  find_package(BLAS REQUIRED)
  if(USE_MKL)
    add_definitions(-DUSE_BLAS_MKL)
  endif()
  add_subdirectory(blas)
endif()

include_directories(${PROJECT_SOURCE_DIR}/include)

add_library(gain_holo STATIC
    holo.cpp
    ${PROJECT_SOURCE_DIR}/include/autd3/gain/linalg_backend.hpp
    ${PROJECT_SOURCE_DIR}/include/autd3/gain/holo.hpp
)

include_directories(${EIGEN_PATH})

target_link_libraries(gain_holo INTERFACE backend_eigen)

set_target_properties(gain_holo
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

if(WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4190")
else()
    set_target_properties(gain_holo PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
