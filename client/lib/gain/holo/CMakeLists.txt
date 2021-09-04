cmake_minimum_required(VERSION 3.10)

include_directories(${PROJECT_SOURCE_DIR}/include ${EIGEN_PATH})

add_library(gain_holo STATIC
    holo.cpp
    ${PROJECT_SOURCE_DIR}/include/autd3/gain/linalg_backend.hpp
    ${PROJECT_SOURCE_DIR}/include/autd3/gain/holo.hpp
)

add_subdirectory(eigen)
target_link_libraries(gain_holo INTERFACE backend_eigen)

# BLAS backend
if(BUILD_BLAS_BACKEND)
  set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${BLAS_LIB_DIR}")
  set(BLA_STATIC ON)
  find_package(BLAS REQUIRED)
  if(USE_MKL)
    add_definitions(-DUSE_BLAS_MKL)
  endif()
  add_subdirectory(blas)
  target_link_libraries(gain_holo INTERFACE backend_blas)
endif()

# CUDA backend
if(BUILD_CUDA_BACKEND)
  find_package(CUDA REQUIRED)
  find_package(CUDAToolkit REQUIRED)
  if(CUDA_FOUND AND CUDAToolkit_FOUND)
    message("CUDA version: " ${CUDA_VERSION})
    message("CUDA toolkit version: " ${CUDAToolkit_VERSION})
    add_subdirectory(cuda)
    target_link_libraries(gain_holo INTERFACE backend_cuda)
  endif()
endif()

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
