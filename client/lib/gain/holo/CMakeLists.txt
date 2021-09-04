cmake_minimum_required(VERSION 3.20)

add_library(gain_holo
    holo.cpp
    ${PROJECT_SOURCE_DIR}/include/autd3/gain/linalg_backend.hpp
    ${PROJECT_SOURCE_DIR}/include/autd3/gain/holo.hpp
)
target_include_directories(gain_holo PUBLIC ${PROJECT_SOURCE_DIR}/include ${EIGEN_PATH})

add_subdirectory(eigen)
target_link_libraries(gain_holo INTERFACE backend_eigen)

# BLAS backend
if(BUILD_BLAS_BACKEND)
  set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${BLAS_LIB_DIR}")
  set(BLA_STATIC ON)
  find_package(BLAS REQUIRED)
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
  set_target_properties(gain_holo PROPERTIES FOLDER "gain")
else()
  set_target_properties(gain_holo PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
