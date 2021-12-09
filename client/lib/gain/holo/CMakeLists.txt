cmake_minimum_required(VERSION 3.16)

add_library(holo_gain
  holo.cpp
  ${PROJECT_SOURCE_DIR}/include/autd3/gain/holo.hpp
)
target_include_directories(holo_gain PUBLIC ${PROJECT_SOURCE_DIR}/include ${EIGEN_PATH})
set_target_properties(holo_gain
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)
if(WIN32)
  set_target_properties(holo_gain PROPERTIES FOLDER "gain")
else()
  set_target_properties(holo_gain PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

add_subdirectory(eigen)
target_link_libraries(holo_gain INTERFACE eigen_backend)

# BLAS backend
if(BUILD_BLAS_BACKEND)
  set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${BLAS_LIB_DIR}")
  set(BLA_STATIC ON)
  find_package(BLAS REQUIRED)
  add_subdirectory(blas)
  target_link_libraries(holo_gain INTERFACE blas_backend)
endif()

# CUDA backend
if(BUILD_CUDA_BACKEND)
  find_package(CUDA REQUIRED)
  find_package(CUDAToolkit REQUIRED)
  if(CUDA_FOUND AND CUDAToolkit_FOUND)
    message("CUDA version: " ${CUDA_VERSION})
    message("CUDA toolkit version: " ${CUDAToolkit_VERSION})
    add_subdirectory(cuda)
    target_link_libraries(holo_gain INTERFACE cuda_backend)
  endif()
endif()

# ArrayFire backend
if(BUILD_ARRAYFIRE_BACKEND)
  find_package(ArrayFire REQUIRED)
  if(ArrayFire_FOUND)
    message("ArrayFire version: " ${ArrayFire_VERSION})
    add_subdirectory(array_fire)
    target_link_libraries(holo_gain INTERFACE arrayfire_backend)
  endif()
endif()
