cmake_minimum_required(VERSION 3.20)

cuda_add_library(backend_cuda
    cuda_backend.cu
    ${PROJECT_SOURCE_DIR}/include/autd3/gain/linalg_backend.hpp
    ${PROJECT_SOURCE_DIR}/include/autd3/gain/cuda_backend.hpp
)
cuda_add_cublas_to_target(backend_cuda)
target_include_directories(backend_cuda INTERFACE ${CUDA_INCLUDE_DIRS})

set_target_properties(backend_cuda
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

if(WIN32)
  set_target_properties(backend_cuda PROPERTIES FOLDER "gain/backend")
else()
  set_target_properties(backend_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
