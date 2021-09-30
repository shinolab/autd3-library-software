cmake_minimum_required(VERSION 3.16)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --expt-relaxed-constexpr -Xcudafe "--diag_suppress=esa_on_defaulted_function_ignored")

if(BUILD_SHARED_LIBS)
  set(CUDA_LINK_LIBRARIES_KEYWORD PRIVATE)
else()
  set(CUDA_LINK_LIBRARIES_KEYWORD INTERFACE)
endif()
cuda_add_library(cuda_backend
    kernel.h
    kernel.cu
    cuda_backend.cu
    cuda_backend.cpp
    cuda_matrix.hpp
    ${PROJECT_SOURCE_DIR}/include/autd3/gain/backend.hpp
    ${PROJECT_SOURCE_DIR}/include/autd3/gain/cuda_backend.hpp
)
cuda_add_cublas_to_target(cuda_backend)
target_include_directories(cuda_backend PRIVATE ${PROJECT_SOURCE_DIR}/lib/gain/holo ${PROJECT_SOURCE_DIR}/include ${EIGEN_PATH})
target_include_directories(cuda_backend INTERFACE ${CUDA_INCLUDE_DIRS})

target_link_libraries(cuda_backend ${AUTD_LINK_LIBRARIES_KEYWORD} ${CUDA_cusolver_LIBRARY})

set_target_properties(cuda_backend
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

if(WIN32)
  set_target_properties(cuda_backend PROPERTIES FOLDER "gain/backend")
else()
  set_target_properties(cuda_backend PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
