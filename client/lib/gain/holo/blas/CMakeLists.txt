cmake_minimum_required(VERSION 3.16)

add_library(blas_backend
    blas_backend.cpp
    blas_matrix.hpp
    ${PROJECT_SOURCE_DIR}/include/autd3/gain/backend.hpp
    ${PROJECT_SOURCE_DIR}/include/autd3/gain/blas_backend.hpp
)

target_include_directories(blas_backend PRIVATE ${BLAS_INCLUDE_DIR} ${PROJECT_SOURCE_DIR}/lib/gain/holo ${PROJECT_SOURCE_DIR}/include ${EIGEN_PATH})
if(USE_MKL)
  target_compile_definitions(blas_backend PRIVATE USE_BLAS_MKL)
endif()

target_link_directories(blas_backend ${AUTD_LINK_LIBRARIES_KEYWORD} ${BLAS_DEPEND_LIB_DIR} eigen_backend)
target_link_libraries(blas_backend ${AUTD_LINK_LIBRARIES_KEYWORD} ${BLAS_LIBRARIES})
if(NOT WIN32)
  target_link_libraries(blas_backend ${AUTD_LINK_LIBRARIES_KEYWORD} gfortran)
endif()

set_target_properties(blas_backend
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

if(WIN32)
  set_target_properties(blas_backend PROPERTIES FOLDER "gain/backend")
else()
  set_target_properties(blas_backend PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
