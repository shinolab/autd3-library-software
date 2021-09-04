cmake_minimum_required(VERSION 3.20)

add_library(eigen_backend
    eigen_backend.cpp
    ${PROJECT_SOURCE_DIR}/include/autd3/gain/linalg_backend.hpp
    ${PROJECT_SOURCE_DIR}/include/autd3/gain/eigen_backend.hpp
)
target_include_directories(eigen_backend PUBLIC ${PROJECT_SOURCE_DIR}/include ${EIGEN_PATH})

set_target_properties(eigen_backend
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

if(WIN32)
  set_target_properties(eigen_backend PROPERTIES FOLDER "gain/backend")
else()
  set_target_properties(eigen_backend PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
