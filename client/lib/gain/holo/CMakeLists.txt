cmake_minimum_required(VERSION 3.10)


include_directories(${PROJECT_SOURCE_DIR}/include)

add_library(gain_holo STATIC
    holo.cpp
    eigen_backend.cpp
    ${PROJECT_SOURCE_DIR}/include/autd3/gain/linalg_backend.hpp
    ${PROJECT_SOURCE_DIR}/include/autd3/gain/eigen_backend.hpp
    ${PROJECT_SOURCE_DIR}/include/autd3/gain/holo.hpp
)

include_directories(${EIGEN_PATH})

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
