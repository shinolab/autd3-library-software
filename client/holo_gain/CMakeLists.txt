cmake_minimum_required(VERSION 3.10)


include_directories(${PROJECT_SOURCE_DIR}/include)

add_library(holo_gain STATIC
    holo_gain.cpp
    eigen_backend.cpp
    ${PROJECT_SOURCE_DIR}/include/linalg_backend.hpp
    ${PROJECT_SOURCE_DIR}/include/eigen_backend.hpp
    ${PROJECT_SOURCE_DIR}/include/holo_gain.hpp
)

include_directories(${EIGEN_PATH})

set_target_properties(holo_gain
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

if(WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4190")
else()
    set_target_properties(holo_gain PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
