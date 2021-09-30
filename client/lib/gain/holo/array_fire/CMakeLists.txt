cmake_minimum_required(VERSION 3.16)

add_library(arrayfire_backend
    arrayfire_backend.cpp
    arrayfire_matrix.hpp
    ${PROJECT_SOURCE_DIR}/include/autd3/gain/backend.hpp
    ${PROJECT_SOURCE_DIR}/include/autd3/gain/arrayfire_backend.hpp
)

target_include_directories(arrayfire_backend PRIVATE ${PROJECT_SOURCE_DIR}/lib/gain/holo ${PROJECT_SOURCE_DIR}/include ${EIGEN_PATH})
target_include_directories(arrayfire_backend PUBLIC ${ArrayFire_INCLUDE_DIRS})
target_link_libraries(arrayfire_backend ${AUTD_LINK_LIBRARIES_KEYWORD} ${ArrayFire_LIBRARIES})

set_target_properties(arrayfire_backend
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

if(WIN32)
  set_target_properties(arrayfire_backend PROPERTIES FOLDER "gain/backend")
else()
  set_target_properties(arrayfire_backend PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
