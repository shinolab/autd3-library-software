cmake_minimum_required(VERSION 3.16)

add_library(autd3capi-holo-gain SHARED
  c_api.cpp
  holo_gain.h
  ../base/header.h
  ../base/wrapper.hpp
)

target_include_directories(autd3capi-holo-gain PRIVATE ${PROJECT_SOURCE_DIR}/include ${EIGEN_PATH})
target_link_libraries(autd3capi-holo-gain eigen_backend)
if(WIN32)
elseif(APPLE)
else()
  target_link_libraries(autd3capi-holo-gain -static-libgcc -static-libstdc++)
endif()

set_target_properties(autd3capi-holo-gain
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

if(WIN32)
  set_target_properties(autd3capi-holo-gain PROPERTIES FOLDER "capi/gain")
else()
  set_target_properties(autd3capi-holo-gain PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

if(BUILD_BLAS_BACKEND)
  add_subdirectory(blas)
endif()

if(BUILD_CUDA_BACKEND)
  add_subdirectory(cuda)
endif()

if(BUILD_ARRAYFIRE_BACKEND)
  add_subdirectory(arrayfire)
endif()
