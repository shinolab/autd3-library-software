cmake_minimum_required(VERSION 3.16)

add_library(autd3capi-blas-backend SHARED
  c_api.cpp
  blas_backend.h
  ../../base/header.h
  ../../base/wrapper.hpp
)

target_include_directories(autd3capi-blas-backend PRIVATE ${PROJECT_SOURCE_DIR}/include ${EIGEN_PATH})
target_link_libraries(autd3capi-blas-backend blas_backend)
if(WIN32)
elseif(APPLE)
else()
  target_link_libraries(autd3capi-holo-gain -static-libgcc -static-libstdc++)
endif()

set_target_properties(autd3capi-blas-backend
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

if(WIN32)
  set_target_properties(autd3capi-blas-backend PROPERTIES FOLDER "capi/gain/backend")
else()
  set_target_properties(autd3capi-blas-backend PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
