cmake_minimum_required(VERSION 3.20)

add_library(autd3capi-from-file-modulation SHARED
  c_api.cpp
  from_file_modulation.h
  ../base/header.h
  ../base/wrapper_modulation.hpp
)

target_link_libraries(autd3capi-from-file-modulation modulation_from_file)
if(WIN32)
elseif(APPLE)
else()
  target_link_libraries(autd3capi-from-file-modulation -static-libgcc -static-libstdc++)
endif()

set_target_properties(autd3capi-from-file-modulation
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

if(WIN32)
  set_target_properties(autd3capi-from-file-modulation PROPERTIES FOLDER "capi")
else()
  set_target_properties(autd3capi-from-file-modulation PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
