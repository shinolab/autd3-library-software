cmake_minimum_required(VERSION 3.16)

add_library(emulator_link
  emulator.cpp
  ${PROJECT_SOURCE_DIR}/include/autd3/link/emulator.hpp
)
target_include_directories(emulator_link PUBLIC ${PROJECT_SOURCE_DIR}/include ${EIGEN_PATH})

if(WIN32)
  target_link_libraries(emulator_link ${AUTD_LINK_LIBRARIES_KEYWORD} ws2_32)
endif()

set_target_properties(emulator_link
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

if(WIN32)
  set_target_properties(emulator_link PROPERTIES FOLDER "link")
  target_compile_definitions(emulator_link PRIVATE WIN32_LEAN_AND_MEAN)
else()
  set_target_properties(emulator_link PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
