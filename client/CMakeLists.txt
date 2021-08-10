cmake_minimum_required(VERSION 3.10)

project(autd3)
set(VERSION_MAJOR   1   CACHE STRING "Project major version number.")
set(VERSION_MINOR   7   CACHE STRING "Project minor version number.")
set(VERSION_PATCH   1   CACHE STRING "Project patch version number.")
mark_as_advanced(VERSION_MAJOR VERSION_MINOR VERSION_PATCH)

set (CMAKE_CXX_STANDARD 17)

option(BUILD_ALL "BUILD_ALL" OFF)
option(BUILD_DOC "BUILD_DOC" OFF)

option(BUILD_HOLO_GAIN "BUILD_HOLO_GAIN" ON)
option(BUILD_HOLO_GAIN "BUILD_FROM_FILE_MODULATION" OFF)

option(BUILD_SOEM_LINK "BUILD_SOEM_LINK" ON)
option(BUILD_TWINCAT_LINK "BUILD_TWINCAT_LINK" OFF)
option(BUILD_EMULATOR_LINK "BUILD_EMULATOR_LINK" OFF)

option(IGNORE_EXAMPLE "IGNORE_EXAMPLE" OFF)
option(BUILD_CAPI "BUILD_CAPI" OFF)
option(ENABLE_LINT "ENABLE_LINT" OFF)

if(BUILD_ALL)
  set(BUILD_HOLO_GAIN ON)
  set(BUILD_FROM_FILE_MODULATION ON)
  set(BUILD_SOEM_LINK ON)
  set(BUILD_TWINCAT_LINK ON)
  set(BUILD_EMULATOR_LINK ON)
  set(BUILD_CAPI ON)
endif()

if(BUILD_DOC)
  find_package(Doxygen)
  if(DOXYGEN_FOUND)
    message(STATUS "Build doc")
  else()
    message(STATUS "Doxygen is needed to build the documentation. Skip building doc...")
    set(BUILD_DOC OFF)
  endif()
endif()

if(WIN32)
  set(OS win32)
  include(cmakes/win_vs.cmake)
elseif(APPLE)
  set(OS macosx)
  include(cmakes/mac.cmake)
else()
  set(OS linux)
  if(DEFINED BUILD_TARGET)
    if (${BUILD_TARGET} STREQUAL ARM32 OR ${BUILD_TARGET} STREQUAL ARM)
      include(cmakes/arm32.cmake)
    elseif(${BUILD_TARGET} STREQUAL ARM64 OR ${BUILD_TARGET} STREQUAL AARCH64)
      include(cmakes/arm64.cmake)
    elseif(${BUILD_TARGET} STREQUAL AMD64 OR ${BUILD_TARGET} STREQUAL x86_64 OR ${BUILD_TARGET} STREQUAL x64)
      include(cmakes/linux.cmake)
    endif()
  else()
    include(cmakes/linux.cmake)
  endif()
endif()

add_compile_definitions(_USE_MATH_DEFINES)
if(WIN32)
  add_compile_definitions(__STDC_LIMIT_MACROS)
endif()

find_package(Git)
if(NOT Git_FOUND)
  message(WARNING "Git not found.")
endif()

if(ENABLE_LINT)
  if(Git_FOUND)
  execute_process(COMMAND git -C ${PROJECT_SOURCE_DIR} submodule update --init --recursive -- ${PROJECT_SOURCE_DIR}/deps/cpplint)
  endif()
  file(GLOB_RECURSE lint_files lib/*.cpp lib/*.hpp include/*.hpp capi/*.cpp capi/*.hpp)
  ADD_CUSTOM_TARGET(cpplint
    python
    ${PROJECT_SOURCE_DIR}/deps/cpplint/cpplint.py
    --filter=-runtime/references,-build/c++11,-whitespace/comments,-readability/braces
    --linelength=150
    ${lint_files}
  )
endif()

if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

if(POLICY CMP0079)
  cmake_policy(SET CMP0079 NEW)
endif()

if(Git_FOUND)
  execute_process(COMMAND git -C ${PROJECT_SOURCE_DIR} submodule update --init --recursive -- ${PROJECT_SOURCE_DIR}/deps/Eigen3)
endif()
add_definitions(-DEIGEN_MPL2_ONLY)
set(EIGEN_PATH ${PROJECT_SOURCE_DIR}/deps/Eigen3)

# Gain
if(BUILD_HOLO_GAIN)
  add_subdirectory(lib/gain/holo)
endif()

# Modulation
if(BUILD_FROM_FILE_MODULATION)
  add_subdirectory(lib/modulation/from_file)
endif()

# Link
if(BUILD_TWINCAT_LINK)
  add_subdirectory(lib/link/twincat)
endif()
if(BUILD_SOEM_LINK)
  if(Git_FOUND)
    execute_process(COMMAND git -C ${PROJECT_SOURCE_DIR} submodule update --init --recursive -- ${PROJECT_SOURCE_DIR}/deps/SOEM)
  endif()
  set(SOEM_PATH ${PROJECT_SOURCE_DIR}/deps/SOEM)
  add_subdirectory(lib/link/soem)
endif()
if(BUILD_EMULATOR_LINK)
  add_subdirectory(lib/link/emulator)
endif()

add_subdirectory(lib)

if(BUILD_CAPI)
  add_subdirectory(capi/base)
  if(BUILD_SOEM_LINK)
    add_subdirectory(capi/soem_link)
  endif()
  if(BUILD_TWINCAT_LINK)
    add_subdirectory(capi/twincat_link)
  endif()
  if(BUILD_HOLO_GAIN)
    add_subdirectory(capi/holo_gain)
  endif()
  if(BUILD_FROM_FILE_MODULATION)
    add_subdirectory(capi/from_file_modulation)
  endif()
  if(BUILD_EMULATOR_LINK)
    add_subdirectory(capi/emulator_link)
  endif()
endif()

if(BUILD_DOC)
  add_subdirectory(doc)
endif()

if(NOT IGNORE_EXAMPLE)
  add_subdirectory(examples)
  if(WIN32 AND BUILD_SOEM_LINK)
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT example_soem)
  endif()
endif()
