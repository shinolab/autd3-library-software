cmake_minimum_required(VERSION 3.10)

project(autd3)
set(VERSION_MAJOR   0   CACHE STRING "Project major version number.")
set(VERSION_MINOR   8   CACHE STRING "Project minor version number.")
set(VERSION_PATCH   1   CACHE STRING "Project patch version number.")
mark_as_advanced(VERSION_MAJOR VERSION_MINOR VERSION_PATCH)

option(DISABLE_MATLAB "DISABLE_MATLAB" OFF)
option(USE_DOUBLE "USE_DOUBLE" OFF)

if(DISABLE_MATLAB)
  message(STATUS "Disabled Matlab")
endif()

if(USE_DOUBLE)
  message(STATUS "Use double precision floating point number")
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

file(GLOB_RECURSE lint_files lib/*.cpp lib/*.hpp include/*.hpp capi/*.cpp capi/*.cpp soem_link/*.cpp soem_link/*.hpp)

ADD_CUSTOM_TARGET(cpplint
  python
  ${PROJECT_SOURCE_DIR}/deps/cpplint.py
  --filter=-runtime/references,-build/c++11,-whitespace/comments
  --linelength=150
  ${lint_files}
  )

# ADS
add_subdirectory(deps/BeckhoffADS)
# SOEM
add_subdirectory(deps/SOEM)

add_subdirectory(soem_link)
add_subdirectory(twincat_link)

add_subdirectory(lib)
add_subdirectory(capi)
add_subdirectory(doc)
if(NOT IGNORE_EXAMPLE)
    add_subdirectory(examples)
    if(WIN32)
      set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT example_soem)
    endif()
endif()
