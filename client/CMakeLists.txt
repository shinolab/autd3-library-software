cmake_minimum_required(VERSION 3.12)

set(CMAKE_CONFIGURATION_TYPES "Debug;Release" )
set(CMAKE_SUPPRESS_REGENERATION true)

option (USE_UNITY "Use Unity" OFF)
option (ENABLE_TESTS "Include Test Project" OFF)

project(autd3)
set(VERSION_MAJOR   3   CACHE STRING "Project major version number.")
set(VERSION_MINOR   3   CACHE STRING "Project minor version number.")
set(VERSION_PATCH   0   CACHE STRING "Project patch version number.")
mark_as_advanced(VERSION_MAJOR VERSION_MINOR VERSION_PATCH)

set(CMAKE_CSharp_FLAGS "/langversion:latest")

ADD_CUSTOM_TARGET(cpplint
  python
  ${PROJECT_SOURCE_DIR}/deps/cpplint.py
  --filter=-runtime/rtti,-build/include_order,-build/c++11
  --linelength=150
  ${PROJECT_SOURCE_DIR}/lib/*
  )

# ADS
add_subdirectory(deps/BeckhoffADS)

add_subdirectory(lib)
add_subdirectory(csharp)
add_subdirectory(doc)
add_subdirectory(example)
add_subdirectory(csharp_example)
add_subdirectory(capi)
if (ENABLE_TESTS)
	add_subdirectory(tests)
endif()

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT AUTD3SharpSample)