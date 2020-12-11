cmake_minimum_required(VERSION 3.10)

project(autd3)
set(VERSION_MAJOR   0   CACHE STRING "Project major version number.")
set(VERSION_MINOR   7   CACHE STRING "Project minor version number.")
set(VERSION_PATCH   2   CACHE STRING "Project patch version number.")
mark_as_advanced(VERSION_MAJOR VERSION_MINOR VERSION_PATCH)

if(WIN32)
	set(OS win32)
elseif(APPLE)
	set(OS macosx)
else()
	set(OS linux)
endif()

file(GLOB_RECURSE lint_files lib/*.cpp lib/*.hpp include/*.hpp capi/*.cpp capi/*.cpp soem_link/*.cpp soem_link/*.hpp)

ADD_CUSTOM_TARGET(cpplint
  python
  ${PROJECT_SOURCE_DIR}/deps/cpplint.py
  --filter=-runtime/rtti,-build/c++11,-whitespace/comments
  --linelength=150
  ${lint_files}
  )

if(WIN32)
    include(cmakes/win_vs.cmake)
elseif(APPLE)
    include(cmakes/mac.cmake)
else()
    include(cmakes/linux.cmake)
endif()

# ADS
add_subdirectory(deps/BeckhoffADS)
# SOEM
add_subdirectory(deps/SOEM)

add_subdirectory(soem_link)
add_subdirectory(twincat_link)

add_subdirectory(capi)
add_subdirectory(lib)
add_subdirectory(doc)
if(NOT IGNORE_EXAMPLE)
    add_subdirectory(examples)
    if(WIN32)
      set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT example_soem)
    endif()
endif()

if(WIN32 AND ENABLE_TESTS)
    add_subdirectory(tests)
endif()
