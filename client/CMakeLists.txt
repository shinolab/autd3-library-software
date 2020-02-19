cmake_minimum_required(VERSION 3.10)

project(autd3)
set(VERSION_MAJOR   3   CACHE STRING "Project major version number.")
set(VERSION_MINOR   0   CACHE STRING "Project minor version number.")
set(VERSION_PATCH   2   CACHE STRING "Project patch version number.")
mark_as_advanced(VERSION_MAJOR VERSION_MINOR VERSION_PATCH)

ADD_CUSTOM_TARGET(cpplint
  python
  ${PROJECT_SOURCE_DIR}/deps/cpplint.py
  --filter=-runtime/rtti,-build/include_order,-build/c++11
  --linelength=150
  ${PROJECT_SOURCE_DIR}/lib/*
  )

#SOEM
add_subdirectory(libsoem)

if(WIN32)
    include(cmakes/win_vs.cmake)
    
    # ADS
    add_subdirectory(deps/BeckhoffADS)

    add_subdirectory(example)
    add_subdirectory(csharp)
    add_subdirectory(csharp_example)
    add_subdirectory(capi)
    if (ENABLE_TESTS)
      add_subdirectory(tests)
    endif()
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT AUTD3SharpSample)
elseif(APPLE)
    include(cmakes/mac.cmake)
else()
    include(cmakes/linux.cmake)
endif()

add_subdirectory(lib)
add_subdirectory(doc)
add_subdirectory(example_soem)
