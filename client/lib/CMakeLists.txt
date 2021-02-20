cmake_minimum_required(VERSION 3.10)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/deps)

set (CMAKE_CXX_STANDARD 17)

set(EIGEN_PATH ../deps/Eigen3)

find_package(Threads REQUIRED)

set(INCLUDE_LIST
  ../include
  ${EIGEN_PATH}
)

file(GLOB cpp_files *.cpp)
file(GLOB hpp_files *.hpp)
file(GLOB include_files  ../include/*.hpp)

set(SOURCE_LIST 
  ${cpp_files}
  ${hpp_files}
  ${OS}/timer.cpp
  ${include_files}
)

add_library(autd3 STATIC ${SOURCE_LIST})
if(USE_DOUBLE)
  add_definitions(-DUSE_DOUBLE_AUTD)
endif()

include_directories(${INCLUDE_LIST})

if(WIN32)
  source_group("link" FILES
    emulator_link.cpp
    debug_link.cpp
    ../include/emulator_link.hpp
    ../include/debug_link.hpp
  )

  source_group("gain" FILES
    gain.cpp
    grouped_gain.cpp
    ../include/gain.hpp
  )

  source_group("modulation" FILES
    modulation.cpp
    ../include/modulation.hpp
  )

  source_group("sequence" FILES
    sequence.cpp
    ../include/sequence.hpp
  )
 
  source_group("controller" FILES
    controller.cpp
    autd_logic.cpp
    ../include/controller.hpp
    autd_logic.hpp
  )

  source_group("geometry" FILES
    geometry.cpp
    ../include/geometry.hpp
  )

  source_group("util" FILES
    ${OS}/timer.cpp
    timer.hpp
  )
endif()

target_link_libraries(autd3 ${CMAKE_THREAD_LIBS_INIT})

if (WIN32)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4068 /wd4819")
else()
  set_target_properties(autd3 PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
