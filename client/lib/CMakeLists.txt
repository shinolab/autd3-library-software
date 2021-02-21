cmake_minimum_required(VERSION 3.10)

set (CMAKE_CXX_STANDARD 17)

find_package(Threads REQUIRED)

set(INCLUDE_LIST
  ${PROJECT_SOURCE_DIR}/include
  ${EIGEN_PATH}
)

file(GLOB cpp_files *.cpp)
file(GLOB hpp_files *.hpp)
file(GLOB include_files ${PROJECT_SOURCE_DIR}include/*.hpp)

set(SOURCE_LIST 
  ${cpp_files}
  ${hpp_files}
  ${OS}/timer.cpp
  ${include_files}
)

add_library(autd3 STATIC ${SOURCE_LIST})

include_directories(${INCLUDE_LIST})

if(WIN32)
  source_group("link" FILES
    emulator_link.cpp
    debug_link.cpp
    ${PROJECT_SOURCE_DIR}/include/emulator_link.hpp
    ${PROJECT_SOURCE_DIR}/include/debug_link.hpp
  )

  source_group("gain" FILES
    gain.cpp
    grouped_gain.cpp
    ${PROJECT_SOURCE_DIR}include/gain.hpp
  )

  source_group("modulation" FILES
    modulation.cpp
    ${PROJECT_SOURCE_DIR}include/modulation.hpp
  )

  source_group("sequence" FILES
    sequence.cpp
    ${PROJECT_SOURCE_DIR}include/sequence.hpp
  )
 
  source_group("controller" FILES
    controller.cpp
    autd_logic.cpp
    ${PROJECT_SOURCE_DIR}/include/controller.hpp
    autd_logic.hpp
  )

  source_group("geometry" FILES
    geometry.cpp
    ${PROJECT_SOURCE_DIR}include/geometry.hpp
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
