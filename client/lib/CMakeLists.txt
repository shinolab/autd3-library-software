cmake_minimum_required(VERSION 3.10)

set (CMAKE_CXX_STANDARD 17)

file(GLOB cpp_files *.cpp)
file(GLOB hpp_files *.hpp)
file(GLOB include_files ${PROJECT_SOURCE_DIR}/include/*.hpp ${PROJECT_SOURCE_DIR}/include/core/*.hpp)

set(SOURCE_LIST 
  ${cpp_files}
  ${hpp_files}
  ${include_files}
)

include_directories(${PROJECT_SOURCE_DIR}/include ${EIGEN_PATH})
if(WIN32)
  include_directories(${PROJECT_SOURCE_DIR}/osal_timer/win32)
elseif(APPLE)
  include_directories(${PROJECT_SOURCE_DIR}/osal_timer/macosx)
else()
  include_directories(${PROJECT_SOURCE_DIR}/linux/win32)
endif()

add_library(autd3 STATIC ${SOURCE_LIST})

if(WIN32)
  source_group("gain" FILES
    primitive_gain.cpp
    grouped_gain.cpp
    ${PROJECT_SOURCE_DIR}/include/primitive_gain.hpp
  )

  source_group("modulation" FILES
    primitive_modulation.cpp
    ${PROJECT_SOURCE_DIR}/include/primitive_modulation.hpp
  )

  source_group("controller" FILES
    controller.cpp
    autd_logic.cpp
    ${PROJECT_SOURCE_DIR}/include/controller.hpp
    autd_logic.hpp
  )

  source_group("geometry" FILES
    geometry.cpp
    ${PROJECT_SOURCE_DIR}/include/geometry.hpp
  )

  source_group("core" FILES
    ${PROJECT_SOURCE_DIR}/include/core/*.hpp
  )
endif()

if (WIN32)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4068")
else()
  set_target_properties(autd3 PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
