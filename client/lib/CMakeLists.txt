cmake_minimum_required(VERSION 3.10)

set (CMAKE_CXX_STANDARD 17)

find_package(Threads REQUIRED)

file(GLOB cpp_files *.cpp)
file(GLOB hpp_files *.hpp)
file(GLOB include_files ${PROJECT_SOURCE_DIR}/include/*.hpp ${PROJECT_SOURCE_DIR}/include/linalg/*.hpp)

set(SOURCE_LIST 
  ${cpp_files}
  ${hpp_files}
  ${include_files}
)

add_library(autd3 STATIC ${SOURCE_LIST})

include_directories(${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/osal_timer)
if(NOT DISABLE_EIGEN)
  include_directories(${EIGEN_PATH})
endif()

if(WIN32)
  source_group("gain" FILES
    gain.cpp
    grouped_gain.cpp
    ${PROJECT_SOURCE_DIR}/include/gain.hpp
  )

  source_group("modulation" FILES
    modulation.cpp
    ${PROJECT_SOURCE_DIR}/include/modulation.hpp
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

  source_group("linalg" FILES
    ${PROJECT_SOURCE_DIR}/include/linalg.hpp
    ${PROJECT_SOURCE_DIR}/include/linalg/helper.hpp
    ${PROJECT_SOURCE_DIR}/include/linalg/matrix.hpp
    ${PROJECT_SOURCE_DIR}/include/linalg/quaternion.hpp
    ${PROJECT_SOURCE_DIR}/include/linalg/vector.hpp
  )
endif()

target_link_libraries(autd3 ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(autd3 osal_timer)

if (WIN32)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4068 /wd4819")
else()
  set_target_properties(autd3 PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
