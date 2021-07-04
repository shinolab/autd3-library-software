cmake_minimum_required(VERSION 3.10)

file(GLOB include_files ${PROJECT_SOURCE_DIR}/include/*.hpp ${PROJECT_SOURCE_DIR}/include/**/*.hpp)

set(SOURCE_LIST
  controller.cpp
  gain/primitive.cpp
  modulation/primitive.cpp
  sequence/primitive.cpp
  ${include_files}
)

include_directories(${PROJECT_SOURCE_DIR}/include ${EIGEN_PATH})

add_library(autd3 STATIC ${SOURCE_LIST})

if(WIN32)
  source_group("gain" FILES
    gain/primitive.cpp
    ${PROJECT_SOURCE_DIR}/include/autd3/gain/primitive.hpp
  )

  source_group("modulation" FILES
    modulation/primitive.cpp
    ${PROJECT_SOURCE_DIR}/include/autd3/modulation/primitive.hpp
  )

  source_group("sequence" FILES
    sequence/primitive.cpp
    ${PROJECT_SOURCE_DIR}/include/autd3/sequence/primitive.hpp
  )

  source_group("controller" FILES
    controller.cpp
    ${PROJECT_SOURCE_DIR}/include/autd3/controller.hpp
  )

  source_group("core" FILES
    ${PROJECT_SOURCE_DIR}/include/autd3/core/*.hpp
  )
endif()

if (WIN32)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4068")
else()
  set_target_properties(autd3 PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
