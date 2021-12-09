cmake_minimum_required(VERSION 3.16)

file(GLOB include_files ${PROJECT_SOURCE_DIR}/include/*.hpp ${PROJECT_SOURCE_DIR}/include/autd3/*.hpp)
file(GLOB core_include_files ${PROJECT_SOURCE_DIR}/include/autd3/core/*.hpp)

set(SOURCE_LIST
  controller.cpp
  gain/primitive.cpp
  modulation/primitive.cpp
  sequence/primitive.cpp
  ${include_files}
  ${core_include_files}
)

add_library(autd3 ${SOURCE_LIST})
target_include_directories(autd3 PUBLIC ${PROJECT_SOURCE_DIR}/include ${EIGEN_PATH})
if(WIN32)
  target_link_libraries(autd3 ${AUTD_LINK_LIBRARIES_KEYWORD} winmm)
endif()

if(WIN32)
  target_sources(autd3 PRIVATE ${PROJECT_SOURCE_DIR}/include/autd3/gain/primitive.hpp)
  source_group("gain" FILES
    gain/primitive.cpp
    ${PROJECT_SOURCE_DIR}/include/autd3/gain/primitive.hpp
  )
  target_sources(autd3 PRIVATE ${PROJECT_SOURCE_DIR}/include/autd3/modulation/primitive.hpp)
  source_group("modulation" FILES
    modulation/primitive.cpp
    ${PROJECT_SOURCE_DIR}/include/autd3/modulation/primitive.hpp
  )
  target_sources(autd3 PRIVATE ${PROJECT_SOURCE_DIR}/include/autd3/sequence/primitive.hpp)
  source_group("sequence" FILES
    sequence/primitive.cpp
    ${PROJECT_SOURCE_DIR}/include/autd3/sequence/primitive.hpp
  )
  source_group("controller" FILES
    controller.cpp
    ${PROJECT_SOURCE_DIR}/include/autd3/controller.hpp
  )
  source_group("core" FILES
    ${core_include_files}
  )
endif()

if (NOT WIN32)
  set_target_properties(autd3 PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
