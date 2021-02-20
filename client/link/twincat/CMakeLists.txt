cmake_minimum_required(VERSION 3.10)

set (CMAKE_CXX_STANDARD 17)

include_directories(
	../../include
	${Ads_INCLUDE_DIR}
)

add_library(link_twincat STATIC
    twincat.cpp
	../../include/link/twincat.hpp
)

target_link_libraries(link_twincat ${Ads_LIBRARIES})

if(NOT WIN32)
  set_target_properties(link_twincat PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
