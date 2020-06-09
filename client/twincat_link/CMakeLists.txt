cmake_minimum_required(VERSION 3.10)

set (CMAKE_CXX_STANDARD 17)

if(WIN32)
	set(OS win32)
elseif(APPLE)
	set(OS macosx)
else()
	set(OS linux)
endif()

include_directories(
	../include
	${Ads_INCLUDE_DIR}
)

add_library(twincat_link STATIC
	twincat_link.cpp
	../include/twincat_link.hpp
)

target_link_libraries(twincat_link ${Ads_LIBRARIES})

if(NOT WIN32)
  set_target_properties(twincat_link PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
