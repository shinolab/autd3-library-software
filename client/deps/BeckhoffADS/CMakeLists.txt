cmake_minimum_required(VERSION 3.1)

set (CMAKE_CXX_STANDARD 17)

execute_process(COMMAND git submodule update)
message("-- Updating git submodule done")

include_directories(./ADS/AdsLib ./ADS/AdsLib/standalone)

set (CMAKE_C_FLAGS "-pedantic -Wall")

file(GLOB src_files ADS/AdsLib/*.cpp ADS/AdsLib/standalone/*.cpp)

add_library(ads STATIC ${src_files})

if(APPLE)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -lc++")
elseif(WIN32)
  target_link_libraries(ads ws2_32)
endif()

set (Ads_LIBRARIES ads PARENT_SCOPE)
set (Ads_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ADS/AdsLib/ PARENT_SCOPE)

if (WIN32)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4267 /wd4996 /wd4834")
endif()

#INSTALL(TARGETS ads LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)

if(NOT WIN32)
  set_target_properties(ads PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
