cmake_minimum_required(VERSION 3.1)

set (CMAKE_CXX_STANDARD 11)

execute_process(COMMAND git submodule update)
message("-- Updating git submodule done")

include_directories(./ADS/AdsLib)

set (CMAKE_C_FLAGS "-pedantic -Wall")

add_library(ads STATIC
  ADS/AdsLib/AdsDef.cpp
  ADS/AdsLib/AdsLib.cpp
  ADS/AdsLib/AmsConnection.cpp
  ADS/AdsLib/AmsPort.cpp
  ADS/AdsLib/AmsRouter.cpp
  ADS/AdsLib/Log.cpp
  ADS/AdsLib/NotificationDispatcher.cpp
  ADS/AdsLib/Sockets.cpp
  ADS/AdsLib/Frame.cpp
)

if(APPLE)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -lc++")
elseif(WIN32)
  target_link_libraries(ads ws2_32)
endif()

set (Ads_LIBRARIES ads PARENT_SCOPE)
set (Ads_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ADS/AdsLib/ PARENT_SCOPE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4267")

#INSTALL(TARGETS ads LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)