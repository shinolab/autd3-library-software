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

if (WIN32)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4267")
endif()

#INSTALL(TARGETS ads LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)

if(UNIX AND NOT APPLE)
  add_library(ads_shared SHARED
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
  set_property(TARGET ads_shared PROPERTY POSITION_INDEPENDENT_CODE ON)
  set_target_properties(ads_shared PROPERTIES OUTPUT_NAME ads)
  set_target_properties(ads_shared
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )
endif()
