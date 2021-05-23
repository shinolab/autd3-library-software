cmake_minimum_required(VERSION 3.10)


find_package(Threads REQUIRED)

include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${SOEM_PATH}/soem
  ${SOEM_PATH}/osal
  ${SOEM_PATH}/osal/${OS}
  ${SOEM_PATH}/oshw/${OS}
)

if(WIN32)
  include_directories(
    ${SOEM_PATH}/oshw/${OS}/wpcap/Include
    ${SOEM_PATH}/oshw/${OS}/wpcap/Include/pcap
  )
endif()

if(WIN32)
  include_directories(${PROJECT_SOURCE_DIR}/osal_timer/win32)
elseif(APPLE)
  include_directories(${PROJECT_SOURCE_DIR}/osal_timer/macosx)
else()
  include_directories(${PROJECT_SOURCE_DIR}/linux/win32)
endif()

if(WIN32)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /D _CRT_SECURE_NO_WARNINGS")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4996")
endif()

file(GLOB_RECURSE SOEM_SRC_FILES ${SOEM_PATH}/soem/*.c ${SOEM_PATH}/osal/${OS}/*.c ${SOEM_PATH}/oshw/${OS}/*.c)

add_library(soem_link STATIC
  autdsoem.cpp
  autdsoem.hpp
  soem_link.cpp
  ${SOEM_SRC_FILES}
  ${PROJECT_SOURCE_DIR}/include/soem_link.hpp
)

if(WIN32)
  if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "Win32")
    add_custom_command(TARGET soem_link
	  POST_BUILD
	  COMMAND lib /OUT:$(OUTDIR)temp.lib $(TARGETPATH) ${PROJECT_SOURCE_DIR}\\deps\\SOEM\\SOEM\\oshw\\win32\\wpcap\\Lib\\*.lib /ignore:4006
	  COMMAND del $(TARGETPATH)
	  COMMAND rename $(OUTDIR)temp.lib $(TargetFileName)
	)
  else()
	add_custom_command(TARGET soem_link
	  POST_BUILD
	  COMMAND lib /OUT:$(OUTDIR)temp.lib $(TARGETPATH) ${PROJECT_SOURCE_DIR}\\deps\\SOEM\\SOEM\\oshw\\win32\\wpcap\\Lib\\x64\\*.lib /ignore:4006
	  COMMAND del $(TARGETPATH)
	  COMMAND rename $(OUTDIR)temp.lib $(TargetFileName)
	)
  endif()

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd26451 /wd6385 /wd6386 /wd6011 /wd26495 /wd4996 /wd6001 /wd4200")
endif()

target_link_libraries(soem_link ${CMAKE_THREAD_LIBS_INIT})
if(WIN32)
  target_link_libraries(soem_link ws2_32.lib)
  target_link_libraries(soem_link winmm.lib)
elseif(APPLE)
  target_link_libraries(soem_link pcap)
else()
  target_link_libraries(soem_link rt)
  target_link_libraries(soem_link -static-libgcc -static-libstdc++)
endif()

set_target_properties(soem_link
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

if(NOT WIN32)
  set_target_properties(soem_link PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
