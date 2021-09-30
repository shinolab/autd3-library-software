cmake_minimum_required(VERSION 3.16)

find_package(Threads REQUIRED)

file(GLOB_RECURSE SOEM_SRC_FILES ${SOEM_PATH}/soem/*.c ${SOEM_PATH}/osal/${OS}/*.c ${SOEM_PATH}/oshw/${OS}/*.c)

add_library(soem_link
  autdsoem.cpp
  autdsoem.hpp
  soem.cpp
  ${SOEM_SRC_FILES}
  ${PROJECT_SOURCE_DIR}/include/autd3/link/soem.hpp
)
target_include_directories(soem_link PUBLIC ${PROJECT_SOURCE_DIR}/include)

target_include_directories(soem_link PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${SOEM_PATH}/soem
  ${SOEM_PATH}/osal
  ${SOEM_PATH}/osal/${OS}
  ${SOEM_PATH}/oshw/${OS}
)
if(WIN32)
  target_include_directories(soem_link PRIVATE
    ${SOEM_PATH}/oshw/${OS}/wpcap/Include
    ${SOEM_PATH}/oshw/${OS}/wpcap/Include/pcap
  )
  target_compile_options(soem_link PRIVATE /D_CRT_SECURE_NO_WARNINGS /wd4996)
endif()

target_link_libraries(soem_link ${AUTD_LINK_LIBRARIES_KEYWORD} ${CMAKE_THREAD_LIBS_INIT})
if(WIN32)
  if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "Win32")
  target_link_directories(soem_link PUBLIC ${PROJECT_SOURCE_DIR}/lib/link/soem/Lib)
  else()
    target_link_directories(soem_link PUBLIC ${PROJECT_SOURCE_DIR}/lib/link/soem/Lib/x64)
  endif()
  target_link_libraries(soem_link ${AUTD_LINK_LIBRARIES_KEYWORD} ws2_32 winmm)
  target_link_libraries(soem_link PUBLIC Packet.lib wpcap.lib)
elseif(APPLE)
  target_link_libraries(soem_link ${AUTD_LINK_LIBRARIES_KEYWORD} pcap)
else()
  target_link_libraries(soem_link ${AUTD_LINK_LIBRARIES_KEYWORD} rt -static-libgcc -static-libstdc++)
endif()

set_target_properties(soem_link
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

if(WIN32)
  set_target_properties(soem_link PROPERTIES FOLDER "link")
else()
  set_target_properties(soem_link PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
