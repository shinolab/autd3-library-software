cmake_minimum_required(VERSION 3.1)


include_directories(${PROJECT_SOURCE_DIR}/include ${EIGEN_PATH} examples)

file(GLOB hpp_files examples/*.hpp runner.hpp)

link_libraries(autd3)
if(BUILD_HOLO_GAIN)
  link_libraries(gain_holo)
  add_definitions(-DBUILD_HOLO_GAIN)
endif()
if(BUILD_FROM_FILE_MODULATION)
  link_libraries(modulation_from_file)
  add_definitions(-DBUILD_FROM_FILE_MOD)
  add_compile_definitions(AUTD3_RESOURCE_PATH="${PROJECT_SOURCE_DIR}/examples/resources")
endif()

if(WIN32)
  link_libraries(ws2_32.lib)
  link_libraries(winmm.lib)
elseif(APPLE)
  link_libraries(pcap)
else()
  link_libraries(rt)
endif()
  
if(BUILD_SOEM_LINK)
  add_executable(example_soem soem.cpp ${hpp_files})
  target_link_libraries(example_soem link_soem)
endif()

if(WIN32)
  if(BUILD_TWINCAT_LINK)
    add_executable(example_twincat twincat.cpp ${hpp_files})
    target_link_libraries(example_twincat link_twincat)
  endif()  
endif()

if(BUILD_EMULATOR_LINK)
  add_executable(example_emulator emulator.cpp ${hpp_files})
  target_link_libraries(example_emulator link_emulator)
endif()
