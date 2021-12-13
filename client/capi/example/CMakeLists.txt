cmake_minimum_required(VERSION 3.16)

# currently Windows only
if(WIN32)
  if(BUILD_SOEM_LINK)
    add_executable(c_example_soem soem.c)
    target_include_directories(c_example_soem PRIVATE ${PROJECT_SOURCE_DIR}/capi/base ${PROJECT_SOURCE_DIR}/capi/soem_link)
    target_link_libraries(c_example_soem autd3capi autd3capi-soem-link) 
    target_link_libraries(c_example_soem winmm ws2_32)
    set_target_properties(c_example_soem PROPERTIES FOLDER "capi/example")
    add_custom_command(TARGET c_example_soem POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy
      ${CMAKE_BINARY_DIR}/bin/$<CONFIG>/autd3capi.dll
      ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>)
    add_custom_command(TARGET c_example_soem POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy
      ${CMAKE_BINARY_DIR}/bin/$<CONFIG>/autd3capi-soem-link.dll
      ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>)
  endif()
endif()
