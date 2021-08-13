enable_testing()

include_directories(. ${PROJECT_SOURCE_DIR}/include ${EIGEN_PATH})
add_executable(test_autd3)
target_link_libraries(test_autd3 PRIVATE autd3 gtest_main)

if(BUILD_HOLO_GAIN)
  add_subdirectory(${PROJECT_SOURCE_DIR}/tests/gain/holo)
  target_link_libraries(test_autd3 PRIVATE gain_holo)
endif()

target_sources(test_autd3 PRIVATE test.cpp)
add_test(NAME test_autd3 COMMAND test_autd3)

include(GoogleTest)
gtest_discover_tests(test_autd3)
