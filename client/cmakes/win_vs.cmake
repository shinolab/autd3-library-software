set(CMAKE_CONFIGURATION_TYPES "Debug;Release" )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj /wd4819 /GR-")
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
if(BUILD_WITH_STATIC_CRT)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MT")
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MD")
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()
