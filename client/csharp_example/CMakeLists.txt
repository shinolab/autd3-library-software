cmake_minimum_required(VERSION 3.8)
enable_language(CSharp)
set(CMAKE_CSharp_FLAGS "/langversion:7.3")

file(GLOB TestSources Test/*.cs)
add_executable(AUTD3SharpSample 
  program.cs
  ${TestSources}
)


if("${CMAKE_GENERATOR}" MATCHES "(Win64|IA64)")
  set_target_properties(AUTD3SharpSample
    PROPERTIES
    VS_GLOBAL_PlatformTarget "x64"
  )
else()
  set_target_properties(AUTD3SharpSample
    PROPERTIES
    VS_GLOBAL_PlatformTarget "x86"
  )
endif()

set_target_properties(AUTD3SharpSample
  PROPERTIES
  VS_DOTNET_TARGET_FRAMEWORK_VERSION "v4.7.2"
)

target_link_libraries(AUTD3SharpSample AUTD3Sharp)