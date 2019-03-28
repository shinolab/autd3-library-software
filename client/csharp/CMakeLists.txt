cmake_minimum_required(VERSION 3.8)
enable_language(CSharp)
include(CSharpUtilities)

set(CMAKE_CSharp_FLAGS "/langversion:7.3")

add_library(AUTD3Sharp 
  SHARED
  AUTD3Sharp.cs
  NativeMethods.cs
  GlobalSuppressions.cs
  CustomDictionary.xaml
  Util/GainMap.cs
  Util/Matrix3x3f.cs
  Util/Vector3f.cs
  Util/Quaternionf.cs
)

csharp_set_xaml_cs_properties(CustomDictionary.xaml)

set_property(SOURCE "CustomDictionary.xaml" 
  PROPERTY
  VS_XAML_TYPE "CodeAnalysisDictionary"
)

set_property(SOURCE "CustomDictionary.xaml" 
  PROPERTY
  VS_CSHARP_SubType "Designer"
)

target_compile_options(AUTD3Sharp PUBLIC "/unsafe")

if("${CMAKE_GENERATOR}" MATCHES "(Win64|IA64)")
  set_target_properties(AUTD3Sharp
    PROPERTIES
    VS_GLOBAL_PlatformTarget "x64"
  )
else()
  set_target_properties(AUTD3Sharp
    PROPERTIES
    VS_GLOBAL_PlatformTarget "x86"
  )
endif()

set_target_properties(AUTD3Sharp
  PROPERTIES
  VS_DOTNET_TARGET_FRAMEWORK_VERSION "v4.7.2"
  VS_GLOBAL_ROOTNAMESPACE "AUTD3Sharp"
)

target_link_libraries(AUTD3Sharp PRIVATE AUTD3SharpDll)