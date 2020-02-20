: File: build.bat
: Project: client
: Created Date: 25/08/2019
: Author: Shun Suzuki
: -----
: Last Modified: 20/02/2020
: Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
: -----
: Copyright (c) 2019-2020 Hapis Lab. All rights reserved.
: 

@echo off
@setlocal enabledelayedexpansion
cd /d %~dp0

for /F "tokens=1,2 delims=#" %%a in ('"prompt #$H#$E# & echo on & for %%b in (1) do rem"') do (
    set "DEL=%%a" 
)

chcp 65001 > nul

set BUILD_DIR=%~dp0
set PROJECT_DIR=%BUILD_DIR%build\

set VS_VERSION=2019
set ARCH="x64"
set USE_UNITY=ON
set ENABLE_TEST=OFF
set TOOL_CHAIN=""

for %%f in (%*) do (
  if %%f == -vs2017 (
    set VS_VERSION=2017
  )
  if %%f == -x86 (
    set ARCH="Win32"
  )
  if %%f == -nounity (
    set USE_UNITY=OFF
    call :colorEcho 0a "INFO"
    echo : USE_UNITY=OFF
  )
  if %%f == -test (
    set ENABLE_TEST=ON
    call :colorEcho 0a "INFO"
    echo : ENABLE_TEST=ON
  )
)

if %ENABLE_TEST% == ON (
  goto SET_GTEST
) else (
  goto SET_IDE
)

:SET_GTEST

for %%f in (%*) do (
    set tempval=%%f
    set temppre=!tempval:~1,22!
    if !temppre! == -DCMAKE_TOOLCHAIN_FILE (
      set TOOL_CHAIN=%%f
    )
)

if %TOOL_CHAIN% == "" (
  call :colorEcho 0c "ERROR"
  echo : Please specify vcpkg tool chain file. ex. -test "-DCMAKE_TOOLCHAIN_FILE=C:[...]\vcpkg\scripts\buildsystems\vcpkg.cmake"
  exit /B
)

:SET_IDE

set IDE_NAME="Visual Studio
if %VS_VERSION% equ 2017 if %ARCH% == "Win32" (
  set IDE_NAME=%IDE_NAME% 15 2017"
)
if %VS_VERSION% equ 2017 if %ARCH% == "x64" (
  set IDE_NAME=%IDE_NAME% 15 2017 Win64"
)
if %VS_VERSION% equ 2019 (
  set IDE_NAME=%IDE_NAME% 16 2019"
)

call :colorEcho 0a "INFO"
echo : Use %IDE_NAME% with %ARCH% architecture. 

cmake -version > nul 2>&1
if not %errorlevel% == 0 (
  call :colorEcho 0a "INFO"
  echo : CMake not found in PATH. Looking for CMake... 
  call :setCMake
)

cmake -version > nul 2>&1
if not %errorlevel% == 0 (
  call :colorEcho 0c "ERROR"
  echo : CMake not found. Install CMake or set CMake binary folder to PATH.
  exit /B
)

call :colorEcho 0a "INFO"
SET /P<NUL=: Find 
for /F "tokens=1 delims=" %%i in ('cmake -version') do (
  set CMAKE_VERSION=%%i 
  goto :tmp
)
:tmp
echo %CMAKE_VERSION%

call :colorEcho 0a "INFO"
echo : Create project directory if not exists.
if exist "%PROJECT_DIR%" (
    call :colorEcho 0c "WARNING"
    echo : Directory %PROJECT_DIR% is already exists.

    call :colorEcho 0e "CFM"
    set /P ans=": Overwrite (Y/[N])?"

    if "!ans!" == "y" (
         goto :yes
    ) else if "!ans!" == "Y" (
         goto :yes
    )
    echo Canceled...
    @pause
    exit /B

    :yes
    rmdir /s /q %PROJECT_DIR%
    if exist "%PROJECT_DIR%" (
        call :colorEcho 0c "WARNING"
        echo : Cannot remove directory %PROJECT_DIR%
        @pause
        exit /B
    )
    call :colorEcho 0a "INFO"
    echo : Remove directory %PROJECT_DIR%
    rem
)

mkdir %PROJECT_DIR%
call :colorEcho 0a "INFO"
echo : Success to create project directory "%PROJECT_DIR%"

call :colorEcho 0a "INFO"
echo : Creating VS solution...
pushd %PROJECT_DIR%

if %ENABLE_TEST% == ON (
  goto USE_GTEST
) else (
  goto NOT_USE_GTEST
)

:USE_GTEST
if %VS_VERSION% equ 2017 (
  cmake .. -G %IDE_NAME% -D USE_UNITY=%USE_UNITY% -D ENABLE_TESTS=%ENABLE_TEST% %TOOL_CHAIN%
)
if %VS_VERSION% equ 2019 (
  cmake .. -G %IDE_NAME% -A %ARCH% -D USE_UNITY=%USE_UNITY% -D ENABLE_TESTS=%ENABLE_TEST% %TOOL_CHAIN%
)

:NOT_USE_GTEST
if %VS_VERSION% equ 2017 (
  cmake .. -G %IDE_NAME% -D USE_UNITY=%USE_UNITY% -D ENABLE_TESTS=%ENABLE_TEST%
)
if %VS_VERSION% equ 2019 (
  cmake .. -G %IDE_NAME% -A %ARCH% -D USE_UNITY=%USE_UNITY% -D ENABLE_TESTS=%ENABLE_TEST%
)

popd

if %USE_UNITY% == ON (
  call :colorEcho 0a "INFO"
  echo : Adding unity project...
  @copy /Y .\csharp\AUTD3Sharp.cs autdunity\Assets\AUTD\Scripts\AUTD3Sharp.cs > nul
  @copy /Y .\csharp\NativeMethods.cs autdunity\Assets\AUTD\Scripts\NativeMethods.cs > nul
  @copy /Y .\csharp\Util\Matrix3x3d.cs autdunity\Assets\AUTD\Scripts\Util\Matrix3x3d.cs > nul
  @copy /Y .\csharp\Util\GainMap.cs autdunity\Assets\AUTD\Scripts\Util\GainMap.cs > nul
  @xcopy .\autdunity %PROJECT_DIR%\autdunity /s/i > nul
)

if %VS_VERSION% equ 2019 if %ARCH%=="x64" (
  call :colorEcho 0a "INFO"
  echo : Setting PlatformTarget to x64...

  call SetTargetx64.bat %PROJECT_DIR%csharp\ AUTD3Sharp.csproj tmp.csproj x86 x64
  call SetTargetx64.bat %PROJECT_DIR%csharp_example\ AUTD3SharpSample.csproj tmp.csproj x86 x64
)

call :colorEcho 0a "INFO"
echo : Done.

pause
exit /B

:setCMake
for /f "tokens=2,*" %%I in ('reg query "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall" /s ^| find "InstallLocation"') do (
  set x=%%J
  set tail=!x:~-6!
  if "!tail!" == "CMake\" (	
     set path=!path!;!x!bin
     goto :findCMake
  )
)
 
for /f "tokens=2,*" %%I in ('reg query "HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Uninstall" /s ^| find "InstallLocation"') do (
  set x=%%J
  set tail=!x:~-6!
  if "!tail!" == "CMake\" (	
     set path=!path!;!x!bin
     goto :findCMake
  )
)
 
for /f "tokens=2,*" %%I in ('reg query "HKEY_LOCAL_MACHINE\SOFTWARE\Wow6432Node\Microsoft\Windows\CurrentVersion\Uninstall" /s ^| find "InstallLocation"') do (
  set x=%%J
  set tail=!x:~-6!
  if "!tail!" == "CMake\" (	
     set path=!path!;!x!bin
     goto :findCMake
  )
)

:findCMake
exit /B

:colorEcho 
echo off 
SET /P<NUL=%DEL% > "%~2" 
findstr /v /a:%1 /R "^$" "%~2" nul 
del "%~2" > nul 2>&1 
exit /B