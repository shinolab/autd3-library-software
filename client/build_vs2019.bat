@echo off
cd /d %~dp0

set BUILD_DIR=%~dp0
set PROJECT_DIR=%BUILD_DIR%build\
set IDE_NAME="Visual Studio 16 2019"
set USE_UNITY=ON

if "%1" == "-nounity" (
  set USE_UNITY=OFF
)

SETLOCAL ENABLEDELAYEDEXPANSION
for /F "tokens=1,2 delims=#" %%a in ('"prompt #$H#$E# & echo on & for %%b in (1) do rem"') do (
    set "DEL=%%a" 
)

chcp 65001 > nul

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
cmake -G %IDE_NAME% -A "x64" -D USE_UNITY=%USE_UNITY% %BUILD_DIR%
popd

if not "%1" == "-nounity" (
  call :colorEcho 0a "INFO"
  echo : Adding unity project...
  @copy /Y .\csharp\AUTD3Sharp.cs autdunity\Assets\AUTD\Scripts\AUTD3Sharp.cs > nul
  @copy /Y .\csharp\NativeMethods.cs autdunity\Assets\AUTD\Scripts\NativeMethods.cs > nul
  @copy /Y .\csharp\Util\Matrix3x3f.cs autdunity\Assets\AUTD\Scripts\Util\Matrix3x3f.cs > nul
  @copy /Y .\csharp\Util\GainMap.cs autdunity\Assets\AUTD\Scripts\Util\GainMap.cs > nul
  @xcopy .\autdunity %PROJECT_DIR%\autdunity /s/i > nul
)

call :colorEcho 0a "INFO"
echo : Setting PlatformTarget to x64...

call SetTargetx64.bat %PROJECT_DIR%csharp\ AUTD3Sharp.csproj tmp.csproj x86 x64
call SetTargetx64.bat %PROJECT_DIR%csharp_example\ AUTD3SharpTest.csproj tmp.csproj x86 x64

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