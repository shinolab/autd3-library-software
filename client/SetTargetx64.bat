: File: SetTargetx64.bat
: Project: client
: Created Date: 25/08/2019
: Author: Shun Suzuki
: -----
: Last Modified: 04/09/2019
: Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
: -----
: Copyright (c) 2019 Hapis Lab. All rights reserved.
: 

@echo off

set dir=%1
set input=%dir%%2
set output=%dir%%3
set from=%4
set to=%5

setlocal enabledelayedexpansion
for /f "tokens=1* delims=: eol=" %%a in ('findstr /n "^" %input%') do (
  set line=%%b
  if not "!line!" == "" (
    set line=!line:%from%=%to%!
  )
  echo.!line!>> %output%
)
endlocal

del /Q %input%
ren %output% %2

exit /B