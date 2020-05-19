# 
# File: check.ps1
# Project: client
# Created Date: 14/05/2020
# Author: Shun Suzuki
# -----
# Last Modified: 14/05/2020
# Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
# -----
# Copyright (c) 2020 Hapis Lab. All rights reserved.
# 
# 

function ColorEcho($color, $PREFIX, $message) {
    Write-Host $PREFIX -ForegroundColor $color -NoNewline
    Write-Host ":", $message
}

function check_hyper_v_disabled() {
    foreach ($line in Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V) {
        $state = $line -split " " | Where-Object { $_ -match "^State" }
        if ($state -notlike "*Disabled*") {
            ColorEcho "Red" "Error" "Hyper-V is enabled. It will cause unexpected behavior."
            return
        }
        else {
            ColorEcho "Green" "Ok" "Hyper-V is disabled."
            return 
        }
    }
}

check_hyper_v_disabled
return
