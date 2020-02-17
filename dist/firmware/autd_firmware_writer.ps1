﻿#
# File: build.ps1
# Project: firmware
# Created Date: 14/02/2020
# Author: Shun Suzuki
# -----
# Last Modified: 14/02/2020
# Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
# -----
# Copyright (c) 2020 Hapis Lab. All rights reserved.
# 
#
function ColorEcho($color, $PREFIX, $message) {
    Write-Host $PREFIX -ForegroundColor $color -NoNewline
    Write-Host ":", $message
}

function FindVivado() {
    $xilinx_reg = Get-ChildItem HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall | ForEach-Object { Get-ItemProperty $_.PsPath } | Where-Object DisplayName -match ^Xilinx.* | Select-Object -first 1
    if ($xilinx_reg) {
        return $xilinx_reg.InstallLocation
    }
    else {
        return "NULL"
    }
}

function FindJLink() {
    $jlink_reg = Get-ChildItem HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall | ForEach-Object { Get-ItemProperty $_.PsPath } | Where-Object DisplayName -match ^J-Link.*
    if ($jlink_reg) {
        return $jlink_reg.InstallLocation
    }
    else {
        return "NULL"
    }
}

function UpdateCPU([string]$cpuFirmwareFile) {
    if (-not (Get-Command jlink -ea SilentlyContinue)) {
        ColorEcho "Green" "INFO" "J-Link is not found in PATH. Looking for J-Link..."
        $jlink_path = FindJLink
        if ($jlink_path -eq "NULL") {
            ColorEcho "Red" "Error" "J-Link is not found. Install J-Link or add J-Link install folder to PATH."
            $host.UI.RawUI.ReadKey() | Out-Null
            exit
        }
        else {
            $env:Path = $env:Path + ";" + $jlink_path
        }
    }
    ColorEcho "Green" "INFO" "Find J-Link"

    $firm_path = Join-Path "firmwares" $cpuFirmwareFile
    Copy-Item -Path $firm_path -Destination "tmp.bin" -Force
    $command = "jlink -device R7S910018_R4F -if JTAG -speed 4000 -jtagconf -1,-1 -autoconnect 1 -CommanderScript ./scripts/cpu_flash.jlink"
    $success = $TRUE
    Invoke-Expression $command | Out-String -Stream | ForEach-Object {
        Write-Host $_
        if ($_.StartsWith("Cannot connect")) {
            ColorEcho "Red" "ERROR" "Cannnot connect to AUTD. Make sure that AUTD is connected and power on."
            $success = $FALSE
        }
    }
    Remove-Item -Path "tmp.bin"
    if ($success) {
        ColorEcho "Green" "INFO" "Update CPU done."
    }
}

function UpdateFPGA([string]$fpgaFirmwareFile) {
    if (-not (Get-Command vivado -ea SilentlyContinue)) {
        ColorEcho "Green" "INFO" "Vivado is not found in PATH. Looking for Vivado..."
        $xilinx_path = FindVivado
        if (($xilinx_path -eq "NULL")) {
            ColorEcho "Red" "Error" "Vivado is not found. Install Vivado."
            $host.UI.RawUI.ReadKey() | Out-Null
            exit
        }
        
        $vivado_path = Join-Path $xilinx_path "Vivado"
        if (-not (Test-Path $vivado_path)) {
            ColorEcho "Red" "Error" "Vivado is not found. Install Vivado."
            $host.UI.RawUI.ReadKey() | Out-Null
            exit
        }
        
        $vivados = Get-ChildItem $vivado_path
        if ($vivados.Length -eq 0) {
            ColorEcho "Red" "Error" "Vivado is not found. Install Vivado."
            $host.UI.RawUI.ReadKey() | Out-Null
            exit
        }

        $vivado_ver = $vivados | Select-Object -first 1
        ColorEcho "Green" "INFO" "Find Vivado", $vivado_ver
        $vivado_bin = Join-Path $vivado_path $vivado_ver | Join-Path -ChildPath "bin"
        $vivado_lib = Join-Path $vivado_path $vivado_ver | Join-Path -ChildPath "lib" | Join-Path -ChildPath "win64.o" 
        $env:Path = $env:Path + ";" + $vivado_bin + ";" + $vivado_lib
    }

    $firm_path = Join-Path "firmwares" $fpgaFirmwareFile
    Copy-Item -Path $firm_path -Destination "./scripts/tmp.mcs" -Force
    ColorEcho "Green" "INFO" "Invoking Vivado..."
    $command = "vivado -mode batch -nojournal -nolog -notrace -source ./scripts/fpga_configuration_script.tcl"
    $success = $TRUE
    Invoke-Expression $command | Out-String -Stream | ForEach-Object {
        Write-Host $_
        if ($_.StartsWith("ERROR")) {
            ColorEcho "Red" "ERROR" "Cannnot connect to AUTD. Make sure that AUTD is connected and power on."
            $success = $FALSE
        }
    }
    Remove-Item -Path "./scripts/tmp.mcs"
    if ($success) {
        ColorEcho "Green" "INFO" "Update FPGA done."
    }
}

Write-Host "AUTD3 Firmware Writer"
ColorEcho "Green" "INFO" "Make sure that you connected configuration cabels and AUTD's power is on."
ColorEcho "Green" "INFO" "Found firmwares are..."

$firmwares = Get-ChildItem firmwares
$fpga_firmware = ""
$cpu_firmware = ""
foreach ($firmware in $firmwares) {
    $ext = $firmware.Name.Split('.') | Select-Object -last 1
    if ($ext -eq "bin") {
        $cpu_firmware = $firmware
        ColorEcho "Blue" "CPU " $cpu_firmware
    }
    if ($ext -eq "mcs") {
        $fpga_firmware = $firmware
        ColorEcho "Blue" "FPGA" $fpga_firmware
    }
}

ColorEcho "Green" "INFO" "Select which firmware to be updated."
Write-Host "[0]: Both"
Write-Host "[1]: FPGA"
Write-Host "[2]: CPU"
do {
    try {
        $is_num = $true
        [int]$select = Read-host "Select"
    }
    catch { $is_num = $false }
} 
until (($select -ge 0 -and $select -le 2) -and $is_num)

if ($select -eq 0) {
    UpdateCPU $cpu_firmware
    UpdateFPGA $fpga_firmware
}
if ($select -eq 1) {
    UpdateFPGA $fpga_firmware
}
if ($select -eq 2) {
    UpdateCPU $cpu_firmware
}
ColorEcho "Green" "INFO" "Press any key to exit..."
$host.UI.RawUI.ReadKey() | Out-Null
exit
