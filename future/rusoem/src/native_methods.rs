/*
 * File: native_methods.rs
 * Project: rusoem
 * Created Date: 30/08/2019
 * Author: Shun Suzuki
 * -----
 * Last Modified: 06/02/2020
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2019 Hapis Lab. All rights reserved.
 *
 */

use libc::{c_char, c_void};

const EC_MAXNAME: usize = 40;
const EC_MAXSLAVE: usize = 200;
const EC_MAXSM: usize = 8;
const EC_MAXFMMU: usize = 4;
const EC_MAXLEN_ADAPTERNAME: usize = 128;

pub const EC_STATE_NONE: u16 = 0x00;
pub const EC_STATE_INIT: u16 = 0x01;
pub const EC_STATE_PRE_OP: u16 = 0x02;
pub const EC_STATE_BOOT: u16 = 0x03;
pub const EC_STATE_SAFE_OP: u16 = 0x04;
pub const EC_STATE_OPERATIONAL: u16 = 0x08;
pub const EC_STATE_ACK: u16 = 0x10;
pub const EC_STATE_ERROR: u16 = 0x10;

#[allow(non_snake_case)]
#[repr(C, packed)]
struct ec_fmmut {
    LogStart: u32,
    LogLength: u16,
    LogStartbit: u8,
    LogEndbit: u8,
    PhysStart: u16,
    PhysStartBit: u8,
    FMMUtype: u8,
    FMMUactive: u8,
    unused1: u8,
    unused2: u16,
}

#[allow(non_snake_case)]
#[repr(C, packed)]
struct ec_smt {
    StartAddr: u16,
    SMlength: u16,
    SMflags: u32,
}

#[allow(non_snake_case)]
#[repr(C)]
pub struct ec_slavet {
    pub state: u16,
    ALstatuscode: u16,
    configadr: u16,
    aliasadr: u16,
    eep_man: u32,
    eep_id: u32,
    eep_rev: u32,
    Itype: u16,
    Dtype: u16,
    Obits: u16,
    Obytes: u32,
    outputs: *const u8,
    Ostartbit: u8,
    Ibits: u16,
    Ibytes: u32,
    inputs: *const u8,
    Istartbit: u8,
    SM: [ec_smt; EC_MAXSM],
    SMtype: [u8; EC_MAXSM],
    FMMU: [ec_fmmut; EC_MAXFMMU],
    FMMU0func: u8,
    FMMU1func: u8,
    FMMU2func: u8,
    FMMU3func: u8,
    mbx_l: u16,
    mbx_wo: u16,
    mbx_rl: u16,
    mbx_ro: u16,
    mbx_proto: u16,
    mbx_cnt: u8,
    hasdc: bool,
    ptype: u8,
    topology: u8,
    activeports: u8,
    consumedports: u8,
    parent: u16,
    parentport: u8,
    entryport: u8,
    DCrtA: i32,
    DCrtB: i32,
    DCrtC: i32,
    DCrtD: i32,
    pdelay: i32,
    DCnext: u16,
    DCprevious: u16,
    DCcycle: i32,
    DCshift: i32,
    DCactive: u8,
    configindex: u16,
    SIIindex: u16,
    eep_8byte: u8,
    eep_pdi: u8,
    CoEdetails: u8,
    FoEdetails: u8,
    EoEdetails: u8,
    SoEdetails: u8,
    Ebuscurrent: i16,
    blockLRW: u8,
    group: u8,
    FMMUunused: u8,
    islost: bool,
    config_function: i32,
    config_function_ecx: i32,
    name: [c_char; EC_MAXNAME + 1],
}

#[repr(C)]
pub struct ec_adapter {
    pub name: [c_char; EC_MAXLEN_ADAPTERNAME],
    pub desc: [c_char; EC_MAXLEN_ADAPTERNAME],
    pub next: *const ec_adapter,
}

#[link(name = "soem", kind = "static")]
extern "C" {
    pub static ec_DCtime: i64;
    pub fn ec_init(finame: *const c_char) -> i32;
    pub fn ec_config(usetable: u8, p_IOmap: *const c_void) -> i32;
    pub fn ec_configdc() -> bool;
    pub fn ec_statecheck(slave: u16, reqstate: u16, timeout: i32) -> u16;
    pub static mut ec_slave: [ec_slavet; EC_MAXSLAVE];
    pub fn ec_send_processdata() -> i32;
    pub fn ec_receive_processdata(timeout: i32) -> i32;
    pub fn ec_writestate(slave: u16) -> i32;
    pub fn ec_close();
    pub fn ec_dcsync0(slave: u16, act: bool, CyclTime: u32, CyclShift: i32);
    pub fn ec_find_adapters() -> *const ec_adapter;
}
