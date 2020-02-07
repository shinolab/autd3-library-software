/*
 * File: soem_handler.rs
 * Project: rusoem
 * Created Date: 30/08/2019
 * Author: Shun Suzuki
 * -----
 * Last Modified: 07/02/2020
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2019 Hapis Lab. All rights reserved.
 *
 */

#[cfg(target_os = "linux")]
use libc::{c_int, siginfo_t};
#[cfg(windows)]
use winapi::um::winnt::PVOID;

use crate::native_methods::*;
use crate::native_timer_wrapper::NativeTimerWrapper;
use crate::SOEMError;

use libc::{c_char, c_void};

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::thread::{self, JoinHandle};
use std::vec::Vec;

const EC_TIMEOUTSTATE: i32 = 2_000_000;
const EC_TIMEOUTRET: i32 = 2_000;

static SEND_COND: AtomicBool = AtomicBool::new(false);
static RTTHREAD_LOCK: AtomicBool = AtomicBool::new(false);

macro_rules! if_not_open_or_cannot_read {
    ($is_open:expr, $cnt:stmt) => {
        if let Ok(open) = $is_open.read() {
            if !*open {
                $cnt
            }
        }
    };
}

macro_rules! write_rwlock {
    ($x_rwlock:expr, $value: expr) => {
        if let Ok(mut x) = $x_rwlock.write() {
            *x = $value;
        }
    };
}

pub struct ECConfig {
    pub header_size: usize,
    pub body_size: usize,
    pub input_size: usize,
}

impl ECConfig {
    pub fn new(header_size: usize, body_size: usize, input_size: usize) -> Self {
        ECConfig {
            header_size,
            body_size,
            input_size,
        }
    }

    pub fn size(&self) -> usize {
        self.header_size + self.body_size + self.input_size
    }
}

pub struct RuSOEM {
    timer_handle: NativeTimerWrapper,
    is_open: Arc<RwLock<bool>>,
    ifname: std::ffi::CString,
    dev_num: u16,
    config: ECConfig,
    io_map: Arc<RwLock<Vec<u8>>>,
    cpy_handle: Option<JoinHandle<()>>,
    send_buf_q: Arc<(Mutex<VecDeque<Vec<u8>>>, Condvar)>,
    sync0_cyctime: u32,
}

#[allow(clippy::mutex_atomic)]
impl RuSOEM {
    pub fn new(ifname: &str, config: ECConfig) -> RuSOEM {
        RuSOEM {
            timer_handle: NativeTimerWrapper::new(),
            is_open: Arc::new(RwLock::new(false)),
            config,
            dev_num: 0,
            ifname: std::ffi::CString::new(ifname.to_string()).unwrap(),
            io_map: Arc::new(RwLock::new(vec![])),
            cpy_handle: None,
            send_buf_q: Arc::new((Mutex::new(VecDeque::new()), Condvar::new())),
            sync0_cyctime: 0,
        }
    }

    pub fn start(
        &mut self,
        dev_num: u16,
        ec_sm2_cyctime_ns: u32,
        ec_sync0_cyctime_ns: u32,
    ) -> Result<(), failure::Error> {
        self.dev_num = dev_num;
        let size = self.config.size() * dev_num as usize;
        self.sync0_cyctime = ec_sync0_cyctime_ns;

        unsafe {
            if ec_init(self.ifname.as_ptr() as *const c_char) != 1 {
                return Err(SOEMError::NoSocketConnection {
                    ifname: self.ifname.to_str().unwrap().to_string(),
                }
                .into());
            }

            self.io_map = Arc::new(RwLock::new(vec![0x00; size]));
            if let Ok(io_map) = self.io_map.read() {
                let wc = ec_config(0, io_map.as_ptr() as *const c_void) as u16;
                if wc != dev_num {
                    return Err(SOEMError::SlaveNotFound { wc, num: dev_num }.into());
                }
            };

            ec_configdc();
            ec_statecheck(0, EC_STATE_SAFE_OP, EC_TIMEOUTSTATE * 4);

            ec_slave[0].state = EC_STATE_OPERATIONAL;
            ec_send_processdata();
            ec_receive_processdata(EC_TIMEOUTRET);

            ec_writestate(0);

            let mut chk = 200;
            ec_statecheck(0, EC_STATE_OPERATIONAL, 50000);
            while chk > 0 && (ec_slave[0].state != EC_STATE_OPERATIONAL) {
                ec_statecheck(0, EC_STATE_OPERATIONAL, 50000);
                chk -= 1;
            }

            if ec_slave[0].state != EC_STATE_OPERATIONAL {
                return Err(SOEMError::NotResponding.into());
            }

            write_rwlock!(self.is_open, true);
            RuSOEM::setup_sync0(true, dev_num, self.sync0_cyctime);

            if self
                .timer_handle
                .start(Some(Self::rt_thread), ec_sm2_cyctime_ns)
                != 1
            {
                return Err(SOEMError::CreateTimerError.into());
            }

            self.create_cpy_thread();
        }
        Ok(())
    }

    pub fn close(&mut self) -> bool {
        if_not_open_or_cannot_read!(self.is_open, return true);

        write_rwlock!(self.is_open, false);
        let (send_lk, send_cvar) = &*self.send_buf_q;
        {
            let mut deq = send_lk.lock().unwrap();
            deq.clear();
        }
        send_cvar.notify_one();

        if let Some(jh) = self.cpy_handle.take() {
            jh.join().unwrap();
            self.cpy_handle = None;
        }

        if let Ok(mut io_map) = self.io_map.write() {
            let output_frame_size = self.config.header_size + self.config.body_size;
            unsafe {
                std::ptr::write_bytes(
                    io_map.as_mut_ptr(),
                    0x00,
                    self.dev_num as usize * output_frame_size,
                );
            }
        }
        SEND_COND.store(false, Ordering::Release);
        while !SEND_COND.load(Ordering::Acquire) {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        self.timer_handle.close();

        let mut clear = true;
        for _ in 0..200 {
            unsafe {
                RuSOEM::rt_thread_impl();
            }
            std::thread::sleep(std::time::Duration::from_millis(1));

            let r = self.read::<u16>();
            if let Some(r) = r {
                for v in r {
                    if v != 0 {
                        clear = false;
                        break;
                    } else {
                        clear = true;
                    }
                }

                if clear {
                    break;
                }
            }
        }

        unsafe {
            RuSOEM::setup_sync0(false, self.dev_num, self.sync0_cyctime);

            ec_slave[0].state = EC_STATE_INIT;
            ec_writestate(0);
            ec_statecheck(0, EC_STATE_INIT, EC_TIMEOUTSTATE);
            ec_close();
        }

        clear
    }

    pub fn send(&self, data: Vec<u8>) {
        let (send_lk, send_cvar) = &*self.send_buf_q;
        {
            let mut deq = send_lk.lock().unwrap();
            deq.push_back(data);
        }
        send_cvar.notify_one();
    }

    pub fn is_open(&self) -> bool {
        match self.is_open.read() {
            Ok(is_open) => *is_open,
            _ => false,
        }
    }

    unsafe fn write_io_map(
        src: Vec<u8>,
        dst: *mut u8,
        dev_num: u16,
        header_size: usize,
        body_size: usize,
    ) {
        let size = src.len();
        let includes_body = (size - header_size) > 0;
        for i in 0..(dev_num as usize) {
            if includes_body {
                std::ptr::copy_nonoverlapping(
                    src.as_ptr().add(header_size + body_size * i),
                    dst.add((header_size + body_size) * i),
                    body_size,
                );
            }
            std::ptr::copy_nonoverlapping(
                src.as_ptr(),
                dst.add((header_size + body_size) * i + body_size),
                header_size,
            );
        }
    }

    pub fn read<T>(&self) -> Option<Vec<T>> {
        if let Ok(io_map) = self.io_map.read() {
            let size = io_map.len();
            let output_frame_size = self.config.header_size + self.config.body_size;
            let dev_num = size / self.config.size();
            let element_size = std::mem::size_of::<T>();
            let len = dev_num * self.config.input_size / element_size;
            let mut v = Vec::<T>::with_capacity(len);
            unsafe {
                v.set_len(len);
                std::ptr::copy_nonoverlapping(
                    io_map.as_ptr().add(output_frame_size * dev_num) as *const T,
                    v.as_mut_ptr(),
                    len,
                );
            }
            Some(v)
        } else {
            None
        }
    }

    unsafe fn setup_sync0(actiavte: bool, dev_num: u16, cycle_time: u32) {
        let exceed = cycle_time > 100_000_000u32;
        for slave in 1..=dev_num {
            if exceed {
                ec_dcsync0(slave, actiavte, cycle_time, 0);
            } else {
                let shift = (dev_num - slave) as i32 * cycle_time as i32;
                ec_dcsync0(slave, actiavte, cycle_time, shift);
            }
        }
    }

    unsafe fn create_cpy_thread(&mut self) {
        let is_open = self.is_open.clone();
        let send_buf_q = self.send_buf_q.clone();
        let dev_num = self.dev_num;
        let io_map = self.io_map.clone();
        let header_size = self.config.header_size;
        let body_size = self.config.body_size;
        self.cpy_handle = Some(thread::spawn(move || {
            let (send_lk, send_cvar) = &*send_buf_q;
            let mut send_buf = send_lk.lock().unwrap();
            loop {
                if_not_open_or_cannot_read!(is_open, break);
                match send_buf.pop_front() {
                    None => send_buf = send_cvar.wait(send_buf).unwrap(),
                    Some(buf) => {
                        if let Ok(mut io_map) = io_map.write() {
                            RuSOEM::write_io_map(
                                buf,
                                io_map.as_mut_ptr(),
                                dev_num,
                                header_size,
                                body_size,
                            );
                        }
                        {
                            SEND_COND.store(false, Ordering::Release);
                            while !SEND_COND.load(Ordering::Acquire) {}
                        }
                    }
                }
            }
        }));
    }

    #[inline]
    unsafe fn rt_thread_impl() {
        if !RTTHREAD_LOCK.compare_and_swap(false, true, Ordering::SeqCst) {
            let pre = SEND_COND.load(Ordering::Acquire);
            ec_send_processdata();
            if !pre {
                SEND_COND.store(true, Ordering::Release);
            }
            ec_receive_processdata(EC_TIMEOUTRET);
            RTTHREAD_LOCK.store(false, Ordering::Release);
        }
    }

    #[cfg(windows)]
    unsafe extern "system" fn rt_thread(_lp_param: PVOID, _t: u8) {
        RuSOEM::rt_thread_impl();
    }

    #[cfg(target_os = "linux")]
    unsafe extern "C" fn rt_thread(_sig: c_int, _si: *mut siginfo_t, _uc: *mut c_void) {
        RuSOEM::rt_thread_impl();
    }

    #[cfg(target_os = "macos")]
    unsafe extern "C" fn rt_thread(_ptr: *mut c_void) {
        RuSOEM::rt_thread_impl();
    }
}
