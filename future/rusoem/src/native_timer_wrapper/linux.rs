/*
 * File: native_timer_wrapper.rs
 * Project: rusoem
 * Created Date: 15/12/2019
 * Author: Shun Suzuki
 * -----
 * Last Modified: 16/12/2019
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2019 Hapis Lab. All rights reserved.
 *
 */

use std::mem;
use std::ptr;

use libc::{
    c_int, c_void, clockid_t, itimerspec, sigaction, sigevent, siginfo_t, timespec, CLOCK_REALTIME,
};

#[allow(non_camel_case_types)]
type timer_t = usize;

extern "C" {
    fn timer_create(clockid_t: clockid_t, sevp: *mut sigevent, timerid: *mut timer_t) -> c_int;
    fn timer_settime(
        timerid: timer_t,
        flags: c_int,
        new_value: *const itimerspec,
        old_value: *mut itimerspec,
    ) -> c_int;
    fn timer_delete(timerid: timer_t) -> c_int;
}

const SIGNUM: c_int = 40;
type WAITORTIMERCALLBACKN = unsafe extern "C" fn(c_int, *mut siginfo_t, *mut c_void);
type WAITORTIMERCALLBACK = Option<WAITORTIMERCALLBACKN>;

struct TimerHandle {
    timer: timer_t,
}

unsafe impl Send for TimerHandle {}

pub struct NativeTimerWrapper {
    timer_handle: Option<TimerHandle>,
}

impl NativeTimerWrapper {
    pub fn new() -> NativeTimerWrapper {
        NativeTimerWrapper { timer_handle: None }
    }

    pub fn start(&mut self, cb: WAITORTIMERCALLBACK, period_ns: u32) -> i32 {
        unsafe {
            if let Some(cb) = cb {
                let mut sa: sigaction = mem::zeroed();
                sa.sa_flags = libc::SA_SIGINFO;
                sa.sa_sigaction = cb as usize;
                libc::sigemptyset(&mut sa.sa_mask);
                let res = sigaction(SIGNUM, &sa, ptr::null_mut());
                if res < 0 {
                    return res;
                }
            }

            let mut sev: sigevent = mem::zeroed();
            sev.sigev_value = libc::sigval {
                sival_ptr: ptr::null_mut(),
            };
            sev.sigev_signo = SIGNUM;

            sev.sigev_notify = libc::SIGEV_THREAD_ID;
            let tid = libc::syscall(libc::SYS_gettid);
            sev.sigev_notify_thread_id = tid as i32;

            let mut timer = 0;
            let res = timer_create(CLOCK_REALTIME, &mut sev, &mut timer);
            if res < 0 {
                return res;
            }

            let start = timespec {
                tv_sec: 0,
                tv_nsec: period_ns as i64,
            };
            let repeat = timespec {
                tv_sec: 0,
                tv_nsec: period_ns as i64,
            };

            let new_value = itimerspec {
                it_interval: repeat,
                it_value: start,
            };

            let res = timer_settime(timer, 0, &new_value, ptr::null_mut());

            if res < 0 {
                return res;
            }

            self.timer_handle = Some(TimerHandle { timer });
            1
        }
    }

    pub fn close(&mut self) -> i32 {
        if let Some(handle) = self.timer_handle.take() {
            unsafe {
                let res = timer_delete(handle.timer);
                if res < 0 {
                    res
                } else {
                    1
                }
            }
        } else {
            1
        }
    }
}

impl Drop for NativeTimerWrapper {
    fn drop(&mut self) {
        let _ = self.close();
    }
}
