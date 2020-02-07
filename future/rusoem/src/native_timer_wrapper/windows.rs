/*
 * File: native_timer_wrapper.rs
 * Project: rusoem
 * Created Date: 12/12/2019
 * Author: Shun Suzuki
 * -----
 * Last Modified: 16/12/2019
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2019 Hapis Lab. All rights reserved.
 *
 */

use winapi::shared::winerror::ERROR_IO_PENDING;
use winapi::um::errhandlingapi;
use winapi::um::handleapi::INVALID_HANDLE_VALUE;
use winapi::um::threadpoollegacyapiset;
use winapi::um::winnt::{HANDLE, PHANDLE, WAITORTIMERCALLBACK};

struct TimerHandle {
    timer: HANDLE,
    timer_queue: HANDLE,
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
            let timer_queue = threadpoollegacyapiset::CreateTimerQueue();
            if timer_queue.is_null() {
                return 0;
            }

            let mut timer: HANDLE = std::ptr::null_mut();
            let res = threadpoollegacyapiset::CreateTimerQueueTimer(
                &mut timer as PHANDLE,
                timer_queue,
                cb,
                std::ptr::null_mut(),
                0,
                period_ns / 1000 / 1000,
                0,
            );
            if res != 0 {
                let timer_handle = TimerHandle { timer, timer_queue };
                self.timer_handle = Some(timer_handle);
            }
            res
        }
    }

    pub fn close(&mut self) -> i32 {
        if let Some(handle) = self.timer_handle.take() {
            let timer = handle.timer;
            let timer_queue = handle.timer_queue;
            unsafe {
                let res = threadpoollegacyapiset::DeleteTimerQueueTimer(
                    timer_queue,
                    timer,
                    INVALID_HANDLE_VALUE,
                );
                if res != 1 {
                    let err = errhandlingapi::GetLastError();
                    if err != ERROR_IO_PENDING {
                        eprintln!("Delete timer queue failed.");
                    } else {
                        std::thread::sleep(std::time::Duration::from_millis(200));
                    }
                }
                res
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
