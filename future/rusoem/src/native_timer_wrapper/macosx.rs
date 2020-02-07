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

use std::ffi::CString;

use libc::{c_char, c_long, c_ulong, c_void, uintptr_t};

#[allow(non_camel_case_types)]
type dispatch_object_t = *const c_void;
#[allow(non_camel_case_types)]
type dispatch_queue_t = *const c_void;
#[allow(non_camel_case_types)]
type dispatch_source_t = *const c_void;
#[allow(non_camel_case_types)]
type dispatch_source_type_t = *const c_void;
#[allow(non_camel_case_types)]
type dispatch_time_t = u64;

const DISPATCH_TIME_NOW: dispatch_time_t = 0;

type WAITORTIMERCALLBACKN = unsafe extern "C" fn(*mut c_void);
type WAITORTIMERCALLBACK = Option<WAITORTIMERCALLBACKN>;

extern "C" {
    static _dispatch_source_type_timer: c_long;
    fn dispatch_queue_create(label: *const c_char, attr: c_ulong) -> dispatch_queue_t;
    fn dispatch_source_create(
        type_: dispatch_source_type_t,
        handle: uintptr_t,
        mask: c_ulong,
        queue: dispatch_queue_t,
    ) -> dispatch_source_t;
    fn dispatch_source_set_timer(
        source: dispatch_source_t,
        start: dispatch_time_t,
        interval: u64,
        leeway: u64,
    );
    fn dispatch_source_set_event_handler_f(
        source: dispatch_source_t,
        handler: WAITORTIMERCALLBACKN,
    );
    fn dispatch_resume(object: dispatch_object_t);
    fn dispatch_release(object: dispatch_object_t);
    fn dispatch_time(when: dispatch_time_t, delta: i64) -> dispatch_time_t;
}

struct TimerHandle {
    timer: dispatch_source_t,
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
            let queue = dispatch_queue_create(CString::new("timerQueue").unwrap().as_ptr(), 0);
            let timer = dispatch_source_create(
                &_dispatch_source_type_timer as *const _ as dispatch_source_type_t,
                0,
                0,
                queue,
            );
            if let Some(cb) = cb {
                dispatch_source_set_event_handler_f(timer, cb);
            }

            let start = dispatch_time(DISPATCH_TIME_NOW, 0);
            dispatch_source_set_timer(timer, start, period_ns as u64, 0);
            dispatch_resume(timer);

            self.timer_handle = Some(TimerHandle { timer });

            1
        }
    }

    pub fn close(&mut self) -> i32 {
        if let Some(handle) = self.timer_handle.take() {
            unsafe {
                dispatch_release(handle.timer);
                1
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
