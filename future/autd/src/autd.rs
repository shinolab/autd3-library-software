/*
 * File: controller.rs
 * Project: autd
 * Created Date: 02/09/2019
 * Author: Shun Suzuki
 * -----
 * Last Modified: 05/02/2020
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2019 Hapis Lab. All rights reserved.
 *
 */

use std::collections::VecDeque;
use std::mem::size_of;
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::thread::{self, JoinHandle};

use chrono::Duration;
use timer::Timer;

use crate::consts::{MOD_FRAME_SIZE, NUM_TRANS_IN_UNIT};
use crate::gain::{Gain, NullGain};
use crate::geometry::Geometry;
#[cfg(feature = "dbg_link")]
use crate::link::DbgLink;
#[cfg(all(windows, feature = "twincat"))]
use crate::link::EtherCATLink;
#[cfg(all(windows, feature = "twincat"))]
use crate::link::LocalEtherCATLink;
use crate::link::{Link, LinkType, SoemLink};
use crate::modulation::Modulation;
use crate::rx_global_header::{RxGlobalControlFlags, RxGlobalHeader};
use crate::utils::*;

use crate::utils::Vector3f;

type GainPtr = Box<dyn Gain>;
type GainQueue = VecDeque<GainPtr>;
type ModulationQueue = VecDeque<Modulation>;
struct SendQueue {
    gain_q: GainQueue,
    modulation_q: ModulationQueue,
}

/// The structure that controls AUTDs.
#[repr(C)]
pub struct AUTD {
    geometry: Arc<Mutex<Geometry>>,
    is_open: Arc<RwLock<bool>>,
    is_silent: Arc<RwLock<bool>>,
    link: Option<Arc<Mutex<Box<dyn Link>>>>,
    build_gain_q: Arc<(Mutex<GainQueue>, Condvar)>,
    send_gain_q: Arc<(Mutex<SendQueue>, Condvar)>,
    build_th_handle: Option<JoinHandle<()>>,
    sned_th_handle: Option<JoinHandle<()>>,
    stm_gains: Arc<Mutex<Vec<GainPtr>>>,
    stm_timer: Timer,
    stm_timer_guard: Option<timer::Guard>,
}

impl AUTD {
    /// constructor
    pub fn create() -> AUTD {
        let send_gain_q = Arc::new((
            Mutex::new(SendQueue {
                gain_q: GainQueue::new(),
                modulation_q: ModulationQueue::new(),
            }),
            Condvar::new(),
        ));
        AUTD {
            link: None,
            is_open: Arc::new(RwLock::new(true)),
            is_silent: Arc::new(RwLock::new(true)),
            geometry: Arc::new(Mutex::new(Default::default())),
            build_gain_q: Arc::new((Mutex::new(VecDeque::new()), Condvar::new())),
            send_gain_q,
            build_th_handle: None,
            sned_th_handle: None,
            stm_gains: Arc::new(Mutex::new(Vec::new())),
            stm_timer: Timer::new(),
            stm_timer_guard: None,
        }
    }

    /// Open AUTDs
    ///
    /// You must call [add_device](#method.add_device) or [add_device_quaternion](#method.add_device_quaternion) before.
    ///
    /// # Arguments
    ///
    /// * `ifname` - A string slice that holds the name of the interface name conneted to AUTDs.
    ///             With SOEM, you can get ifname via [EthernetAdapters](../struct.EthernetAdapters.html).
    /// * `link_type` - Only SOEM is supported.
    ///
    /// # Example
    ///
    /// ```
    /// use autd::AUTD;
    /// use autd::utils::Vector3f;
    ///
    /// let autd = AUTD::create();
    ///
    /// autd.add_device(Vector3f::zeros(), Vector3f::zeros());
    ///
    /// match autd.open(ifname, LinkType::SOEM) {
    ///     Ok(()) => (),
    ///     Err(e) => println!("{}", e),
    /// }
    /// ```
    pub fn open(&mut self, ifname: &str, link_type: LinkType) -> Result<(), failure::Error> {
        let link = match link_type {
            LinkType::SOEM => SoemLink::open(ifname, self.num_devices() as u16)? as Box<dyn Link>,
            #[cfg(feature = "dbg_link")]
            LinkType::DBG => DbgLink::open(ifname)? as Box<dyn Link>,
            #[cfg(all(windows, feature = "twincat"))]
            LinkType::TwinCAT => {
                if let Ok(mut props) = self.props.write() {
                    props.is_sync_first_sync0 = true;
                };

                if ifname == ""
                    || ifname.find("localhost").is_some()
                    || ifname.find("0.0.0.0").is_some()
                    || ifname.find("127.0.0.1").is_some()
                {
                    LocalEtherCATLink::open()? as Box<dyn Link>
                } else {
                    EtherCATLink::open(ifname)? as Box<dyn Link>
                }
            }
        };
        self.link = Some(Arc::new(Mutex::new(link)));
        self.init_pipeline();
        Ok(())
    }

    /// Add AUTD geometry to the controller.
    ///
    /// Use this method to specify the device geometry in order of proximity to the master.
    /// Call this method or [add_device_quaternion](#method.add_device_quaternion) as many times as the number of AUTDs connected to the master and not too many times.
    ///
    /// # Argumnts
    ///
    /// * `pos` - Global position of AUTD.
    /// * `rot` - ZYZ Euler angles.
    ///
    /// # Exmaple
    ///
    /// ```
    /// use autd::utils::Vector3f;
    /// use std::f32::consts::PI;
    ///
    /// autd.add_device(Vector3f::zeros(), Vector3f::zeros());
    /// autd.add_device(Vector3f::new(192., 0, 0,), Vector3f::new(-PI, 0, 0));
    /// ```
    pub fn add_device(&mut self, pos: Vector3f, rot: Vector3f) -> usize {
        let mut geo = self.geometry.lock().unwrap();
        geo.add_device(pos, rot)
    }

    /// Add AUTD geometry to the controller.
    ///
    /// Use this method to specify the device geometry in order of proximity to the master.
    /// Call this method or [add_device](#method.add_device) as many times as the number of AUTDs connected to the master and not too many times.
    ///
    /// # Argumnts
    ///
    /// * `pos` - Global position of AUTD.
    /// * `rot` - rotation quaternion.
    ///
    pub fn add_device_quaternion(&mut self, pos: Vector3f, rot: Quaternionf) -> usize {
        let mut geo = self.geometry.lock().unwrap();
        geo.add_device_quaternion(pos, rot)
    }

    pub fn del_device(&mut self, id: usize) {
        let mut geo = self.geometry.lock().unwrap();
        geo.del_device(id);
    }

    pub fn set_silentmode(&mut self, silent: bool) {
        if let Ok(mut is_silent) = self.is_silent.write() {
            *is_silent = silent;
        };
    }

    pub fn calibrate(&mut self) -> bool {
        let link = match &self.link {
            Some(link) => link.clone(),
            None => return false,
        };
        if self.num_devices() == 1 {
            return true;
        }
        let mut l = (&*link).lock().unwrap();
        l.calibrate()
    }

    pub fn close(mut self) {
        self.close_impl();
    }

    pub fn is_open(&self) -> bool {
        if let Ok(open) = self.is_open.read() {
            *open
        } else {
            false
        }
    }

    pub fn is_silent(&self) -> bool {
        if let Ok(is_silent) = self.is_silent.read() {
            *is_silent
        } else {
            true
        }
    }

    pub fn num_devices(&self) -> usize {
        let geometry = self.geometry.lock().unwrap();
        geometry.num_devices()
    }

    pub fn num_transducers(&self) -> usize {
        self.num_devices() * crate::consts::NUM_TRANS_IN_UNIT
    }

    pub fn remaining_in_buffer(&self) -> usize {
        let (build_lk, _) = &*self.build_gain_q;
        let remain_build = {
            let build_q = build_lk.lock().unwrap();
            build_q.len()
        };
        let (send_lk, _) = &*self.send_gain_q;
        let remain_send = {
            let send_q = send_lk.lock().unwrap();
            send_q.gain_q.len() + send_q.modulation_q.len()
        };
        remain_build + remain_send
    }

    pub fn stop(&mut self) {
        self.append_gain_sync(NullGain::create());
    }

    pub fn append_gain(&mut self, gain: GainPtr) {
        let (build_lk, build_cvar) = &*self.build_gain_q;
        {
            let mut build_q = build_lk.lock().unwrap();
            build_q.push_back(gain);
        }
        build_cvar.notify_one();
    }

    pub fn append_gain_sync(&mut self, mut gain: GainPtr) {
        let link = match &self.link {
            Some(link) => link.clone(),
            None => return,
        };
        let geometry = self.geometry.clone();
        let geo = geometry.lock().unwrap();
        gain.build(&geo);
        let is_silent = self.is_silent();
        let body = AUTD::make_body(Some(gain), None, &geo, is_silent);
        {
            let mut l = (&*link).lock().unwrap();
            l.send(body);
        }
    }

    pub fn append_modulation(&mut self, modulation: Modulation) {
        let (send_lk, send_cvar) = &*self.send_gain_q;
        {
            let mut deq = send_lk.lock().unwrap();
            deq.modulation_q.push_back(modulation);
        }
        send_cvar.notify_one();
    }

    pub fn append_modulation_sync(&mut self, modulation: Modulation) {
        let link = match &self.link {
            Some(link) => link.clone(),
            None => return,
        };
        let geometry = self.geometry.clone();
        let geo = geometry.lock().unwrap();

        let mut modulation = modulation;
        while modulation.sent < modulation.buffer.len() {
            let is_silent = self.is_silent();
            let body = AUTD::make_body(None, Some(&mut modulation), &geo, is_silent);
            {
                let mut l = (&*link).lock().unwrap();
                l.send(body);
            }
        }
    }

    pub fn append_stm_gains(&mut self, gains: Vec<GainPtr>) {
        self.stop_stm();
        let mut stm_gains = self.stm_gains.lock().unwrap();
        stm_gains.extend(gains);
    }

    pub fn start_stm(&mut self, freq: f32) {
        let len = { self.stm_gains.lock().unwrap().len() };
        assert!(len != 0);
        let itvl_ms = 1000. / freq / len as f32;
        let link = match &self.link {
            Some(link) => link.clone(),
            None => return,
        };
        let geometry = self.geometry.lock().unwrap();
        let is_silent = self.is_silent();
        let mut stm_gains = self.stm_gains.lock().unwrap();
        let mut body_q = Vec::<Vec<u8>>::new();
        for _ in 0..stm_gains.len() {
            if let Some(mut gain) = stm_gains.pop() {
                gain.build(&geometry);
                body_q.push(AUTD::make_body(Some(gain), None, &geometry, is_silent));
            }
        }

        let mut idx = 0;
        self.stm_timer_guard = Some(self.stm_timer.schedule_repeating(
            Duration::milliseconds(itvl_ms as i64),
            move || {
                let body = &body_q[idx % len];
                let mut body_copy = Vec::with_capacity(body.len());
                unsafe {
                    body_copy.set_len(body.len());
                    std::ptr::copy_nonoverlapping(
                        body.as_ptr(),
                        body_copy.as_mut_ptr(),
                        body.len(),
                    );
                }
                {
                    let mut l = (&*link).lock().unwrap();
                    l.send(body_copy);
                }
                idx = (idx + 1) % len;
            },
        ));
    }

    fn stop_stm(&mut self) {
        if let Some(guard) = self.stm_timer_guard.take() {
            drop(guard);
            self.stm_timer_guard = None;
        };
    }

    pub fn finish_stm(&mut self) {
        self.stop_stm();
        let mut stm_gains = self.stm_gains.lock().unwrap();
        stm_gains.clear();
    }

    pub fn flush(&mut self) {
        let (build_lk, _) = &*self.build_gain_q;
        {
            let mut build_q = build_lk.lock().unwrap();
            build_q.clear();
        }
        let (send_lk, _) = &*self.send_gain_q;
        {
            let mut send_q = send_lk.lock().unwrap();
            send_q.gain_q.clear();
            send_q.modulation_q.clear();
        }
    }

    fn init_pipeline(&mut self) {
        let link = match &self.link {
            Some(link) => link.clone(),
            None => return,
        };
        // Build thread
        let geometry = self.geometry.clone();
        let build_gain_q = self.build_gain_q.clone();
        let send_gain_q = self.send_gain_q.clone();
        let is_open = self.is_open.clone();
        self.build_th_handle = Some(thread::spawn(move || {
            let (build_lk, build_cvar) = &*build_gain_q;
            loop {
                if let Ok(open) = is_open.read() {
                    if !*open {
                        break;
                    }
                }
                let mut gain_q = build_lk.lock().unwrap();
                let gain = match gain_q.pop_front() {
                    None => {
                        let _ = build_cvar.wait(gain_q).unwrap();
                        continue;
                    }
                    Some(mut gain) => {
                        let geo = geometry.lock().unwrap();
                        gain.build(&geo);
                        gain
                    }
                };

                let (send_lk, send_cvar) = &*send_gain_q;
                {
                    let mut deq = send_lk.lock().unwrap();
                    deq.gain_q.push_back(gain);
                }
                send_cvar.notify_all();
            }
        }));

        // Send thread
        let send_gain_q = self.send_gain_q.clone();
        let geometry = self.geometry.clone();
        let is_open = self.is_open.clone();
        let is_silent = self.is_silent.clone();
        self.sned_th_handle = Some(thread::spawn(move || {
            let (send_lk, send_cvar) = &*send_gain_q;
            loop {
                if let Ok(open) = is_open.read() {
                    if !*open {
                        break;
                    }
                }
                let mut send_buf = send_lk.lock().unwrap();
                match (
                    send_buf.gain_q.pop_front(),
                    send_buf.modulation_q.get_mut(0),
                ) {
                    (None, None) => {
                        let _ = send_cvar.wait(send_buf).unwrap();
                    }
                    (Some(g), None) => {
                        let geo = geometry.lock().unwrap();
                        let is_silent = match is_silent.read() {
                            Ok(is_silent) => *is_silent,
                            Err(_) => true,
                        };
                        let body = AUTD::make_body(Some(g), None, &geo, is_silent);
                        {
                            let mut l = (&*link).lock().unwrap();
                            l.send(body);
                        }
                    }
                    (g, Some(m)) => {
                        let geo = geometry.lock().unwrap();
                        let is_silent = match is_silent.read() {
                            Ok(is_silent) => *is_silent,
                            Err(_) => true,
                        };
                        let body = AUTD::make_body(g, Some(m), &geo, is_silent);
                        {
                            let mut l = (&*link).lock().unwrap();
                            l.send(body);
                        }
                        if m.buffer.len() <= m.sent {
                            send_buf.modulation_q.pop_front();
                        }
                    }
                }
            }
        }));
    }

    fn make_body(
        gain: Option<GainPtr>,
        modulation: Option<&mut Modulation>,
        geometry: &Geometry,
        is_silent: bool,
    ) -> Vec<u8> {
        let num_devices = if gain.is_some() {
            geometry.num_devices()
        } else {
            0
        };
        let size = size_of::<RxGlobalHeader>() + NUM_TRANS_IN_UNIT * 2 * num_devices;

        let mut body = vec![0x00; size];
        let mut ctrl_flags = RxGlobalControlFlags::NONE;
        if is_silent {
            ctrl_flags |= RxGlobalControlFlags::SILENT;
        }
        let mut mod_data: &[u8] = &[];
        match modulation {
            None => (),
            Some(modulation) => {
                let mod_size = clamp(modulation.buffer.len() - modulation.sent, 0, MOD_FRAME_SIZE);
                if modulation.sent == 0 {
                    ctrl_flags |= RxGlobalControlFlags::LOOP_BEGIN;
                }
                if modulation.sent + mod_size >= modulation.buffer.len() {
                    ctrl_flags |= RxGlobalControlFlags::LOOP_END;
                }
                mod_data = &modulation.buffer[modulation.sent..(modulation.sent + mod_size)];
                modulation.sent += mod_size;
            }
        }
        unsafe {
            let header = RxGlobalHeader::new(ctrl_flags, mod_data);
            let src_ptr = &header as *const RxGlobalHeader as *const u8;
            let dst_ptr = body.as_mut_ptr();
            std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, size_of::<RxGlobalHeader>());
        }
        match gain {
            None => (),
            Some(gain) => {
                let mut cursor = size_of::<RxGlobalHeader>();
                let byte_size = NUM_TRANS_IN_UNIT * 2;
                let gain_ptr = gain.get_data().as_ptr();
                unsafe {
                    for i in 0..geometry.num_devices() {
                        let src_ptr = gain_ptr.add(i * byte_size);
                        let dst_ptr = body.as_mut_ptr().add(cursor);

                        std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, byte_size);
                        cursor += byte_size;
                    }
                }
            }
        }
        body
    }

    fn close_impl(&mut self) {
        if let Ok(open) = self.is_open.read() {
            if !*open {
                return;
            }
        }

        self.finish_stm();
        self.flush();
        self.append_gain_sync(NullGain::create());

        if let Ok(mut open) = self.is_open.write() {
            *open = false;
        }

        if let Some(jh) = self.build_th_handle.take() {
            let (_, build_cvar) = &*self.build_gain_q;
            build_cvar.notify_one();
            jh.join().unwrap();
        }

        if let Some(jh) = self.sned_th_handle.take() {
            let (_, send_cvar) = &*self.send_gain_q;
            send_cvar.notify_one();
            jh.join().unwrap();
        }

        match &self.link {
            Some(link) => (&*link).lock().unwrap().close(),
            None => {}
        };
    }
}

impl Drop for AUTD {
    fn drop(&mut self) {
        self.close_impl();
    }
}
