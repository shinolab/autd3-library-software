/*
 * File: geometry.rs
 * Project: autd
 * Created Date: 02/09/2019
 * Author: Shun Suzuki
 * -----
 * Last Modified: 28/11/2019
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2019 Hapis Lab. All rights reserved.
 *
 */

use crate::consts::{NUM_TRANS_IN_UNIT, NUM_TRANS_X, NUM_TRANS_Y, TRANS_SIZE};
use crate::utils;
use crate::utils::{Quaternionf, Vector3f};

struct Device {
    device_id: usize,
    global_trans_positions: Vec<Vector3f>,
    x_direction: Vector3f,
    y_direction: Vector3f,
    z_direction: Vector3f,
}

impl Device {
    pub fn new(device_id: usize, position: Vector3f, rotation: Quaternionf) -> Device {
        let trans_mat = utils::tr(position, rotation);
        let x_direction = (rotation * Quaternionf::from_imag(Vector3f::x()) * rotation.conjugate())
            .imag()
            .normalize();
        let y_direction = (rotation * Quaternionf::from_imag(Vector3f::y()) * rotation.conjugate())
            .imag()
            .normalize();
        let z_direction = (rotation * Quaternionf::from_imag(Vector3f::z()) * rotation.conjugate())
            .imag()
            .normalize();

        let mut global_trans_positions = Vec::with_capacity(NUM_TRANS_IN_UNIT);
        for y in 0..NUM_TRANS_Y {
            for x in 0..NUM_TRANS_X {
                if !utils::is_missing_transducer(x, y) {
                    let local_pos =
                        na::Vec4::<f32>::new(x as f32 * TRANS_SIZE, y as f32 * TRANS_SIZE, 0., 1.);
                    global_trans_positions.push(utils::convert_to_vec3(trans_mat * local_pos));
                }
            }
        }

        Device {
            device_id,
            global_trans_positions,
            x_direction,
            y_direction,
            z_direction,
        }
    }
}

#[derive(Default)]
pub struct Geometry {
    devices: Vec<Device>,
}

impl Geometry {
    pub fn add_device(&mut self, positoin: Vector3f, euler_angles: Vector3f) -> usize {
        let q = utils::angle_axis(euler_angles.x, Vector3f::z())
            * utils::angle_axis(euler_angles.y, Vector3f::y())
            * utils::angle_axis(euler_angles.z, Vector3f::z());
        self.add_device_quaternion(positoin, q)
    }

    pub fn add_device_quaternion(&mut self, positoin: Vector3f, rotation: Quaternionf) -> usize {
        let device_id = self.devices.len();
        self.devices
            .push(Device::new(device_id, positoin, rotation));
        device_id
    }

    pub fn del_device(&mut self, device_id: usize) {
        let mut index = 0;
        for (i, dev) in self.devices.iter().enumerate() {
            if dev.device_id == device_id {
                index = i;
                break;
            }
        }
        self.devices.remove(index);
    }

    pub fn num_devices(&self) -> usize {
        self.devices.len()
    }

    pub fn position(&self, transducer_id: usize) -> Vector3f {
        let local_trans_id = transducer_id % NUM_TRANS_IN_UNIT;
        let device = self.device(transducer_id);
        device.global_trans_positions[local_trans_id]
    }

    pub fn local_position(&self, device_id: usize, global_position: Vector3f) -> Vector3f {
        let device = &self.devices[device_id];
        let local_origin = device.global_trans_positions[0];
        let x_dir = device.x_direction;
        let y_dir = device.y_direction;
        let z_dir = device.z_direction;
        let rv = global_position - local_origin;
        Vector3f::new(rv.dot(&x_dir), rv.dot(&y_dir), rv.dot(&z_dir))
    }

    pub fn direction(&self, transducer_id: usize) -> Vector3f {
        let device = self.device(transducer_id);
        device.z_direction
    }

    fn device(&self, transducer_id: usize) -> &Device {
        let eid = transducer_id / NUM_TRANS_IN_UNIT;
        &self.devices[eid]
    }
}
