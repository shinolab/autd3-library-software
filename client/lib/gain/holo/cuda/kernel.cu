/*
 * File: kernel.cu
 * Project: cuda
 * Created Date: 06/09/2021
 * Author: Shun Suzuki
 * -----
 * Last Modified: 06/09/2021
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2021 Hapis Lab. All rights reserved.
 *
 */

#include "./kernel.h"
#include "autd3/core/hardware_defined.hpp"

#define BLOCK_SIZE (32)

namespace autd {
namespace gain {
namespace holo {

__global__ void make_complex_kernel(double* r, double* i, const uint32_t row, const uint32_t col, cuDoubleComplex* c) {
  int xi = blockIdx.x * blockDim.x + threadIdx.x;
  int yi = blockIdx.y * blockDim.y + threadIdx.y;
  if (xi >= row || yi >= col) return;

  int idx = xi + yi * row;
  c[idx] = make_cuDoubleComplex(r[idx], i[idx]);
}

void cu_make_complex(double* r, double* i, const uint32_t row, const uint32_t col, cuDoubleComplex* c) {
  dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 grid((row - 1) / BLOCK_SIZE + 1, (col - 1) / BLOCK_SIZE + 1, 1);
  make_complex_kernel<<<grid, block>>>(r, i, row, col, c);
}

__device__ cuDoubleComplex expc(cuDoubleComplex x) {
  double s = exp(x.x);
  double r = cos(x.y);
  double i = sin(x.y);
  return make_cuDoubleComplex(s * r, s * i);
}

__global__ void exp_kernel(const uint32_t row, const uint32_t col, cuDoubleComplex* c) {
  int xi = blockIdx.x * blockDim.x + threadIdx.x;
  int yi = blockIdx.y * blockDim.y + threadIdx.y;
  if (xi >= row || yi >= col) return;

  int idx = xi + yi * row;
  c[idx] = expc(c[idx]);
}

void cu_exp(const uint32_t row, const uint32_t col, cuDoubleComplex* c) {
  dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 grid((row - 1) / BLOCK_SIZE + 1, (col - 1) / BLOCK_SIZE + 1, 1);
  exp_kernel<<<grid, block>>>(row, col, c);
}

__global__ void hadamard_product_kernel(const cuDoubleComplex* a, const cuDoubleComplex* b, const uint32_t row, const uint32_t col,
                                        cuDoubleComplex* c) {
  int xi = blockIdx.x * blockDim.x + threadIdx.x;
  int yi = blockIdx.y * blockDim.y + threadIdx.y;
  if (xi >= row || yi >= col) return;

  int idx = xi + yi * row;
  c[idx] = make_cuDoubleComplex(a[idx].x * b[idx].x - a[idx].y * b[idx].y, a[idx].x * b[idx].y + a[idx].y * b[idx].x);
}

void cu_hadamard_product(const cuDoubleComplex* a, const cuDoubleComplex* b, uint32_t row, uint32_t col, cuDoubleComplex* c) {
  dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 grid((row - 1) / BLOCK_SIZE + 1, (col - 1) / BLOCK_SIZE + 1, 1);
  hadamard_product_kernel<<<grid, block>>>(a, b, row, col, c);
}

__global__ void real_kernel(const cuDoubleComplex* a, const uint32_t row, const uint32_t col, double* b) {
  int xi = blockIdx.x * blockDim.x + threadIdx.x;
  int yi = blockIdx.y * blockDim.y + threadIdx.y;
  if (xi >= row || yi >= col) return;

  int idx = xi + yi * row;
  b[idx] = a[idx].x;
}

void cu_real(const cuDoubleComplex* a, uint32_t row, uint32_t col, double* b) {
  dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 grid((row - 1) / BLOCK_SIZE + 1, (col - 1) / BLOCK_SIZE + 1, 1);
  real_kernel<<<grid, block>>>(a, row, col, b);
}

__device__ double absc(cuDoubleComplex x) { return sqrt(x.x * x.x + x.y * x.y); }

__global__ void arg_kernel(const cuDoubleComplex* a, const uint32_t row, const uint32_t col, cuDoubleComplex* b) {
  int xi = blockIdx.x * blockDim.x + threadIdx.x;
  int yi = blockIdx.y * blockDim.y + threadIdx.y;
  if (xi >= row || yi >= col) return;

  int idx = xi + yi * row;
  double s = absc(a[idx]);
  b[idx] = make_cuDoubleComplex(a[idx].x / s, a[idx].y / s);
}

void cu_arg(const cuDoubleComplex* a, uint32_t row, uint32_t col, cuDoubleComplex* b) {
  dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 grid((row - 1) / BLOCK_SIZE + 1, (col - 1) / BLOCK_SIZE + 1, 1);
  arg_kernel<<<grid, block>>>(a, row, col, b);
}

__global__ void calc_singular_inv_kernel(double* d_s, uint32_t s_size, double alpha, cuDoubleComplex* p_singular_inv) {
  int xi = blockIdx.x * blockDim.x + threadIdx.x;
  int yi = blockIdx.y * blockDim.y + threadIdx.y;
  if (xi >= s_size || yi >= s_size) return;

  if (xi != yi)
    p_singular_inv[xi + yi * s_size] = make_cuDoubleComplex(0.0, 0.0);
  else
    p_singular_inv[xi + yi * s_size] = make_cuDoubleComplex(d_s[xi] / (d_s[xi] * d_s[xi] + alpha), 0.0);
}

void calc_singular_inv(double* d_s, uint32_t s_size, double alpha, cuDoubleComplex* p_singular_inv) {
  dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 grid((s_size - 1) / BLOCK_SIZE + 1, (s_size - 1) / BLOCK_SIZE + 1, 1);
  calc_singular_inv_kernel<<<grid, block>>>(d_s, s_size, alpha, p_singular_inv);
}

template <typename T>
__device__ T clamp(T v, T min, T max) {
  return v < min ? min : v > max ? max : v;
}

__device__ uint8_t to_duty(const double amp) {
  const auto d = asin(clamp(amp, 0.0, 1.0)) / M_PI;
  return (uint8_t)(510.0 * d);
}

__device__ uint8_t to_phase(const double phase) noexcept {
  const uint8_t d_phase = (uint8_t)((int)(round(phase * 256.0)) & 0xFF);
  return core::PHASE_INVERTED ? d_phase : 0xFF - d_phase;
}

__global__ void set_from_complex_drive_kernel(const cuDoubleComplex* drive, uint32_t size, bool normalize, double max_coefficient, uint16_t* d_data) {
  int xi = blockIdx.x * blockDim.x + threadIdx.x;
  if (xi >= size) return;

  const auto f_amp = normalize ? 1.0 : absc(drive[xi]) / max_coefficient;
  const auto f_phase = atan2(drive[xi].y, drive[xi].x) / (2.0 * M_PI);
  const uint16_t phase = (uint16_t)to_phase(f_phase);
  const uint16_t duty = (uint16_t)to_duty(f_amp);
  d_data[xi] = (duty << 8) | phase;
}

void cu_set_from_complex_drive(const cuDoubleComplex* drive, uint32_t size, bool normalize, double max_coefficient, uint16_t* d_data) {
  dim3 block(BLOCK_SIZE, 1, 1);
  dim3 grid((size - 1) / BLOCK_SIZE + 1, 1, 1);
  set_from_complex_drive_kernel<<<grid, block>>>(drive, size, normalize, max_coefficient, d_data);
}

__device__ double3 sub(const double3& a, const double3& b) {
  double x = a.x - b.x;
  double y = a.y - b.y;
  double z = a.z - b.z;
  return double3{x, y, z};
}

__device__ double dot(const double3& a, const double3& b) {
  double x2 = a.x * b.x;
  double y2 = a.y * b.y;
  double z2 = a.z * b.z;
  return x2 + y2 + z2;
}

__device__ double norm(const double3& a) { return sqrt(dot(a, a)); }

__device__ cuDoubleComplex transfer(double3& pos, double3& dir, double3 focus, double wavenum, double attenuation) {
  const auto diff = sub(focus, pos);
  const auto dist = norm(diff);
  // const auto theta = atan2(dot(diff, dir), dist * norm(dir)) * 180.0 / M_PI;
  // const auto directivity = Directivity::t4010a1(theta);
  const auto directivity = 1.0;

  const auto v = make_cuDoubleComplex(-dist * attenuation, -wavenum * dist);
  auto r = expc(v);
  r.x *= directivity / dist;
  r.y *= directivity / dist;
  return r;
}

__global__ void transfer_matrix_kernel(const double3* foci, uint32_t foci_num, const double3* positions, const double3* directions,
                                       uint32_t trans_num, double wavenum, double attenuation, cuDoubleComplex* result) {
  int xi = blockIdx.x * blockDim.x + threadIdx.x;
  int yi = blockIdx.y * blockDim.y + threadIdx.y;
  if (xi >= foci_num || yi >= trans_num) return;

  int dev_idx = yi / core::NUM_TRANS_IN_UNIT;

  double3 focus = foci[xi];
  double3 pos = positions[yi];
  double3 dir = directions[dev_idx];
  result[xi + foci_num * yi] = transfer(pos, dir, focus, wavenum, attenuation);
}

void cu_transfer_matrix(const double3* foci, uint32_t foci_num, const double3* positions, const double3* directions, uint32_t trans_num,
                        double wavenum, double attenuation, cuDoubleComplex* result) {
  dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 grid((foci_num - 1) / BLOCK_SIZE + 1, (trans_num - 1) / BLOCK_SIZE + 1, 1);
  transfer_matrix_kernel<<<grid, block>>>(foci, foci_num, positions, directions, trans_num, wavenum, attenuation, result);
}
}  // namespace holo
}  // namespace gain
}  // namespace autd
