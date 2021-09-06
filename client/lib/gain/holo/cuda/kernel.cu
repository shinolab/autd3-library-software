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

__global__ void exp_kernel(const uint32_t row, const uint32_t col, cuDoubleComplex* c) {
  int xi = blockIdx.x * blockDim.x + threadIdx.x;
  int yi = blockIdx.y * blockDim.y + threadIdx.y;
  if (xi >= row || yi >= col) return;

  int idx = xi + yi * row;
  cuDoubleComplex x = c[idx];
  double s = exp(x.x);
  double r = cos(x.y);
  double i = sin(x.y);
  c[idx] = make_cuDoubleComplex(s * r, s * i);
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

__global__ void arg_kernel(const cuDoubleComplex* a, const uint32_t row, const uint32_t col, cuDoubleComplex* b) {
  int xi = blockIdx.x * blockDim.x + threadIdx.x;
  int yi = blockIdx.y * blockDim.y + threadIdx.y;
  if (xi >= row || yi >= col) return;

  int idx = xi + yi * row;
  double s = sqrt(a[idx].x * a[idx].x + a[idx].y * a[idx].y);
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
}  // namespace holo
}  // namespace gain
}  // namespace autd
