// File: kernel.h
// Project: cuda
// Created Date: 06/09/2021
// Author: Shun Suzuki
// -----
// Last Modified: 10/09/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <cuComplex.h>

#include <cstdint>

namespace autd {
namespace gain {
namespace holo {

constexpr uint32_t BLOCK_SIZE = 32;

template <typename T>
void cu_get_diagonal(const T* src, T* dst, uint32_t size);

template <typename T>
void cu_set_diagonal(const T* src, T* dst, uint32_t size);

void cu_make_complex(const double* r, const double* i, uint32_t row, uint32_t col, cuDoubleComplex* c);
void cu_exp(uint32_t row, uint32_t col, cuDoubleComplex* c);
void cu_exp(uint32_t row, uint32_t col, double* c);
void cu_reciprocal(uint32_t row, uint32_t col, const double* src, double* dst);
void cu_reciprocal(uint32_t row, uint32_t col, const cuDoubleComplex* src, cuDoubleComplex* dst);
void cu_abs(uint32_t row, uint32_t col, const double* src, double* dst);
void cu_abs(uint32_t row, uint32_t col, const cuDoubleComplex* src, cuDoubleComplex* dst);
void cu_hadamard_product(const double* a, const double* b, uint32_t row, uint32_t col, double* c);
void cu_hadamard_product(const cuDoubleComplex* a, const cuDoubleComplex* b, uint32_t row, uint32_t col, cuDoubleComplex* c);
void cu_real(const cuDoubleComplex* a, uint32_t row, uint32_t col, double* b);
void cu_arg(const cuDoubleComplex* a, uint32_t row, uint32_t col, cuDoubleComplex* b);
void calc_singular_inv(double* d_s, uint32_t s_size, double alpha, cuDoubleComplex* p_singular_inv);
void calc_singular_inv(double* d_s, uint32_t s_size, double alpha, double* p_singular_inv);
void cu_set_from_complex_drive(const cuDoubleComplex* drive, uint32_t size, bool normalize, double max_coefficient, uint16_t* d_data);
void cu_set_from_arg(const double* drive, uint32_t size, uint16_t* d_data);
void cu_transfer_matrix(const double3* foci, uint32_t foci_num, const double3* positions, const double3* directions, uint32_t trans_num,
                        double wavenum, double attenuation, cuDoubleComplex* result);
void cu_set_bcd_result(const cuDoubleComplex* vec, uint32_t m, uint32_t idx, cuDoubleComplex* mat);
void cu_col_sum_abs(const cuDoubleComplex* transfer, uint32_t m, uint32_t n, double* denominator, double* buffer);
void cu_make_back_prop(const cuDoubleComplex* amps, const double* denominator, const cuDoubleComplex* transfer, uint32_t m, uint32_t n,
                       cuDoubleComplex* b);
void cu_make_sigma_diagonal(const cuDoubleComplex* transfer, uint32_t m, uint32_t n, const cuDoubleComplex* amps, double gamma,
                            cuDoubleComplex* result, double* buffer);
void cu_col_sum_imag(const cuDoubleComplex* mat, uint32_t m, uint32_t n, double* result, double* buffer);
}  // namespace holo
}  // namespace gain
}  // namespace autd
