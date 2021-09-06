// File: kernel.h
// Project: cuda
// Created Date: 06/09/2021
// Author: Shun Suzuki
// -----
// Last Modified: 06/09/2021
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
void cu_make_complex(double* r, double* i, uint32_t row, uint32_t col, cuDoubleComplex* c);
void cu_exp(uint32_t row, uint32_t col, cuDoubleComplex* c);
void cu_hadamard_product(const cuDoubleComplex* a, const cuDoubleComplex* b, uint32_t row, uint32_t col, cuDoubleComplex* c);
void cu_real(const cuDoubleComplex* a, uint32_t row, uint32_t col, double* b);
void cu_arg(const cuDoubleComplex* a, uint32_t row, uint32_t col, cuDoubleComplex* b);
void calc_singular_inv(double* d_s, uint32_t s_size, double alpha, cuDoubleComplex* p_singular_inv);

}  // namespace holo
}  // namespace gain
}  // namespace autd
