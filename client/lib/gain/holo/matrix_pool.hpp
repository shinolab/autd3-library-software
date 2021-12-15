// File: matrix_pool.hpp
// Project: gain
// Created Date: 06/07/2021
// Author: Shun Suzuki
// -----
// Last Modified: 10/09/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <complex>
#include <memory>
#include <string>
#include <unordered_map>

namespace autd {
namespace gain {
namespace holo {

/**
 * \brief Linear algebra calculation backend
 */
template <typename M, typename Mc>
class MatrixBufferPool final {
 public:
  MatrixBufferPool() = default;
  ~MatrixBufferPool() = default;
  MatrixBufferPool(const MatrixBufferPool& obj) = delete;
  MatrixBufferPool& operator=(const MatrixBufferPool& obj) = delete;
  MatrixBufferPool(const MatrixBufferPool&& v) = delete;
  MatrixBufferPool& operator=(MatrixBufferPool&& obj) = delete;

  std::shared_ptr<M> rent(const std::string& name, const size_t row, const size_t col) { return rent_impl<M>(name, row, col, _cache_mat); }
  std::shared_ptr<Mc> rent_c(const std::string& name, const size_t row, const size_t col) { return rent_impl<Mc>(name, row, col, _cache_mat_c); }

 private:
  std::unordered_map<std::string, std::shared_ptr<M>> _cache_mat;
  std::unordered_map<std::string, std::shared_ptr<Mc>> _cache_mat_c;

  template <typename T>
  static std::shared_ptr<T> rent_impl(const std::string& name, const size_t row, const size_t col,
                                      std::unordered_map<std::string, std::shared_ptr<T>>& cache) {
    const auto it = cache.find(name);
    if (it != cache.end()) {
      if (static_cast<size_t>(it->second->rows()) == row && static_cast<size_t>(it->second->cols()) == col) return it->second;
      cache.erase(name);
    }
    auto v = std::make_shared<T>(row, col);
    cache.emplace(name, v);
    return v;
  }
};

}  // namespace holo
}  // namespace gain
}  // namespace autd