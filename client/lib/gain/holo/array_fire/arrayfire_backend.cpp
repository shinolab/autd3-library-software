
#include "autd3/gain/arrayfire_backend.hpp"

#include "arrayfire.h"

namespace autd::gain::holo {
template <typename T>
struct AFMatrix final : public Matrix<T> {
  explicit AFMatrix(Eigen::Index row, Eigen::Index col);
  ~AFMatrix() override = default;
  AFMatrix(const AFMatrix& obj) = delete;
  AFMatrix& operator=(const AFMatrix& obj) = delete;
  AFMatrix(const AFMatrix&& v) = delete;
  AFMatrix& operator=(AFMatrix&& obj) = delete;

  [[nodiscard]] T at(size_t row, size_t col) const override { throw std::runtime_error("not implemented"); }
  [[nodiscard]] const T* ptr() const override { throw std::runtime_error("not implemented"); }
  T* ptr() override { throw std::runtime_error("not implemented"); }
  [[nodiscard]] double max_element() const override { throw std::runtime_error("not implemented"); }
  void set(const Eigen::Index row, const Eigen::Index col, T v) override { throw std::runtime_error("not implemented"); }
  void get_col(const Eigen::Index i, std::shared_ptr<Matrix<T>> dst) override { throw std::runtime_error("not implemented"); }
  void fill(T v) override { throw std::runtime_error("not implemented"); }
  void get_diagonal(std::shared_ptr<Matrix<T>> v) override {}
  void set_diagonal(std::shared_ptr<Matrix<T>> v) override {}
  void copy_from(const std::vector<T>& v) override {}
  void copy_from(const T* v) override {}
  void copy_to_host() override {}

 private:
  af::array _af_array;
};

template <>
AFMatrix<double>::AFMatrix(const Eigen::Index row, const Eigen::Index col) : Matrix<double>(row, col), _af_array(row, col, af::dtype::f64) {}

template <>
AFMatrix<complex>::AFMatrix(const Eigen::Index row, const Eigen::Index col) : Matrix<complex>(row, col), _af_array(row, col, af::dtype::f64) {}

ArrayFireBackend::ArrayFireBackend(int device_idx) {}

ArrayFireBackend::~ArrayFireBackend() = default;

std::shared_ptr<MatrixX> ArrayFireBackend::allocate_matrix(const std::string& name, size_t row, size_t col) {
  throw std::runtime_error("not implemented yet");
}
std::shared_ptr<MatrixXc> ArrayFireBackend::allocate_matrix_c(const std::string& name, size_t row, size_t col) {
  throw std::runtime_error("not implemented yet");
}

BackendPtr ArrayFireBackend::create(int device_idx) { throw std::runtime_error("not implemented yet"); }

void ArrayFireBackend::make_complex(std::shared_ptr<MatrixX> r, std::shared_ptr<MatrixX> i, std::shared_ptr<MatrixXc> c) {}
void ArrayFireBackend::exp(std::shared_ptr<MatrixXc> a) {}
void ArrayFireBackend::scale(std::shared_ptr<MatrixXc> a, complex s) {}
void ArrayFireBackend::hadamard_product(std::shared_ptr<MatrixXc> a, std::shared_ptr<MatrixXc> b, std::shared_ptr<MatrixXc> c) {}
void ArrayFireBackend::real(std::shared_ptr<MatrixXc> a, std::shared_ptr<MatrixX> b) {}
void ArrayFireBackend::arg(std::shared_ptr<MatrixXc> a, std::shared_ptr<MatrixXc> c) {}
void ArrayFireBackend::pseudo_inverse_svd(std::shared_ptr<MatrixXc> matrix, double alpha, std::shared_ptr<MatrixXc> result) {}
void ArrayFireBackend::pseudo_inverse_svd(std::shared_ptr<MatrixX> matrix, double alpha, std::shared_ptr<MatrixX> result) {}
void ArrayFireBackend::max_eigen_vector(std::shared_ptr<MatrixXc> matrix, std::shared_ptr<MatrixXc> ev) {}
void ArrayFireBackend::matrix_add(double alpha, std::shared_ptr<MatrixX> a, std::shared_ptr<MatrixX> b) {}
void ArrayFireBackend::matrix_add(complex alpha, std::shared_ptr<MatrixXc> a, std::shared_ptr<MatrixXc> b) {}
void ArrayFireBackend::matrix_mul(TRANSPOSE trans_a, TRANSPOSE trans_b, complex alpha, std::shared_ptr<MatrixXc> a, std::shared_ptr<MatrixXc> b,
                                  complex beta, std::shared_ptr<MatrixXc> c) {}
void ArrayFireBackend::matrix_mul(TRANSPOSE trans_a, TRANSPOSE trans_b, double alpha, std::shared_ptr<MatrixX> a, std::shared_ptr<MatrixX> b,
                                  double beta, std::shared_ptr<MatrixX> c) {}
void ArrayFireBackend::solve_ch(std::shared_ptr<MatrixXc> a, std::shared_ptr<MatrixXc> b) {}
void ArrayFireBackend::solve_g(std::shared_ptr<MatrixX> a, std::shared_ptr<MatrixX> b, std::shared_ptr<MatrixX> c) {}
double ArrayFireBackend::dot(std::shared_ptr<MatrixX> a, std::shared_ptr<MatrixX> b) { throw std::runtime_error("not implemented yet"); }
complex ArrayFireBackend::dot(std::shared_ptr<MatrixXc> a, std::shared_ptr<MatrixXc> b) { throw std::runtime_error("not implemented yet"); }
double ArrayFireBackend::max_coefficient(std::shared_ptr<MatrixX> v) { throw std::runtime_error("not implemented yet"); }
double ArrayFireBackend::max_coefficient(std::shared_ptr<MatrixXc> v) { throw std::runtime_error("not implemented yet"); }
std::shared_ptr<MatrixXc> ArrayFireBackend::concat_row(std::shared_ptr<MatrixXc> a, std::shared_ptr<MatrixXc> b) {
  throw std::runtime_error("not implemented yet");
}
std::shared_ptr<MatrixXc> ArrayFireBackend::concat_col(std::shared_ptr<MatrixXc> a, std::shared_ptr<MatrixXc> b) {
  throw std::runtime_error("not implemented yet");
}
void ArrayFireBackend::mat_cpy(std::shared_ptr<MatrixX> a, std::shared_ptr<MatrixX> b) {}
void ArrayFireBackend::mat_cpy(std::shared_ptr<MatrixXc> a, std::shared_ptr<MatrixXc> b) {}

void ArrayFireBackend::set_from_complex_drive(std::vector<core::DataArray>& data, std::shared_ptr<MatrixXc> drive, bool normalize,
                                              double max_coefficient) {}
std::shared_ptr<MatrixXc> ArrayFireBackend::transfer_matrix(const double* foci, size_t foci_num, const std::vector<const double*>& positions,
                                                            const std::vector<const double*>& directions, double wavelength, double attenuation) {
  throw std::runtime_error("not implemented yet");
}

void ArrayFireBackend::set_bcd_result(std::shared_ptr<MatrixXc> mat, std::shared_ptr<MatrixXc> vec, size_t index) {}
std::shared_ptr<MatrixXc> ArrayFireBackend::back_prop(std::shared_ptr<MatrixXc> transfer, std::shared_ptr<MatrixXc> amps) {
  throw std::runtime_error("not implemented yet");
}

std::shared_ptr<MatrixXc> ArrayFireBackend::sigma_regularization(std::shared_ptr<MatrixXc> transfer, std::shared_ptr<MatrixXc> amps, double gamma) {
  throw std::runtime_error("not implemented yet");
}
void ArrayFireBackend::col_sum_imag(std::shared_ptr<MatrixXc> mat, std::shared_ptr<MatrixX> dst) {}
}  // namespace autd::gain::holo
