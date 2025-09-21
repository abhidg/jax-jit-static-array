
#include <cstdint>
#include <utility>
#include <iostream>

#include "nanobind/nanobind.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace nb = nanobind;
namespace ffi = xla::ffi;

template <ffi::DataType T>
std::pair<int64_t, int64_t> GetDims(const ffi::Buffer<T> &buffer) {
  auto dims = buffer.dimensions();
  if (dims.size() == 0) {
    return std::make_pair(0, 0);
  }
  return std::make_pair(buffer.element_count(), dims.back());
}

void ComputeMatMul(const float *x, const float* mul, float *y) {
  // size of x is nrows * ncols
  int nrows = mul[0];
  int ncols = mul[1];
  // rest of mul contains row vector M that will be left multiplied to X
  // M = (x_1 ... x_R) . ( v_1 ... v_C) where R = #rows, C = #cols
  // giving an output of (z_1 ... z_C)

  for (size_t n = 0; n < ncols; ++n) {
    float s = 0;
    for (size_t i = 0; i < nrows; ++i) {
      s += mul[i + 2] * x[i + n * nrows];
    }
    y[n] = s;
  }
}

// A wrapper function providing the interface between the XLA FFI call and our
// library function `ComputeRmsNorm` above. This function handles the batch
// dimensions by calling `ComputeRmsNorm` within a loop.
ffi::Error MatMulImpl(ffi::Buffer<ffi::F32> x, ffi::Buffer<ffi::F32> mul,
                       ffi::ResultBuffer<ffi::F32> y) {
  auto [totalSize, lastDim] = GetDims(x);
  if (lastDim == 0) {
    return ffi::Error::InvalidArgument("MatMul input must be an array");
  }
  for (int64_t n = 0; n < totalSize; n += lastDim) {
    ComputeMatMul(&(x.typed_data()[n]), mul.typed_data(), &(y->typed_data()[n]));
  }
  return ffi::Error::Success();
}

// Wrap `RmsNormImpl` and specify the interface to XLA. If you need to declare
// this handler in a header, you can use the `XLA_FFI_DECLARE_HANDLER_SYMBOL`
// macro: `XLA_FFI_DECLARE_HANDLER_SYMBOL(RmsNorm)`.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    MatMul, MatMulImpl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::F32>>()  // x
        .Arg<ffi::Buffer<ffi::F32>>()  // mul
        .Ret<ffi::Buffer<ffi::F32>>()  // y
);

template <typename T>
nb::capsule EncapsulateFfiHandler(T *fn) {
  static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                "Encapsulated function must be and XLA FFI handler");
  return nb::capsule(reinterpret_cast<void *>(fn));
}

NB_MODULE(_matmul, m) {
  m.def("registrations", []() {
    nb::dict registrations;
    registrations["matmul"] = EncapsulateFfiHandler(MatMul);
    return registrations;
  });
}
