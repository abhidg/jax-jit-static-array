
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

void ComputeAltSquareFreeTensor(int64_t size, const float *x, const int64_t* degree_begin, float *y) {
  int basis_idx = 0;
  // start squaring at depth=1, followed by 3, 5, ...
  // set initial flag to true, immediately turned off by i = degree_begin[0] = 0
  bool should_square_this = true;
  for (size_t i = 0; i < size; ++i) {
    if (i == degree_begin[basis_idx]) {
      // moved onto the next depth, toggle should_square_this
      should_square_this = !should_square_this;
      basis_idx++;
    }
    y[i] = should_square_this ? x[i] * x[i] : x[i];
  }
}

// A wrapper function providing the interface between the XLA FFI call and our
// library function `ComputeRmsNorm` above. This function handles the batch
// dimensions by calling `ComputeRmsNorm` within a loop.
ffi::Error AltSquareFreeTensorImpl(
    int64_t size,
    ffi::Buffer<ffi::F32> x,
    ffi::Buffer<ffi::S64> degree_begin,
    ffi::ResultBuffer<ffi::F32> y) {
  auto [totalSize, lastDim] = GetDims(x);
  if (lastDim == 0) {
    return ffi::Error::InvalidArgument("AltSquareFreeTensorImpl input must be an array");
  }
  for (int64_t n = 0; n < totalSize; n += lastDim) {
    ComputeAltSquareFreeTensor(size, &(x.typed_data()[n]), degree_begin.typed_data(), &(y->typed_data()[n]));
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AltSquareFreeTensor, AltSquareFreeTensorImpl,
    ffi::Ffi::Bind()
        .Attr<int64_t>("size")  // size
        .Arg<ffi::Buffer<ffi::F32>>()  // x
        .Arg<ffi::Buffer<ffi::S64>>()  // degree_begin
        .Ret<ffi::Buffer<ffi::F32>>()  // y
);

template <typename T>
nb::capsule EncapsulateFfiHandler(T *fn) {
  static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                "Encapsulated function must be and XLA FFI handler");
  return nb::capsule(reinterpret_cast<void *>(fn));
}

NB_MODULE(_tensorbasis, m) {
  m.def("registrations", []() {
    nb::dict registrations;
    registrations["alt_square_free_tensor"] = EncapsulateFfiHandler(AltSquareFreeTensor);
    return registrations;
  });
}
