#pragma once
#include <vector>
#include "znet/autograd/tensor.hpp"

namespace znet {

// Reduce `src` to `dst_shape` by summing across broadcast/extra dims.
// Assumes row-major contiguous layout for now (storage_offset==0).
Tensor sum_to_shape(const Tensor& src, const std::vector<int64_t>& dst_shape);

} // namespace znet
