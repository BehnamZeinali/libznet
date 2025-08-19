#include "znet/autograd/shape_ops.hpp"
#include <stdexcept>
#include <vector>
#include <cstdint>
#include <algorithm>

namespace znet {

// ---- helpers: 64-bit, row-major math on logical shapes ----
static inline int64_t numel_of(const std::vector<int64_t>& sizes) {
    int64_t n = 1;
    for (int64_t s : sizes) n *= s;
    return n;
}

static inline std::vector<int64_t> rowmajor_strides(const std::vector<int64_t>& sizes) {
    std::vector<int64_t> s(sizes.size());
    int64_t stride = 1;
    for (int i = static_cast<int>(sizes.size()) - 1; i >= 0; --i) {
        s[static_cast<size_t>(i)] = stride;
        stride *= sizes[static_cast<size_t>(i)];
    }
    return s;
}

static inline std::vector<int64_t> left_pad_sizes(const std::vector<int64_t>& v, int ndim) {
    std::vector<int64_t> r(static_cast<size_t>(ndim) - v.size(), 1); // leading 1s
    r.insert(r.end(), v.begin(), v.end());
    return r;
}

// ---- sum_to_shape: reduce src to dst_shape by summing broadcasted dims ----
Tensor sum_to_shape(const Tensor& src, const std::vector<int64_t>& dst_shape_in) {
    const auto& sshape = src.shape();
    const int   sN = static_cast<int>(sshape.size());
    const int   dN = static_cast<int>(dst_shape_in.size());
    if (dN > sN) {
        throw std::runtime_error("sum_to_shape: dst rank > src rank");
    }

    // Align dst shape to src rank by left-padding with 1s
    const std::vector<int64_t> dst_shape = dst_shape_in;
    const std::vector<int64_t> sp = sshape;
    const std::vector<int64_t> dp = left_pad_sizes(dst_shape, sN); // length == sN

    // Validate broadcasting compatibility (right-aligned)
    for (int d = 0; d < sN; ++d) {
        if (!(sp[static_cast<size_t>(d)] == dp[static_cast<size_t>(d)] ||
              dp[static_cast<size_t>(d)] == 1)) {
            throw std::runtime_error("sum_to_shape: incompatible shapes for reduction");
        }
    }

    const int64_t src_numel = numel_of(sp);
    const int64_t dst_numel = numel_of(dst_shape);

    // Fast path: [B, F] -> [F] and src is contiguous (common case)
    if (sN == 2 && dN == 1 &&
        sp[1] == dst_shape[0] &&
        src.is_contiguous())
    {
        const int64_t B = sp[0];
        const int64_t F = sp[1];
        std::vector<float> out(static_cast<size_t>(F), 0.0f);
        const float* sdata = src.data_ptr() + src.storage_offset();
        for (int64_t i = 0; i < B; ++i) {
            const int64_t base = i * F;
            for (int64_t j = 0; j < F; ++j) {
                out[static_cast<size_t>(j)] += sdata[static_cast<size_t>(base + j)];
            }
        }
        return Tensor({F}, std::move(out), /*requires_grad=*/false);
    }

    // General path (works for non-contiguous src):
    // Enumerate all logical src indices via row-major decoding (independent of storage),
    // map each to a dst index (collapsing reduced dims), and accumulate.

    std::vector<float> out(static_cast<size_t>(dst_numel), 0.0f);

    // Precompute logical (row-major) strides for index decoding/encoding
    const auto sstrides = rowmajor_strides(sp);        // for decoding lin -> src_idx
    const auto dstrides = rowmajor_strides(dst_shape); // for encoding dst_idx -> lin

    // If src is contiguous, we can read by linear index; otherwise use at(src_idx)
    const bool src_contig = src.is_contiguous();
    const float* sdata = src_contig ? (src.data_ptr() + src.storage_offset()) : nullptr;

    // Buffers reused in the loop
    std::vector<int64_t> src_idx(static_cast<size_t>(sN), 0);
    std::vector<int64_t> dst_idx(static_cast<size_t>(dN), 0);

    for (int64_t lin = 0; lin < src_numel; ++lin) {
        // Decode src linear index to src_idx (row-major)
        int64_t t = lin;
        for (int d = 0; d < sN; ++d) {
            const int64_t dim_size = sp[static_cast<size_t>(d)];
            const int64_t stride_d = sstrides[static_cast<size_t>(d)];
            src_idx[static_cast<size_t>(d)] = (t / stride_d) % dim_size;
            t %= stride_d;
        }

        // Build aligned dst_idx (right-aligned w.r.t. src)
        for (int d = 0; d < dN; ++d) {
            const int src_dim = d + (sN - dN); // align right
            dst_idx[static_cast<size_t>(d)] =
                (dp[static_cast<size_t>(src_dim)] == 1) ? 0 : src_idx[static_cast<size_t>(src_dim)];
        }

        // Encode dst_idx to linear index (row-major)
        int64_t dl = 0;
        for (int d = 0; d < dN; ++d) {
            dl += dst_idx[static_cast<size_t>(d)] * dstrides[static_cast<size_t>(d)];
        }

        // Accumulate
        const float val = src_contig ? sdata[static_cast<size_t>(lin)]
                                     : src.at(src_idx);
        out[static_cast<size_t>(dl)] += val;
    }

    return Tensor(dst_shape, std::move(out), /*requires_grad=*/false);
}

} // namespace znet
