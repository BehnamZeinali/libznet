#include "znet/autograd/shape_ops.hpp"
#include <stdexcept>
#include <numeric>

namespace znet {

static inline int64_t numel_of(const std::vector<int>& sizes) {
    int64_t n = 1;
    for (int s : sizes) n *= s;
    return n;
}
static inline std::vector<int64_t> rowmajor_strides(const std::vector<int>& sizes) {
    std::vector<int64_t> s(sizes.size());
    int64_t stride = 1;
    for (int i = (int)sizes.size() - 1; i >= 0; --i) {
        s[i] = stride;
        stride *= sizes[i];
    }
    return s;
}
static inline std::vector<int> left_pad_sizes(const std::vector<int>& v, int ndim) {
    std::vector<int> r(ndim - (int)v.size(), 1);
    r.insert(r.end(), v.begin(), v.end());
    return r;
}

Tensor sum_to_shape(const Tensor& src, const std::vector<int>& dst_shape_in) {
    const auto& sshape = src.shape();
    const int sN = (int)sshape.size();
    const int dN = (int)dst_shape_in.size();
    if (dN > sN) throw std::runtime_error("sum_to_shape: dst rank > src rank");

    // Fast path: [B,F] -> [F]
    if (sN == 2 && dN == 1 && sshape[1] == dst_shape_in[0]) {
        const int B = sshape[0];
        const int F = sshape[1];
        std::vector<float> out(F, 0.0f);
        const auto& sdata = src.data(); // assume row-major contiguous
        for (int i = 0; i < B; ++i) {
            const int base = i * F;
            for (int j = 0; j < F; ++j) {
                out[j] += sdata[base + j];
            }
        }
        return Tensor({F}, std::move(out), /*requires_grad=*/false);
    }

    // General path (contiguous src, row-major):
    // Map every src index to a dst index by clamping dims where dst==1
    // and dropping leading dims that dst doesn't have.
    std::vector<int> dst_shape = dst_shape_in;
    std::vector<int> sp = sshape;
    std::vector<int> dp = left_pad_sizes(dst_shape, sN); // align dst to src rank

    // Validate broadcasting compatibility
    for (int d = 0; d < sN; ++d) {
        if (!(sp[d] == dp[d] || dp[d] == 1)) {
            throw std::runtime_error("sum_to_shape: incompatible shapes for reduction");
        }
    }

    const int64_t src_numel = numel_of(sp);
    const int64_t dst_numel = numel_of(dst_shape);
    std::vector<float> out(dst_numel, 0.0f);

    // Precompute strides
    auto sstrides = rowmajor_strides(sp);
    auto dstrides = rowmajor_strides(dst_shape);

    // For index conversion, we’ll decode src linear -> multi-d idx, then map to dst idx.
    const auto& sdata = src.data();

    for (int64_t lin = 0; lin < src_numel; ++lin) {
        // decode src multi-index
        int64_t t = lin;
        int dst_lin = 0;
        for (int d = 0; d < sN; ++d) {
            const int dim_size = sp[d];
            const int idx_d = (int)(t / sstrides[d]) % dim_size;
            t %= sstrides[d];

            // map to dst index along aligned dim
            int use_idx;
            if (dp[d] == 1) {
                use_idx = 0;                 // reduced/broadcasted dim -> collapses to 0
            } else {
                // dp[d] == sp[d] here
                // Need to place it into the *compressed* dst index (without leading dims)
                // We'll build dst_lin incrementally using dst strides over dst_shape
                // Figure out which dst dim this corresponds to:
                const int dst_dim = d - (sN - dN); // negative => leading src dim (reduced)
                if (dst_dim < 0) {
                    // leading extra src dim -> always reduced (ignored)
                    continue;
                }
                dst_lin += use_idx * (int)dstrides[dst_dim]; // but we haven't set use_idx yet; fix below
            }
        }

        // The above approach needs a cleaner mapping. Let's redo mapping cleanly:

        // Recompute properly:
        t = lin;
        std::vector<int> src_idx(sN, 0);
        for (int d = 0; d < sN; ++d) {
            src_idx[d] = (int)(t / sstrides[d]) % sp[d];
            t %= sstrides[d];
        }
        // Build dst multi-index (length dN)
        std::vector<int> dst_idx(dN, 0);
        for (int d = 0; d < dN; ++d) {
            const int src_dim = d + (sN - dN);    // align right
            dst_idx[d] = (dp[src_dim] == 1) ? 0 : src_idx[src_dim];
        }
        // Convert dst_idx -> linear
        int64_t dl = 0;
        for (int d = 0; d < dN; ++d) {
            dl += (int64_t)dst_idx[d] * dstrides[d];
        }

        out[dl] += sdata[lin];
    }

    return Tensor(dst_shape, std::move(out), /*requires_grad=*/false);
}

} // namespace znet
