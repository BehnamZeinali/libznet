#include <znet/autograd/tensor.hpp>
#include <znet/autograd/autograd_function.hpp>
#include <znet/autograd/grad_mode.hpp>

#include <algorithm>   // std::fill
#include <cstddef>     // SIZE_MAX
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace znet {

// ---------- Internal helpers (local to this TU) ----------

// Purpose: advance an N-D index in row-major order; returns false when finished.
static bool advance_index(std::vector<int64_t>& idx, const std::vector<int64_t>& sizes) {
    for (int d = static_cast<int>(sizes.size()) - 1; d >= 0; --d) {
        if (++idx[static_cast<size_t>(d)] < sizes[static_cast<size_t>(d)]) return true;
        idx[static_cast<size_t>(d)] = 0;
    }
    return false;
}

// Purpose: materialize any tensor (possibly a view) into a new contiguous tensor with same logical order.
static Tensor materialize_to_contiguous(const Tensor& t) {
    std::vector<float> out(static_cast<size_t>(t.numel()));
    // fast path if already contiguous
    if (t.is_contiguous()) {
        const float* src = t.data_ptr() + t.storage_offset();
        std::copy(src, src + t.numel(), out.begin());
        return Tensor(t.shape(), std::move(out), t.requires_grad());
    }
    // generic path for views
    std::vector<int64_t> idx(t.shape().size(), 0);
    size_t lin = 0;
    do {
        out[lin++] = t.at(idx);
    } while (advance_index(idx, t.shape()));
    return Tensor(t.shape(), std::move(out), t.requires_grad());
}

// Purpose: resolve reshape sizes, supporting at most one -1 and validating element count.
static std::vector<int64_t> resolve_reshape_sizes(const std::vector<int64_t>& new_shape, int64_t numel) {
    int neg1 = -1;
    int64_t known_prod = 1;
    for (int i = 0; i < static_cast<int>(new_shape.size()); ++i) {
        const int64_t d = new_shape[static_cast<size_t>(i)];
        if (d == -1) {
            if (neg1 != -1) throw std::invalid_argument("reshape: at most one -1 dimension");
            neg1 = i;
        } else {
            if (d <= 0) throw std::invalid_argument("reshape: dimensions must be > 0 (except one -1)");
            if (known_prod > (std::numeric_limits<int64_t>::max)() / d)
                throw std::overflow_error("reshape: product overflow");
            known_prod *= d;
        }
    }
    std::vector<int64_t> out = new_shape;
    if (neg1 == -1) {
        if (known_prod != numel) throw std::invalid_argument("reshape: element count mismatch");
    } else {
        if (numel % known_prod != 0) throw std::invalid_argument("reshape: -1 cannot be inferred (remainders)");
        out[static_cast<size_t>(neg1)] = numel / known_prod;
    }
    return out;
}

// Purpose: recursive pretty-printer that respects logical shape/strides and truncates per-dimension.
static void print_nd(std::ostream& os, const Tensor& t, int dim,
                     std::vector<int64_t>& idx, int64_t cap) {
    const auto& sizes = t.shape();
    const int64_t S = sizes[static_cast<size_t>(dim)];
    const int64_t show = std::min<int64_t>(S, cap);

    os << "[";
    if (dim + 1 == static_cast<int>(sizes.size())) {
        // leaf: print scalars
        for (int64_t i = 0; i < show; ++i) {
            idx[static_cast<size_t>(dim)] = i;
            os << t.at(idx);
            if (i + 1 < show) os << ", ";
        }
        if (show < S) os << ", ...";
    } else {
        for (int64_t i = 0; i < show; ++i) {
            idx[static_cast<size_t>(dim)] = i;
            print_nd(os, t, dim + 1, idx, cap);
            if (i + 1 < show) os << ", ";
        }
        if (show < S) os << ", ...";
    }
    os << "]";
}

// ======================= TensorImpl =======================

    // Purpose: construct a float32 tensor with 64-bit shape and fill every element with 'val'.
    TensorImpl::TensorImpl(std::vector<int64_t> shape, float val, bool requires_grad)
        : shape_(std::move(shape)) {
        const int rank = static_cast<int>(shape_.size());
        if (rank <= 0 || rank > kMaxRank)
            throw std::invalid_argument("TensorImpl: rank must be in [1, kMaxRank]");

        int64_t total = 1;
        for (int64_t dim : shape_) {
            if (dim <= 0) throw std::invalid_argument("Shape dimensions must be > 0");
            total *= dim;
        }
        if (total < 0 || static_cast<uint64_t>(total) > static_cast<uint64_t>(SIZE_MAX))
            throw std::overflow_error("TensorImpl: size overflow");

        storage_ = std::make_shared<Storage>(static_cast<size_t>(total));
        std::fill(storage_->data->begin(), storage_->data->end(), val);

        compute_stride();

    #ifndef NDEBUG
        {
            const auto expected = TensorImpl::make_contiguous_strides(shape_);
            if (stride_ != expected) {
                throw std::logic_error("TensorImpl: fresh tensor not contiguous (stride mismatch)");
            }
        }
    #endif

        if (requires_grad) {
            autograd_ = std::make_shared<AutogradMeta>();
            autograd_->requires_grad = true;
        }
    }

    // Purpose: construct a float32 tensor with 64-bit shape and move 'values' into storage.
    TensorImpl::TensorImpl(std::vector<int64_t> shape, std::vector<float> values, bool requires_grad)
        : shape_(std::move(shape)) {
        const int rank = static_cast<int>(shape_.size());
        if (rank <= 0 || rank > kMaxRank)
            throw std::invalid_argument("TensorImpl: rank must be in [1, kMaxRank]");

        int64_t total = 1;
        for (int64_t dim : shape_) {
            if (dim <= 0) throw std::invalid_argument("Shape dimensions must be > 0");
            total *= dim;
        }
        if (total != static_cast<int64_t>(values.size()))
            throw std::invalid_argument("Number of elements does not match shape");

        storage_ = std::make_shared<Storage>();
        storage_->data = std::make_shared<std::vector<float>>(std::move(values));

        compute_stride();

    #ifndef NDEBUG
        {
            const auto expected = TensorImpl::make_contiguous_strides(shape_);
            if (stride_ != expected) {
                throw std::logic_error("TensorImpl: fresh tensor not contiguous (stride mismatch)");
            }
        }
    #endif

        if (requires_grad) {
            autograd_ = std::make_shared<AutogradMeta>();
            autograd_->requires_grad = true;
        }
    }

    // Purpose: build row-major contiguous strides for 'sizes'.
    std::vector<int64_t> TensorImpl::make_contiguous_strides(const std::vector<int64_t>& sizes) {
        std::vector<int64_t> strides(sizes.size(), 0);
        int64_t s = 1;
        for (int i = static_cast<int>(sizes.size()) - 1; i >= 0; --i) {
            strides[static_cast<size_t>(i)] = s;
            s *= sizes[static_cast<size_t>(i)];
        }
        return strides;
    }

    // Purpose: compute and set row-major contiguous strides for this tensor.
    void TensorImpl::compute_stride() {
        stride_ = make_contiguous_strides(shape_);
    }

    // Purpose: normalize possibly-negative dim into [0, rank).
    int TensorImpl::norm_dim(int d, int rank) {
        if (d < 0) d += rank;
        if (d < 0 || d >= rank) {
            throw std::out_of_range("norm_dim: dim out of range after normalization");
        }
        return d;
    }

    // Purpose: compute flat storage offset (in elements) from N-D indices and strides.
    int64_t TensorImpl::compute_flat_offset(
        const std::vector<int64_t>& indices,
        const std::vector<int64_t>& sizes,
        const std::vector<int64_t>& strides,
        int64_t storage_offset) {
        const size_t rank = sizes.size();
        if (indices.size() != rank || strides.size() != rank) {
            throw std::invalid_argument("compute_flat_offset: rank mismatch");
        }

        int64_t off = storage_offset;
        for (size_t d = 0; d < rank; ++d) {
            int64_t idx = indices[d];
            const int64_t sz = sizes[d];
            if (idx < 0) idx += sz;

    #ifndef NDEBUG
            if (idx < 0 || idx >= sz) {
                throw std::out_of_range("compute_flat_offset: index out of bounds");
            }
    #endif
            off += idx * strides[d];
        }
        return off;
    }

    // ======================= Tensor (handle) =======================

    // Purpose: default-construct a 1-element tensor filled with 0.0 (no grad).
    Tensor::Tensor()
        : impl_(std::make_shared<TensorImpl>(std::vector<int64_t>{1}, 0.0f, false)) {}

    // Purpose: construct a float32 tensor and fill with 'val'.
    Tensor::Tensor(std::vector<int64_t> shape, float val, bool requires_grad)
        : impl_(std::make_shared<TensorImpl>(std::move(shape), val, requires_grad)) {}

    // Purpose: construct a float32 tensor by moving 'values' (size must match shape product).
    Tensor::Tensor(std::vector<int64_t> shape, std::vector<float> values, bool requires_grad)
        : impl_(std::make_shared<TensorImpl>(std::move(shape), std::move(values), requires_grad)) {}

    Tensor::Tensor(std::initializer_list<int64_t> shape, float val, bool requires_grad)
    : impl_(std::make_shared<TensorImpl>(std::vector<int64_t>(shape), val, requires_grad)) {}

    // Purpose: construct from {…} shape list (64-bit) and move in 'values'.
    Tensor::Tensor(std::initializer_list<int64_t> shape, std::vector<float> values, bool requires_grad)
        : impl_(std::make_shared<TensorImpl>(std::vector<int64_t>(shape), std::move(values), requires_grad)) {}

    // Purpose: construct from {…} 64-bit shape and move 'values'.
    // Tensor::Tensor(std::initializer_list<int64_t> shape, std::vector<float> values, bool requires_grad)
    //     : impl_(std::make_shared<TensorImpl>(std::vector<int64_t>(shape),
    //                                      std::move(values),
    //                                      requires_grad)) {}

   



    // Purpose: return a const reference to the logical sizes.
    const std::vector<int64_t>& Tensor::shape() const { return impl_->shape_; }

    // Purpose: return a const reference to the logical strides.
    const std::vector<int64_t>& Tensor::stride() const { return impl_->stride_; }

    // Purpose: compute the total logical number of elements (product of shape).
    int64_t Tensor::numel() const {
        int64_t total = 1;
        for (int64_t d : impl_->shape_) total *= d;
        return total;
    }

    // Purpose: print only the shape tuple.
    void Tensor::print_shape() const {
        std::cout << "Shape: (";
        for (size_t i = 0; i < impl_->shape_.size(); ++i) {
            std::cout << impl_->shape_[i];
            if (i + 1 < impl_->shape_.size()) std::cout << ", ";
        }
        std::cout << ")\n";
    }

    // Purpose: dump raw storage contents (ignores views/strides).
    void Tensor::print_data() const {
        const auto& vec = *impl_->storage_->data;
        std::cout << "Data: [";
        for (size_t i = 0; i < vec.size(); ++i) {
            std::cout << vec[i];
            if (i + 1 < vec.size()) std::cout << ", ";
        }
        std::cout << "]\n";
    }

    // Purpose: pretty-print the tensor with per-dimension truncation (respects views/strides).
    void Tensor::print_limited(int64_t max_elems_per_dim) const {
        std::cout << "Tensor: shape=(";
        for (size_t i = 0; i < impl_->shape_.size(); ++i) {
            std::cout << impl_->shape_[i];
            if (i + 1 < impl_->shape_.size()) std::cout << ", ";
        }
        std::cout << ")\n";

        const int rank = static_cast<int>(impl_->shape_.size());
        if (rank == 0) { std::cout << "[]\n"; return; }

        std::vector<int64_t> idx(impl_->shape_.size(), 0);
        print_nd(std::cout, *this, /*dim=*/0, idx, max_elems_per_dim);
        std::cout << "\n";
    }

    // Purpose: default printer using a small per-dimension cap (respects views).
    void Tensor::print() const {
        print_limited(8);
    }

    // Purpose: run reverse-mode autodiff from this tensor (seed 1s if needed).
    void Tensor::backward() {
        GradMode::NoGradGuard ng;
        if (!requires_grad()) {
            throw std::runtime_error("backward() called on tensor with requires_grad = false.");
        }

        if (!grad()) {
            auto g = std::make_shared<Tensor>(shape(),
                                            std::vector<float>(static_cast<size_t>(numel()), 1.0f),
                                            /*requires_grad=*/false);
            set_grad(std::move(g));
        }

        std::unordered_set<Tensor*> visited;
        std::vector<Tensor*> stack = { this };

        while (!stack.empty()) {
            Tensor* t = stack.back();
            stack.pop_back();

            if (!t->requires_grad() || visited.count(t)) continue;
            visited.insert(t);

            if (!t->grad()) {
                auto zero = std::make_shared<Tensor>(t->shape(),
                                std::vector<float>(static_cast<size_t>(t->numel()), 0.0f),
                                /*requires_grad=*/false);
                t->set_grad(std::move(zero));
            }

            auto fn = t->grad_fn();
            if (fn) {
                fn->backward(*t->grad());
                for (Tensor* input : fn->inputs()) {
                    if (input && input->requires_grad()) {
                        stack.push_back(input);
                    }
                }
            }
        }
    }

    // Purpose: query whether this tensor participates in autograd.
    bool Tensor::requires_grad() const {
        return impl_->autograd_ && impl_->autograd_->requires_grad;
    }

    // Purpose: enable/disable gradient tracking on this tensor.
    void Tensor::set_requires_grad(bool value) {
        if (!impl_->autograd_) impl_->autograd_ = std::make_shared<AutogradMeta>();
        impl_->autograd_->requires_grad = value;
    }

    // Purpose: get the gradient buffer tensor (may be null).
    std::shared_ptr<Tensor> Tensor::grad() const {
        return (impl_->autograd_) ? impl_->autograd_->grad : nullptr;
    }

    // Purpose: get the autograd function that produced this tensor (may be null).
    std::shared_ptr<Function> Tensor::grad_fn() const {
        return (impl_->autograd_) ? impl_->autograd_->grad_fn : nullptr;
    }

    // Purpose: set the gradient buffer tensor.
    void Tensor::set_grad(std::shared_ptr<Tensor> g) {
        if (!impl_->autograd_) impl_->autograd_ = std::make_shared<AutogradMeta>();
        impl_->autograd_->grad = std::move(g);
    }

    // Purpose: convenience setter for gradient buffer (copy handle).
    void Tensor::set_grad(const Tensor& g) {
        set_grad(std::make_shared<Tensor>(g));
    }

    // Purpose: set the autograd function that produced this tensor.
    void Tensor::set_grad_fn(std::shared_ptr<Function> fn) {
        if (!impl_->autograd_) impl_->autograd_ = std::make_shared<AutogradMeta>();
        impl_->autograd_->grad_fn = std::move(fn);
    }

    // Purpose: flat index read on a 1-D tensor (throws if rank != 1).
    float Tensor::operator()(int64_t index) const {
        return at(std::vector<int64_t>{index});
    }

    // Purpose: flat index write on a 1-D tensor (throws if rank != 1).
    float& Tensor::operator()(int64_t index) {
        return at(std::vector<int64_t>{index});
    }

    // Purpose: materialize a transpose (copy) by making a view then copying to contiguous.
    Tensor Tensor::materialized_transpose(int dim1, int dim2) const {
        const int rank = static_cast<int>(impl_->shape_.size());
        if (rank < 2) throw std::runtime_error("transpose: rank < 2");

        dim1 = TensorImpl::norm_dim(dim1, rank);
        dim2 = TensorImpl::norm_dim(dim2, rank);
        if (dim1 == dim2) return *this;

        Tensor v = transpose_view(dim1, dim2);
        return materialize_to_contiguous(v);
    }

    // Purpose: backward-compat alias for the earlier misspelled name.
    Tensor Tensor::materialiazed_transpose(int dim1, int dim2) const {
        return materialized_transpose(dim1, dim2);
    }

    // Purpose: return a metadata-only transpose view (no copy).
    Tensor Tensor::transpose(int dim1, int dim2) const {
        return transpose_view(dim1, dim2);
    }

    // Purpose: return a materialized slice along 'dim' with range [start, end).
    Tensor Tensor::slice(int dim, int start, int end) const {
        const int rank = static_cast<int>(impl_->shape_.size());
        dim = TensorImpl::norm_dim(dim, rank);

        const int64_t size_d = impl_->shape_[static_cast<size_t>(dim)];
        const int64_t s = static_cast<int64_t>(start);
        const int64_t e = static_cast<int64_t>(end);
        if (s < 0 || e < 0 || s >= e || e > size_d) {
            throw std::runtime_error("slice: invalid [start, end)");
        }

        Tensor v = slice_view(dim, start, end);
        return materialize_to_contiguous(v);
    }

    // Purpose: row helper for dense, contiguous rank-2 tensors (mutable).
    Tensor::TensorSlice Tensor::row_slice(int64_t row) {
        if (impl_->shape_.size() != 2) {
            throw std::runtime_error("operator[]: only valid for rank-2 tensors");
        }
        if (!is_contiguous()) {
            throw std::runtime_error("operator[]: only valid for contiguous row-major tensors");
        }

        const int64_t rows = impl_->shape_[0];
        const int64_t cols = impl_->shape_[1];
        if (row < 0 || row >= rows) throw std::out_of_range("Row index out of bounds");

        float* base = data_ptr();
        const int64_t off = storage_offset();
        const int64_t row_offset = off + row * cols;

        return TensorSlice(base, row_offset, cols);
    }

    // Purpose: row helper for dense, contiguous rank-2 tensors (const).
    const Tensor::TensorSlice Tensor::row_slice(int64_t row) const {
        if (impl_->shape_.size() != 2) {
            throw std::runtime_error("operator[]: only valid for rank-2 tensors");
        }
        if (!is_contiguous()) {
            throw std::runtime_error("operator[]: only valid for contiguous row-major tensors");
        }

        const int64_t rows = impl_->shape_[0];
        const int64_t cols = impl_->shape_[1];
        if (row < 0 || row >= rows) throw std::out_of_range("Row index out of bounds");

        float* base = const_cast<float*>(data_ptr());
        const int64_t off = storage_offset();
        const int64_t row_offset = off + row * cols;

        return TensorSlice(base, row_offset, cols);
    }

    // Purpose: in-place elementwise add; requires same shape (no broadcasting).
    void Tensor::add_(const Tensor& other) {
        if (shape() != other.shape())
            throw std::runtime_error("add_: shapes must match for in-place addition");

        const int64_t n = numel();

        if (is_contiguous() && other.is_contiguous()) {
            float*       d = data_ptr() + storage_offset();
            const float* s = other.data_ptr() + other.storage_offset();
            for (int64_t i = 0; i < n; ++i) d[i] += s[i];
            return;
        }

        auto&       dst = *impl_->storage_->data;
        const auto& src = *other.impl_->storage_->data;
        std::vector<int64_t> idx(shape().size(), 0);

        do {
            const int64_t doff = offset_of(idx);
            const int64_t soff = other.offset_of(idx);
            dst[static_cast<size_t>(doff)] += src[static_cast<size_t>(soff)];
        } while (advance_index(idx, shape()));
    }

    // Purpose: in-place scalar add on all elements.
    void Tensor::add_(float scalar) {
        const int64_t n = numel();
        if (is_contiguous()) {
            float* d = data_ptr() + storage_offset();
            for (int64_t i = 0; i < n; ++i) d[i] += scalar;
            return;
        }

        auto& dst = *impl_->storage_->data;
        std::vector<int64_t> idx(shape().size(), 0);
        do {
            const int64_t off = offset_of(idx);
            dst[static_cast<size_t>(off)] += scalar;
        } while (advance_index(idx, shape()));
    }

    // Purpose: in-place elementwise multiply; requires same shape (no broadcasting).
    void Tensor::mul_(const Tensor& other) {
        if (shape() != other.shape())
            throw std::runtime_error("mul_: shapes must match");

        const int64_t n = numel();
        if (is_contiguous() && other.is_contiguous()) {
            float*       d = data_ptr() + storage_offset();
            const float* s = other.data_ptr() + other.storage_offset();
            for (int64_t i = 0; i < n; ++i) d[i] *= s[i];
            return;
        }

        auto&       dst = *impl_->storage_->data;
        const auto& src = *other.impl_->storage_->data;
        std::vector<int64_t> idx(shape().size(), 0);

        do {
            const int64_t doff = offset_of(idx);
            const int64_t soff = other.offset_of(idx);
            dst[static_cast<size_t>(doff)] *= src[static_cast<size_t>(soff)];
        } while (advance_index(idx, shape()));
    }

    // Purpose: in-place scalar multiply on all elements.
    void Tensor::mul_(float scalar) {
        const int64_t n = numel();
        if (is_contiguous()) {
            float* d = data_ptr() + storage_offset();
            for (int64_t i = 0; i < n; ++i) d[i] *= scalar;
            return;
        }

        auto& dst = *impl_->storage_->data;
        std::vector<int64_t> idx(shape().size(), 0);
        do {
            const int64_t off = offset_of(idx);
            dst[static_cast<size_t>(off)] *= scalar;
        } while (advance_index(idx, shape()));
    }

    // Purpose: out-of-place scalar multiply; returns a new contiguous tensor.
    Tensor Tensor::mul(float scalar) const {
        std::vector<float> result(static_cast<size_t>(numel()));
        if (is_contiguous()) {
            const float* base = data_ptr() + storage_offset();
            for (int64_t i = 0; i < numel(); ++i) result[static_cast<size_t>(i)] = base[static_cast<size_t>(i)] * scalar;
        } else {
            std::vector<int64_t> idx(shape().size(), 0);
            size_t i = 0;
            do {
                result[i++] = at(idx) * scalar;
            } while (advance_index(idx, shape()));
        }
        return Tensor(shape(), std::move(result), requires_grad());
    }

    // Purpose: accumulate gradients into this tensor's grad buffer ( += g ), shape must match.
    void Tensor::accumulate_grad(const Tensor& g) {
        if (!requires_grad()) return;

        if (!grad()) {
            auto zero = std::make_shared<Tensor>(shape(),
                        std::vector<float>(static_cast<size_t>(numel()), 0.0f),
                        /*requires_grad=*/false);
            set_grad(std::move(zero));
        }

        if (shape() != g.shape())
            throw std::runtime_error("Gradient shape mismatch in accumulate_grad.");

        const int64_t n = numel();

        if (grad()->is_contiguous() && g.is_contiguous()) {
            float*       d = grad()->data_ptr() + grad()->storage_offset();
            const float* s = g.data_ptr()     + g.storage_offset();
            for (int64_t i = 0; i < n; ++i) d[i] += s[i];
            return;
        }

        auto&       dst = grad()->data();
        const auto& src = g.data();
        std::vector<int64_t> idx(shape().size(), 0);
        do {
            const int64_t doff = offset_of(idx);
            const int64_t soff = g.offset_of(idx);
            dst[static_cast<size_t>(doff)] += src[static_cast<size_t>(soff)];
        } while (advance_index(idx, shape()));
    }

    // Purpose: create a metadata-only view with custom sizes/strides and extra storage_offset (elements).
    // Tensor Tensor::as_strided(const std::vector<int64_t>& sizes,
    //                         const std::vector<int64_t>& strides,
    //                         int64_t storage_offset) const {
    //     const size_t rank = sizes.size();
    //     if (rank == 0 || rank > static_cast<size_t>(kMaxRank)) {
    //         throw std::runtime_error("as_strided: rank must be in [1, kMaxRank]");
    //     }
    //     if (strides.size() != rank) {
    //         throw std::runtime_error("as_strided: sizes/strides rank mismatch");
    //     }
    //     for (size_t i = 0; i < rank; ++i) {
    //         if (sizes[i] <= 0) {
    //             throw std::runtime_error("as_strided: sizes must be > 0");
    //         }
    //         if (strides[i] < 0) {
    //             throw std::runtime_error("as_strided: negative strides not supported yet");
    //         }
    //     }
    //     if (storage_offset < 0) {
    //         throw std::runtime_error("as_strided: storage_offset must be >= 0");
    //     }

    // #ifndef NDEBUG
    //     {
    //         const int64_t base_off = impl_->storage_offset_ + storage_offset;
    //         int64_t furthest = base_off;
    //         for (size_t d = 0; d < rank; ++d) {
    //             const int64_t step = (sizes[d] - 1) * strides[d];
    //             furthest += step;
    //         }
    //         const int64_t storage_size = static_cast<int64_t>(impl_->storage_->data->size());
    //         if (furthest < 0 || furthest >= storage_size) {
    //             throw std::out_of_range("as_strided: view exceeds underlying storage");
    //         }
    //     }
    // #endif

    //     auto new_impl = std::make_shared<TensorImpl>(std::vector<int64_t>{1}, 0.0f, /*requires_grad=*/false);
    //     new_impl->storage_        = impl_->storage_;
    //     new_impl->storage_offset_ = impl_->storage_offset_ + storage_offset;
    //     new_impl->shape_          = sizes;
    //     new_impl->stride_         = strides;

    //     new_impl->autograd_ = std::make_shared<AutogradMeta>();
    //     new_impl->autograd_->requires_grad = this->requires_grad();

    //     return Tensor(std::move(new_impl));
    // }

    Tensor Tensor::as_strided(const std::vector<int64_t>& sizes,
                            const std::vector<int64_t>& strides,
                            int64_t storage_offset) const
    {
        const size_t rank = sizes.size();
        // BEFORE: if (rank == 0 || rank > kMaxRank) throw ...
        if (rank > static_cast<size_t>(kMaxRank)) {
            throw std::runtime_error("as_strided: rank must be <= kMaxRank");
        }
        if (strides.size() != rank) {
            throw std::runtime_error("as_strided: sizes/strides rank mismatch");
        }
        for (size_t i = 0; i < rank; ++i) {
            if (sizes[i] <= 0) {
                throw std::runtime_error("as_strided: sizes must be > 0");
            }
            if (strides[i] < 0) {
                throw std::runtime_error("as_strided: negative strides not supported yet");
            }
        }
        if (storage_offset < 0) {
            throw std::runtime_error("as_strided: storage_offset must be >= 0");
        }

    #ifndef NDEBUG
        {
            const int64_t base_off = impl_->storage_offset_ + storage_offset;
            int64_t furthest = base_off;
            // For rank==0, this loop won’t run; furthest == base_off.
            for (size_t d = 0; d < rank; ++d) {
                furthest += (sizes[d] - 1) * strides[d];
            }
            const int64_t storage_size = static_cast<int64_t>(impl_->storage_->data->size());
            if (furthest < 0 || furthest >= storage_size) {
                throw std::out_of_range("as_strided: view exceeds underlying storage");
            }
        }
    #endif

        auto new_impl = std::make_shared<TensorImpl>(std::vector<int64_t>{1}, 0.0f, false);
        new_impl->storage_        = impl_->storage_;
        new_impl->storage_offset_ = impl_->storage_offset_ + storage_offset;
        new_impl->shape_          = sizes;     // may be empty for scalar
        new_impl->stride_         = strides;   // may be empty for scalar
        new_impl->autograd_       = std::make_shared<AutogradMeta>();
        new_impl->autograd_->requires_grad = this->requires_grad();
        return Tensor(std::move(new_impl));
    }


    // Purpose: return a metadata-only transpose view (swap two dims).
    Tensor Tensor::transpose_view(int dim0, int dim1) const {
        const int rank = static_cast<int>(impl_->shape_.size());
        dim0 = TensorImpl::norm_dim(dim0, rank);
        dim1 = TensorImpl::norm_dim(dim1, rank);
        if (dim0 == dim1) return *this;

        auto sizes   = impl_->shape_;
        auto strides = impl_->stride_;
        std::swap(sizes[static_cast<size_t>(dim0)],   sizes[static_cast<size_t>(dim1)]);
        std::swap(strides[static_cast<size_t>(dim0)], strides[static_cast<size_t>(dim1)]);

        return as_strided(sizes, strides, /*storage_offset=*/0);
    }

    // Purpose: return a metadata-only slice view along 'dim' with range [start, end).
    Tensor Tensor::slice_view(int dim, int start, int end) const {
        const int rank = static_cast<int>(impl_->shape_.size());
        dim = TensorImpl::norm_dim(dim, rank);

        const int64_t size_d = impl_->shape_[static_cast<size_t>(dim)];
        int64_t s = static_cast<int64_t>(start);
        int64_t e = static_cast<int64_t>(end);
        if (s < 0 || e < 0 || s >= e || e > size_d) {
            throw std::runtime_error("slice_view: invalid [start, end)");
        }

        auto sizes   = impl_->shape_;
        auto strides = impl_->stride_;
        sizes[static_cast<size_t>(dim)] = e - s;
        const int64_t delta = s * strides[static_cast<size_t>(dim)];

        return as_strided(sizes, strides, /*storage_offset=*/delta);
    }

    // Purpose: return the starting storage offset (in elements) for this view.
    int64_t Tensor::storage_offset() const {
        return impl_->storage_offset_;
    }

    // Purpose: return a const pointer to the base of storage (ignores view offset).
    const float* Tensor::data_ptr() const {
        return impl_->storage_->data->data();
    }

    // Purpose: return a pointer to the base of storage (ignores view offset).
    float* Tensor::data_ptr() {
        return impl_->storage_->data->data();
    }

    // Purpose: return a const reference to the whole backing vector (ignores view metadata).
    const std::vector<float>& Tensor::data() const { return *impl_->storage_->data; }

    // Purpose: return a reference to the whole backing vector (ignores view metadata).
    std::vector<float>& Tensor::data() { return *impl_->storage_->data; }

    // Purpose: check if tensor is row-major contiguous (stride equals canonical strides).
    bool Tensor::is_contiguous() const {
        return impl_->stride_ == TensorImpl::make_contiguous_strides(impl_->shape_);
    }

    // Purpose: normalize a possibly-negative dim using this tensor's rank.
    int Tensor::norm_dim(int d) const {
        return TensorImpl::norm_dim(d, static_cast<int>(impl_->shape_.size()));
    }

    // Purpose: compute flat storage offset for this tensor's sizes/strides/offset.
    int64_t Tensor::offset_of(const std::vector<int64_t>& indices) const {
        return TensorImpl::compute_flat_offset(indices, impl_->shape_, impl_->stride_, impl_->storage_offset_);
    }

    // Purpose: N-D element read respecting this tensor's view/strides.
    float Tensor::at(const std::vector<int64_t>& indices) const {
        const int64_t off = TensorImpl::compute_flat_offset(indices, impl_->shape_, impl_->stride_, impl_->storage_offset_);
        return (*impl_->storage_->data)[static_cast<size_t>(off)];
    }

    // Purpose: N-D element write respecting this tensor's view/strides.
    float& Tensor::at(const std::vector<int64_t>& indices) {
        const int64_t off = TensorImpl::compute_flat_offset(indices, impl_->shape_, impl_->stride_, impl_->storage_offset_);
        return (*impl_->storage_->data)[static_cast<size_t>(off)];
    }

    // Purpose: return a contiguous copy of this tensor (materialize if needed).
    Tensor Tensor::contiguous() const {
        return materialize_to_contiguous(*this);
    }

    // Purpose: reshape to 'new_shape' (supports one -1); view if contiguous, else copy then view.
    Tensor Tensor::reshape(const std::vector<int64_t>& new_shape) const {
        const int64_t n = numel();
        auto target = resolve_reshape_sizes(new_shape, n);

        if (is_contiguous()) {
            auto new_strides = TensorImpl::make_contiguous_strides(target);
            return as_strided(target, new_strides, /*storage_offset=*/0);
        }

        Tensor c = contiguous();
        auto new_strides = TensorImpl::make_contiguous_strides(target);
        return c.as_strided(target, new_strides, /*storage_offset=*/0);
    }

    // Purpose: build a pure view with 'new_shape' (supports one -1); requires this tensor be contiguous.
    Tensor Tensor::view(const std::vector<int64_t>& new_shape) const {
        const int64_t n = numel();
        auto target = resolve_reshape_sizes(new_shape, n);

        if (!is_contiguous()) {
            throw std::runtime_error("view: tensor is not contiguous; call .contiguous() first or use reshape()");
        }

        auto new_strides = TensorImpl::make_contiguous_strides(target);
        return as_strided(target, new_strides, /*storage_offset=*/0);
    }

    // Purpose: return a metadata-only view that selects `index` along `dim` and drops that dimension.
    Tensor Tensor::select(int dim, int64_t index) const {
        const int rank = static_cast<int>(impl_->shape_.size());
        if (rank <= 0) throw std::runtime_error("select: rank==0 tensor cannot be indexed");

        dim = TensorImpl::norm_dim(dim, rank);

        int64_t size_d = impl_->shape_[static_cast<size_t>(dim)];
        if (index < 0) index += size_d;
        if (index < 0 || index >= size_d) throw std::out_of_range("select: index OOB");

        auto sizes   = impl_->shape_;
        auto strides = impl_->stride_;
        const int64_t delta = index * strides[static_cast<size_t>(dim)];

        sizes.erase(sizes.begin()   + dim);   // may produce rank 0
        strides.erase(strides.begin()+ dim);

        return as_strided(sizes, strides, /*storage_offset=*/delta);
    }

    // Purpose: Python-like a[i] → select first dimension (returns a view).
    Tensor Tensor::operator[](int64_t index) const {
        return select(0, index);
    }

    // Purpose: check if tensor is rank-0 (scalar view).
    bool Tensor::is_scalar() const {
        return impl_->shape_.empty();
    }

    // Purpose: read scalar value (requires rank-0).
    float Tensor::item() const {
    if (numel() != 1) {
        throw std::runtime_error("item(): tensor has more than one element");
    }
    if (is_scalar()) {
        const int64_t off = impl_->storage_offset_;
        return (*impl_->storage_->data)[static_cast<size_t>(off)];
    }
    // Non-scalar single-element: offset of all zeros
    std::vector<int64_t> zeros(impl_->shape_.size(), 0);
    const int64_t off = TensorImpl::compute_flat_offset(zeros, impl_->shape_, impl_->stride_, impl_->storage_offset_);
    return (*impl_->storage_->data)[static_cast<size_t>(off)];
}

    // Purpose: write scalar value; accepts rank-0 OR any tensor with numel()==1.
    float& Tensor::item_ref() {
        if (numel() != 1) {
            throw std::runtime_error("item_ref(): tensor has more than one element");
        }
        if (is_scalar()) {
            const int64_t off = impl_->storage_offset_;
            return (*impl_->storage_->data)[static_cast<size_t>(off)];
        }
        std::vector<int64_t> zeros(impl_->shape_.size(), 0);
        const int64_t off = TensorImpl::compute_flat_offset(zeros, impl_->shape_, impl_->stride_, impl_->storage_offset_);
        return (*impl_->storage_->data)[static_cast<size_t>(off)];
    }



    } // namespace znet



// ================= Pretty printing (Step 7) =================
// namespace {  // anonymous

//     // Print indices like (2,3,4) in debug lines if you ever need it (not used below)
//     // static void print_indices(std::ostream& os, const std::vector<int64_t>& idx) { ... }

//     // Recursively print slices.
//     // - `dim`     : current dimension we are printing
//     // - `idx`     : mutable index vector reused during recursion
//     // - `limit`   : max elements to print per dimension (truncate with "...")
//     // Uses Tensor::at(...) so it works for any strides/offset. This is for DEBUG,
//     // so perf is less important than correctness/readability.
//     static void print_nd(std::ostream& os,
//                         const znet::Tensor& t,
//                         int dim,
//                         std::vector<int64_t>& idx,
//                         int64_t limit) {                            // NEW (Step 7)
//         const auto& sizes = t.shape();
//         const int rank = static_cast<int>(sizes.size());

//         if (rank == 0) { os << "[]"; return; }                      // defensive (shouldn’t happen)

//         if (dim == rank - 1) {
//             // Leaf: print a 1-D row
//             os << "[";
//             const int64_t len = sizes[static_cast<size_t>(dim)];
//             const int64_t shown = std::min(len, limit);
//             for (int64_t j = 0; j < shown; ++j) {
//                 idx[static_cast<size_t>(dim)] = j;
//                 os << t.at(idx);
//                 if (j + 1 < shown) os << ", ";
//             }
//             if (len > shown) os << ", ...";
//             os << "]";
//             return;
//         }

//         // Non-leaf: print a list of sub-slices
//         os << "[";
//         const int64_t len = sizes[static_cast<size_t>(dim)];
//         const int64_t shown = std::min(len, limit);
//         for (int64_t i = 0; i < shown; ++i) {
//             idx[static_cast<size_t>(dim)] = i;
//             print_nd(os, t, dim + 1, idx, limit);
//             if (i + 1 < shown) os << ",\n ";   // newline+space between blocks for readability
//         }
//         if (len > shown) os << ",\n ...";
//         os << "]";
//     }

//     // Increment a multi-index in row-major order: idx over sizes.
//     // Returns false when we've wrapped past the last element.
//     bool advance_index(std::vector<int64_t>& idx,
//                     const std::vector<int64_t>& sizes) {                   // NEW (Step 8)
//         for (int d = static_cast<int>(sizes.size()) - 1; d >= 0; --d) {
//             ++idx[static_cast<size_t>(d)];
//             if (idx[static_cast<size_t>(d)] < sizes[static_cast<size_t>(d)]) {
//                 return true;  // advanced this dim successfully
//             }
//             idx[static_cast<size_t>(d)] = 0; // carry into next dim
//         }
//         return false; // finished all elements
//     }


//     // Materialize ANY tensor (possibly non-contiguous) into a new contiguous Tensor
//     // with the SAME logical shape. Uses logical index traversal so works for 1–5D.
//     // NEW (Step 9)
//     znet::Tensor materialize_to_contiguous(const znet::Tensor& src) {
//         const auto& sizes = src.shape();
//         const int64_t n = src.numel();

//         // Allocate a flat buffer for the result
//         std::vector<float> out(static_cast<size_t>(n), 0.0f);

//         // Walk logical indices in row-major order and copy
//         std::vector<int64_t> idx(sizes.size(), 0);
//         int64_t linear = 0;

//         if (sizes.empty()) {
//             return znet::Tensor(std::vector<int64_t>{}, std::move(out), src.requires_grad());
//         }

//         // First element
//         out[static_cast<size_t>(linear++)] = src.at(idx);

//         while (advance_index(idx, sizes)) {
//             out[static_cast<size_t>(linear++)] = src.at(idx);
//         }

//         return znet::Tensor(sizes, std::move(out), /*requires_grad=*/src.requires_grad());
//     }

//     // NEW (Step 13)
//     std::vector<int64_t> resolve_reshape_sizes(const std::vector<int64_t>& req,
//                                             int64_t numel) {
//         int negatives = 0;
//         int64_t known_prod = 1;
//         for (int64_t s : req) {
//             if (s == -1) { ++negatives; continue; }
//             if (s <= 0) throw std::invalid_argument("reshape: sizes must be >0 or -1");
//             known_prod *= s;
//         }
//         if (negatives > 1) throw std::invalid_argument("reshape: at most one -1 is allowed");

//         std::vector<int64_t> out = req;
//         if (negatives == 1) {
//             if (numel % known_prod != 0)
//                 throw std::invalid_argument("reshape: -1 dimension not divisible by numel/known_prod");
//             int64_t inferred = numel / known_prod;
//             for (auto& s : out) if (s == -1) { s = inferred; break; }
//         } else {
//             if (known_prod != numel)
//                 throw std::invalid_argument("reshape: element count must stay the same");
//         }
//         return out;
//     }

// } // anonymous namespace


// What you’ve completed (5-D milestone recap)

// 64-bit shapes/strides/offsets (kMaxRank = 5).

// Solid contiguous stride generator + is_contiguous().

// Canonical dim/index & single offset rule.

// One true view primitive: as_strided (with bounds checks in debug).

// N-D transpose_view and slice_view built on as_strided.

// N-D scalar access at({...}) (works for any view).

// Rank-agnostic pretty-printer (honors strides).

// Elementwise ops fixed for views (generic index path + contiguous fast path).

// N-D materialized_transpose (view → copy).

// N-D slice (copying) built on slice_view.

// reshape (view if contiguous, else copy) with -1 inference.

// view (strict view-only; throws if not possible).