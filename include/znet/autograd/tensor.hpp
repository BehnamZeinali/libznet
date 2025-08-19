// include/znet/tensor.hpp
#pragma once
#include <vector>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <cstdint>
#include <initializer_list>

#include "znet/autograd/autograd_meta.hpp"
#include "znet/autograd/autograd.hpp"

namespace znet {

// ---- Limits ----
inline constexpr int kMaxRank = 5;

// ---- Backing storage (float32-only) ----
struct Storage {
    std::shared_ptr<std::vector<float>> data;
    Storage() : data(std::make_shared<std::vector<float>>()) {}
    explicit Storage(size_t n) : data(std::make_shared<std::vector<float>>(n)) {}
};

class TensorImpl;

class Tensor {
public:
    // ----- Lightweight row view (2D helper for convenience) -----
    class TensorSlice {
    public:
        TensorSlice(float* data_ptr, int64_t row_offset, int64_t row_stride)
            : data_ptr_(data_ptr), row_offset_(row_offset), row_stride_(row_stride) {}
        float& operator[](int64_t col) { return data_ptr_[row_offset_ + col]; }
    private:
        float*  data_ptr_;
        int64_t row_offset_;
        int64_t row_stride_;
    };

    // ----- Construction -----
    Tensor(); // default: shape {1}, value 0
    Tensor(std::vector<int64_t> shape, float val = 0.0f, bool requires_grad = false);
    Tensor(std::vector<int64_t> shape, std::vector<float> values, bool requires_grad = false);

    // tensor.hpp (inside class Tensor public:)
    // Tensor(std::initializer_list<int64_t> shape, float val = 0.0f, bool requires_grad = false);                // NEW
    // Tensor(std::initializer_list<int64_t> shape, std::vector<float> values, bool requires_grad = false);       // NEW

    // Exact-match for brace lists of plain int (forward to 64-bit)
    

    // You should also already have these:
    Tensor(std::initializer_list<int64_t> shape, float val = 0.0f, bool requires_grad = false);
    Tensor(std::initializer_list<int64_t> shape, std::vector<float> values, bool requires_grad = false);

    
    // Back-compat convenience ctors (int -> int64_t)
    Tensor(std::vector<int> shape, float val, bool requires_grad)
        : Tensor(std::vector<int64_t>(shape.begin(), shape.end()), val, requires_grad) {}
    Tensor(std::vector<int> shape, std::vector<float> values, bool requires_grad)
        : Tensor(std::vector<int64_t>(shape.begin(), shape.end()), std::move(values), requires_grad) {}

    explicit Tensor(std::shared_ptr<TensorImpl> impl) : impl_(std::move(impl)) {}

    // Factory for SavedTensor::unpack()
    static Tensor from_impl(const std::shared_ptr<TensorImpl>& p) {
        Tensor t; t.impl_ = p; return t;
    }

    // ----- Impl handle accessors -----
    std::shared_ptr<TensorImpl>& impl() { return impl_; }
    const std::shared_ptr<TensorImpl>& impl() const { return impl_; }

    // ----- Shape / data -----
    const std::vector<int64_t>& shape() const;
    const std::vector<int64_t>& stride() const;
    int64_t numel() const;

    // Raw storage access (vector reflects the whole allocation; ignores view metadata)
    const std::vector<float>& data() const;
    std::vector<float>&       data();

    // Raw pointer to storage (use with storage_offset()/strides if non-contiguous)
    const float* data_ptr() const;
    float*       data_ptr();

    // ----- Pretty print (debug) -----
    void print_shape() const;
    void print_data() const;
    void print() const;
    void print_limited(int64_t max_elems_per_dim = 8) const; // keep this utility

    // ----- Autograd flags / links -----
    bool requires_grad() const;
    void set_requires_grad(bool value);

    std::shared_ptr<Tensor>  grad() const;
    void                     set_grad(std::shared_ptr<Tensor> g);
    void                     set_grad(const Tensor& g);

    std::shared_ptr<Function> grad_fn() const;
    void                       set_grad_fn(std::shared_ptr<Function> fn);

    void backward();

    // ----- Indexing -----
    // 1-D flat indexing into logical tensor (checks via compute_flat_offset)
    float  operator()(int index) const { return (*this)(static_cast<int64_t>(index)); }
    float& operator()(int index)       { return (*this)(static_cast<int64_t>(index)); }
    float  operator()(int64_t index) const;
    float& operator()(int64_t index);

    // N-D element access (logical)
    float  at(const std::vector<int64_t>& indices) const;
    float& at(const std::vector<int64_t>& indices);
    float  at(std::initializer_list<int64_t> indices) const {
        return at(std::vector<int64_t>(indices));
    }
    float& at(std::initializer_list<int64_t> indices) {
        return at(std::vector<int64_t>(indices));
    }

    // 2D row helper (dense convenience)
    TensorSlice row_slice(int64_t row);
    const TensorSlice row_slice(int64_t row) const;

    // ----- Basic ops (front API) -----
    Tensor transpose(int dim1, int dim2) const;              // metadata swap
    Tensor materialized_transpose(int dim1, int dim2) const; // copy (2D fast path OK; N-D via view+copy)
    // Deprecated misspelling kept for compatibility (inline alias):
    Tensor materialiazed_transpose(int dim1, int dim2) const ; 

    Tensor slice(int dim, int start, int end) const;         // copy path (1D/2D convenience)

    void   add_(const Tensor& other); // in-place elementwise (same shape)
    void   add_(float scalar);
    void   mul_(const Tensor& other); // in-place elementwise (same shape)
    void   mul_(float scalar);
    Tensor mul(float scalar) const;   // out-of-place scalar multiply
    Tensor mul(const Tensor& other) const; // optional out-of-place tensor multiply (if implemented)

    // ----- Grad accumulation -----
    void accumulate_grad(const Tensor& g);

    // --- View primitives ---
    Tensor as_strided(const std::vector<int64_t>& sizes,
                      const std::vector<int64_t>& strides,
                      int64_t storage_offset) const;

    Tensor transpose_view(int dim0, int dim1) const;
    Tensor slice_view(int dim, int start, int end) const;

    // --- Low-level helpers for kernels/views ---
    int      rank() const { return static_cast<int>(shape().size()); }
    int64_t  storage_offset() const;

    // Contiguity
    bool   is_contiguous() const;
    Tensor contiguous() const;

    // Index math helpers
    int     norm_dim(int d) const;
    int64_t offset_of(const std::vector<int64_t>& indices) const;

    // Reshape / View
    Tensor reshape(const std::vector<int64_t>& new_shape) const;
    Tensor reshape(std::initializer_list<int64_t> new_shape) const {
        return reshape(std::vector<int64_t>(new_shape));
    }
    Tensor view(const std::vector<int64_t>& new_shape) const; // throws if not viewable
    Tensor view(std::initializer_list<int64_t> new_shape) const {
        return view(std::vector<int64_t>(new_shape));
    }

    // In class Tensor (public):

// 1) Scalar helpers
    bool is_scalar() const;      // rank() == 0 ?
    float item() const;          // read scalar (requires rank 0)
    float& item_ref();           // write scalar (requires rank 0)

    // 2) General select view (already added earlier)
    Tensor select(int dim, int64_t index) const;

    // 3) Python-like chaining: a[0] returns a Tensor view selecting first dim
    Tensor operator[](int64_t index) const;  // returns select(0, index)

private:
    std::shared_ptr<TensorImpl> impl_;
};

// ----- Implementation object (owned via shared_ptr) -----
class TensorImpl {
public:
    TensorImpl(std::vector<int64_t> shape, float val, bool requires_grad);
    TensorImpl(std::vector<int64_t> shape, std::vector<float> values, bool requires_grad);

    void compute_stride();

    // Backing storage + view metadata
    std::shared_ptr<Storage> storage_;
    int64_t                  storage_offset_ = 0; // in elements

    // Sizes/strides (row-major by default)
    std::vector<int64_t> shape_;
    std::vector<int64_t> stride_;

    // Autograd meta
    std::shared_ptr<AutogradMeta> autograd_ = nullptr;

    // Helpers
    static std::vector<int64_t> make_contiguous_strides(const std::vector<int64_t>& sizes);
    static int                  norm_dim(int d, int rank);
    static int64_t              compute_flat_offset(const std::vector<int64_t>& indices,
                                                    const std::vector<int64_t>& sizes,
                                                    const std::vector<int64_t>& strides,
                                                    int64_t storage_offset);
};

} // namespace znet
