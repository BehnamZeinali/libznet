#include <znet/autograd/tensor.hpp>
#include <znet/autograd/autograd_function.hpp>
#include <znet/autograd/grad_mode.hpp>
#include <unordered_set>
#include <stdexcept>
#include <iostream>

namespace znet {

// ================= TensorImpl =================

TensorImpl::TensorImpl(std::vector<int> shape, float val, bool requires_grad)
    : shape_(std::move(shape)) {
    int total = 1;
    for (int dim : shape_) {
        if (dim <= 0) throw std::invalid_argument("Shape dimensions must be > 0");
        total *= dim;
    }
    // allocate storage and fill with val
    storage_ = std::make_shared<Storage>(static_cast<size_t>(total));
    std::fill(storage_->data->begin(), storage_->data->end(), val);

    compute_stride();
    if (requires_grad) {
        autograd_ = std::make_shared<AutogradMeta>();
        autograd_->requires_grad = true;
    }
}

TensorImpl::TensorImpl(std::vector<int> shape, std::vector<float> values, bool requires_grad)
    : shape_(std::move(shape)) {
    int total = 1;
    for (int dim : shape_) {
        if (dim <= 0) throw std::invalid_argument("Shape dimensions must be > 0");
        total *= dim;
    }
    if (total != static_cast<int>(values.size())) {
        throw std::invalid_argument("Number of elements does not match shape");
    }
    // move values into storage
    storage_ = std::make_shared<Storage>();
    storage_->data = std::make_shared<std::vector<float>>(std::move(values));

    compute_stride();
    if (requires_grad) {
        autograd_ = std::make_shared<AutogradMeta>();
        autograd_->requires_grad = true;
    }
}

void TensorImpl::compute_stride() {
    stride_.resize(shape_.size());
    int stride = 1;
    for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
        stride_[i] = stride;
        stride *= shape_[i];
    }
}

// ================= Tensor (handle) =================

Tensor::Tensor() : impl_(std::make_shared<TensorImpl>(std::vector<int>{1}, 0.0f, false)) {}

Tensor::Tensor(std::vector<int> shape, float val, bool requires_grad)
    : impl_(std::make_shared<TensorImpl>(std::move(shape), val, requires_grad)) {}

Tensor::Tensor(std::vector<int> shape, std::vector<float> values, bool requires_grad)
    : impl_(std::make_shared<TensorImpl>(std::move(shape), std::move(values), requires_grad)) {}

const std::vector<int>& Tensor::shape() const { return impl_->shape_; }
const std::vector<int>& Tensor::stride() const { return impl_->stride_; }

// Use logical element count from shape (safer once we have views)
int Tensor::numel() const {
    int total = 1;
    for (int d : impl_->shape_) total *= d;
    return total;
}

void Tensor::print_shape() const {
    std::cout << "Shape: (";
    for (size_t i = 0; i < impl_->shape_.size(); ++i) {
        std::cout << impl_->shape_[i];
        if (i + 1 < impl_->shape_.size()) std::cout << ", ";
    }
    std::cout << ")\n";
}

void Tensor::print_data() const {
    const auto& vec = *impl_->storage_->data;
    std::cout << "Data: [";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << vec[i];
        if (i + 1 < vec.size()) std::cout << ", ";
    }
    std::cout << "]\n";
}

void Tensor::print() const {
    std::cout << "Tensor: shape=(";
    for (size_t i = 0; i < impl_->shape_.size(); ++i) {
        std::cout << impl_->shape_[i];
        if (i + 1 < impl_->shape_.size()) std::cout << ", ";
    }
    std::cout << ")\n";

    const auto& vec = *impl_->storage_->data;
    if (impl_->shape_.size() == 1) {
        std::cout << "[";
        for (size_t i = 0; i < vec.size(); ++i) {
            std::cout << vec[i];
            if (i + 1 < vec.size()) std::cout << ", ";
        }
        std::cout << "]\n";
    } else if (impl_->shape_.size() == 2) {
        int rows = impl_->shape_[0];
        int cols = impl_->shape_[1];
        for (int i = 0; i < rows; ++i) {
            std::cout << "[";
            for (int j = 0; j < cols; ++j) {
                std::cout << vec[i * cols + j];
                if (j + 1 < cols) std::cout << ", ";
            }
            std::cout << "]\n";
        }
    } else {
        std::cout << "Higher-dimensional tensor printing not yet implemented.\n";
        print_data();
    }
}

void Tensor::backward() {
    GradMode::NoGradGuard ng;
    if (!requires_grad()) {
        throw std::runtime_error("backward() called on tensor with requires_grad = false.");
    }

    // Seed grad at root: ones
    if (!grad()) {
        auto g = std::make_shared<Tensor>(shape(), std::vector<float>(numel(), 1.0f), /*requires_grad=*/false);
        set_grad(std::move(g));
    }

    std::unordered_set<Tensor*> visited;
    std::vector<Tensor*> stack = { this };

    while (!stack.empty()) {
        Tensor* t = stack.back();
        stack.pop_back();

        if (!t->requires_grad() || visited.count(t)) continue;
        visited.insert(t);

        // Ensure this tensor has a grad buffer
        if (!t->grad()) {
            auto zero = std::make_shared<Tensor>(t->shape(), std::vector<float>(t->numel(), 0.0f), /*requires_grad=*/false);
            t->set_grad(std::move(zero));
        }

        auto fn = t->grad_fn();
        if (fn) {
            // std::cout << "Grad fn: " << fn->name() << "\n";
            fn->backward(*t->grad());

            for (Tensor* input : fn->inputs()) {
                if (input && input->requires_grad()) {
                    stack.push_back(input);
                }
            }
        }
    }
}

bool Tensor::requires_grad() const {
    return impl_->autograd_ && impl_->autograd_->requires_grad;
}

void Tensor::set_requires_grad(bool value) {
    if (!impl_->autograd_) impl_->autograd_ = std::make_shared<AutogradMeta>();
    impl_->autograd_->requires_grad = value;
}

std::shared_ptr<Tensor> Tensor::grad() const {
    return (impl_->autograd_) ? impl_->autograd_->grad : nullptr;
}

std::shared_ptr<Function> Tensor::grad_fn() const {
    return (impl_->autograd_) ? impl_->autograd_->grad_fn : nullptr;
}

// Set grad buffer (Tensor overloads)
void Tensor::set_grad(std::shared_ptr<Tensor> g) {
    if (!impl_->autograd_) impl_->autograd_ = std::make_shared<AutogradMeta>();
    impl_->autograd_->grad = std::move(g);
}

void Tensor::set_grad(const Tensor& g) {
    set_grad(std::make_shared<Tensor>(g));
}

void Tensor::set_grad_fn(std::shared_ptr<Function> fn) {
    if (!impl_->autograd_) impl_->autograd_ = std::make_shared<AutogradMeta>();
    impl_->autograd_->grad_fn = std::move(fn);
}

float Tensor::operator()(int index) const {
    return (*impl_->storage_->data)[impl_->storage_offset_ + index];
}
float& Tensor::operator()(int index) {
    return (*impl_->storage_->data)[impl_->storage_offset_ + index];
}

const std::vector<float>& Tensor::data() const { return *impl_->storage_->data; }
std::vector<float>& Tensor::data() { return *impl_->storage_->data; }

// Materializing transpose (kept as-is for now)
Tensor Tensor::transpose(int dim1, int dim2) const {
    const auto& s = impl_->shape_;
    if (dim1 == dim2) return *this;
    if (s.size() < 2) throw std::runtime_error("transpose: rank < 2");

    // Only handle 2D for now
    if (s.size() != 2) throw std::runtime_error("transpose: only 2D supported here");

    int rows = s[0], cols = s[1];
    std::vector<float> out(rows * cols);
    const auto& base = *impl_->storage_->data;
    // materialize: [rows, cols] -> [cols, rows]
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            out[j * rows + i] = base[i * cols + j];

    return Tensor({cols, rows}, std::move(out), /*requires_grad=*/requires_grad());
}

Tensor Tensor::slice(int dim, int start, int end) const {
    if (start < 0 || end > impl_->shape_[dim] || start >= end) {
        throw std::runtime_error("slice: invalid slice range");
    }

    const auto& base = *impl_->storage_->data;

    if (impl_->shape_.size() == 1) {
        std::vector<float> sliced_data(base.begin() + start, base.begin() + end);
        return Tensor({end - start}, sliced_data, requires_grad());
    }

    if (impl_->shape_.size() == 2) {
        int rows = impl_->shape_[0];
        int cols = impl_->shape_[1];
        std::vector<float> sliced_data;

        if (dim == 0) {
            for (int i = start; i < end; ++i)
                for (int j = 0; j < cols; ++j)
                    sliced_data.push_back(base[i * cols + j]);
            return Tensor({end - start, cols}, sliced_data, requires_grad());
        } else if (dim == 1) {
            for (int i = 0; i < rows; ++i)
                for (int j = start; j < end; ++j)
                    sliced_data.push_back(base[i * cols + j]);
            return Tensor({rows, end - start}, sliced_data, requires_grad());
        } else {
            throw std::runtime_error("slice: invalid dimension for 2D tensor");
        }
    }

    throw std::runtime_error("slice: only supports 1D and 2D tensors");
}

Tensor::TensorSlice Tensor::operator[](int row) {
    if (impl_->shape_.size() != 2) throw std::runtime_error("Only 2D indexing is supported for now.");
    int stride = impl_->shape_[1];
    if (row < 0 || row >= impl_->shape_[0]) throw std::out_of_range("Row index out of bounds");
    return TensorSlice(impl_->storage_->data->data(), row * stride, stride);
}

const Tensor::TensorSlice Tensor::operator[](int row) const {
    if (impl_->shape_.size() != 2) throw std::runtime_error("Only 2D indexing is supported for now.");
    int stride = impl_->shape_[1];
    if (row < 0 || row >= impl_->shape_[0]) throw std::out_of_range("Row index out of bounds");
    return TensorSlice(const_cast<float*>(impl_->storage_->data->data()), row * stride, stride);
}

void Tensor::add_(const Tensor& other) {
    if (shape() != other.shape())
        throw std::runtime_error("add_: shapes must match for in-place addition");
    auto& dst = *impl_->storage_->data;
    const auto& src = other.data();
    for (int i = 0; i < numel(); ++i)
        dst[i] += src[i];
}

void Tensor::add_(float scalar) {
    auto& dst = *impl_->storage_->data;
    for (int i = 0; i < numel(); ++i)
        dst[i] += scalar;
}

void Tensor::mul_(const Tensor& other) {
    if (shape() != other.shape())
        throw std::runtime_error("mul_: shapes must match");
    auto& dst = *impl_->storage_->data;
    const auto& src = other.data();
    for (int i = 0; i < numel(); ++i)
        dst[i] *= src[i];
}

void Tensor::mul_(float scalar) {
    auto& dst = *impl_->storage_->data;
    for (int i = 0; i < numel(); ++i)
        dst[i] *= scalar;
}

Tensor Tensor::mul(float scalar) const {
    std::vector<float> result(numel());
    const auto& base = *impl_->storage_->data;
    for (int i = 0; i < numel(); ++i)
        result[i] = base[i] * scalar;
    return Tensor(shape(), result, requires_grad());
}

// ========= Grad accumulation =========
void Tensor::accumulate_grad(const Tensor& g) {
    if (!requires_grad()) return;

    if (!grad()) {
        // initialize zero grad buffer
        auto zero = std::make_shared<Tensor>(shape(), std::vector<float>(numel(), 0.0f), /*requires_grad=*/false);
        set_grad(std::move(zero));
    }

    auto& dst = grad()->data();         // write into grad buffer
    const auto& src = g.data();

    if (dst.size() != src.size()) {
        throw std::runtime_error("Gradient shape mismatch in accumulate_grad.");
    }
    for (size_t i = 0; i < dst.size(); ++i) {
        dst[i] += src[i];
    }
}

// ===== NEW: low-level view builder =====
Tensor Tensor::as_strided(const std::vector<int>& sizes,
                          const std::vector<int>& strides,
                          int64_t storage_offset) const {
    // Basic checks
    if (sizes.size() != strides.size()) {
        throw std::runtime_error("as_strided: sizes and strides rank mismatch");
    }
    // For now, ensure non-negative sizes
    for (int s : sizes) {
        if (s < 0) throw std::runtime_error("as_strided: negative size not allowed");
    }

    // Create a fresh impl that shares Storage
    auto impl = std::make_shared<TensorImpl>(std::vector<int>{1}, 0.0f, /*requires_grad=*/false);
    // Overwrite fields to avoid reallocating Storage
    impl->storage_ = this->impl_->storage_;                      // share storage
    impl->storage_offset_ = this->impl_->storage_offset_ + storage_offset;
    impl->shape_  = sizes;                                       // logical sizes
    impl->stride_ = strides;                                     // logical strides

    // Fresh AutogradMeta, but preserve requires_grad flag from base
    impl->autograd_ = std::make_shared<AutogradMeta>();
    impl->autograd_->requires_grad = this->requires_grad();
    // NOTE: grad_fn is nullptr here. Proper view-autograd will come in Step 4.

    return Tensor(std::move(impl));
}

// ===== NEW: transpose_view (metadata-only) =====
Tensor Tensor::transpose_view(int dim0, int dim1) const {
    const auto rank = static_cast<int>(impl_->shape_.size());
    if (dim0 == dim1) return *this;
    if (rank < 2) throw std::runtime_error("transpose_view: rank < 2");
    if (dim0 < 0 || dim1 < 0 || dim0 >= rank || dim1 >= rank) {
        throw std::runtime_error("transpose_view: dim out of range");
    }

    auto sizes   = impl_->shape_;
    auto strides = impl_->stride_;
    std::swap(sizes[dim0],   sizes[dim1]);
    std::swap(strides[dim0], strides[dim1]);

    // Offset unchanged for a pure transpose view
    return as_strided(sizes, strides, /*storage_offset=*/0);
}

// ===== NEW: slice_view (metadata-only) =====
// Slice [start:end) along 'dim'
Tensor Tensor::slice_view(int dim, int start, int end) const {
    const auto rank = static_cast<int>(impl_->shape_.size());
    if (rank == 0) throw std::runtime_error("slice_view: rank == 0");
    if (dim < 0 || dim >= rank) throw std::runtime_error("slice_view: dim out of range");

    const int size_d = impl_->shape_[dim];
    if (start < 0 || end < 0 || start > end || end > size_d) {
        throw std::runtime_error("slice_view: invalid [start,end) for dimension");
    }

    auto sizes   = impl_->shape_;
    auto strides = impl_->stride_;
    sizes[dim]   = end - start;

    // Increase storage_offset by start * stride_at_dim
    const int64_t delta = static_cast<int64_t>(start) * static_cast<int64_t>(strides[dim]);

    return as_strided(sizes, strides, /*storage_offset=*/delta);
}


    int64_t Tensor::storage_offset() const {
        return impl_->storage_offset_;
    }
    const float* Tensor::data_ptr() const {
        return impl_->storage_->data->data();
    }
    float* Tensor::data_ptr() {
        return impl_->storage_->data->data();
    }


} // namespace znet
