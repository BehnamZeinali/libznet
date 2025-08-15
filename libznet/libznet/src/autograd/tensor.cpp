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
    data_.resize(total, val);
    compute_stride();
    if (requires_grad) {
        autograd_ = std::make_shared<AutogradMeta>();
        autograd_->requires_grad = true;
    }
}

TensorImpl::TensorImpl(std::vector<int> shape, std::vector<float> values, bool requires_grad)
    : shape_(std::move(shape)), data_(std::move(values)) {
    int total = 1;
    for (int dim : shape_) {
        if (dim <= 0) throw std::invalid_argument("Shape dimensions must be > 0");
        total *= dim;
    }
    if (total != static_cast<int>(data_.size())) {
        throw std::invalid_argument("Number of elements does not match shape");
    }
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
int Tensor::numel() const { return static_cast<int>(impl_->data_.size()); }

void Tensor::print_shape() const {
    std::cout << "Shape: (";
    for (size_t i = 0; i < impl_->shape_.size(); ++i) {
        std::cout << impl_->shape_[i];
        if (i + 1 < impl_->shape_.size()) std::cout << ", ";
    }
    std::cout << ")\n";
}

void Tensor::print_data() const {
    std::cout << "Data: [";
    for (size_t i = 0; i < impl_->data_.size(); ++i) {
        std::cout << impl_->data_[i];
        if (i + 1 < impl_->data_.size()) std::cout << ", ";
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

    if (impl_->shape_.size() == 1) {
        std::cout << "[";
        for (size_t i = 0; i < impl_->data_.size(); ++i) {
            std::cout << impl_->data_[i];
            if (i + 1 < impl_->data_.size()) std::cout << ", ";
        }
        std::cout << "]\n";
    } else if (impl_->shape_.size() == 2) {
        int rows = impl_->shape_[0];
        int cols = impl_->shape_[1];
        for (int i = 0; i < rows; ++i) {
            std::cout << "[";
            for (int j = 0; j < cols; ++j) {
                std::cout << impl_->data_[i * cols + j];
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

float Tensor::operator()(int index) const { return impl_->data_[index]; }
float& Tensor::operator()(int index) { return impl_->data_[index]; }
const std::vector<float>& Tensor::data() const { return impl_->data_; }
std::vector<float>& Tensor::data() { return impl_->data_; }

// Tensor Tensor::transpose(int dim1, int dim2) const {
//     // NOTE: This mutates shared impl metadata in-place in your original design.
//     // Keeping behavior unchanged per your request.
//     std::vector<int> new_shape = impl_->shape_;
//     std::vector<int> new_stride = impl_->stride_;
//     std::swap(new_shape[dim1], new_shape[dim2]);
//     std::swap(new_stride[dim1], new_stride[dim2]);

//     Tensor result = *this;
//     result.impl_->shape_ = std::move(new_shape);
//     result.impl_->stride_ = std::move(new_stride);
//     return result;
// }

// tensor.cpp
Tensor Tensor::transpose(int dim1, int dim2) const {
    const auto& s = impl_->shape_;
    if (dim1 == dim2) return *this;
    if (s.size() < 2) throw std::runtime_error("transpose: rank < 2");

    // Only handle 2D for now (matches your printing/ops)
    if (s.size() != 2) throw std::runtime_error("transpose: only 2D supported here");

    int rows = s[0], cols = s[1];
    std::vector<float> out(rows * cols);
    // materialize: [rows, cols] -> [cols, rows]
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            out[j * rows + i] = impl_->data_[i * cols + j];

    // Keep requires_grad the same; this is a *forward* op (autograd wiring happens in the wrapper/impl)
    return Tensor({cols, rows}, std::move(out), /*requires_grad=*/requires_grad());
}


Tensor Tensor::slice(int dim, int start, int end) const {
    if (start < 0 || end > impl_->shape_[dim] || start >= end) {
        throw std::runtime_error("slice: invalid slice range");
    }

    if (impl_->shape_.size() == 1) {
        std::vector<float> sliced_data(impl_->data_.begin() + start, impl_->data_.begin() + end);
        return Tensor({end - start}, sliced_data, requires_grad());
    }

    if (impl_->shape_.size() == 2) {
        int rows = impl_->shape_[0];
        int cols = impl_->shape_[1];
        std::vector<float> sliced_data;

        if (dim == 0) {
            for (int i = start; i < end; ++i)
                for (int j = 0; j < cols; ++j)
                    sliced_data.push_back(impl_->data_[i * cols + j]);
            return Tensor({end - start, cols}, sliced_data, requires_grad());
        } else if (dim == 1) {
            for (int i = 0; i < rows; ++i)
                for (int j = start; j < end; ++j)
                    sliced_data.push_back(impl_->data_[i * cols + j]);
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
    return TensorSlice(impl_->data_.data(), row * stride, stride);
}

const Tensor::TensorSlice Tensor::operator[](int row) const {
    if (impl_->shape_.size() != 2) throw std::runtime_error("Only 2D indexing is supported for now.");
    int stride = impl_->shape_[1];
    if (row < 0 || row >= impl_->shape_[0]) throw std::out_of_range("Row index out of bounds");
    return TensorSlice(const_cast<float*>(impl_->data_.data()), row * stride, stride);
}

void Tensor::add_(const Tensor& other) {
    if (shape() != other.shape())
        throw std::runtime_error("add_: shapes must match for in-place addition");
    for (int i = 0; i < numel(); ++i)
        impl_->data_[i] += other.data()[i];
}

void Tensor::add_(float scalar) {
    for (int i = 0; i < numel(); ++i)
        impl_->data_[i] += scalar;
}

void Tensor::mul_(const Tensor& other) {
    if (shape() != other.shape())
        throw std::runtime_error("mul_: shapes must match");
    for (int i = 0; i < numel(); ++i)
        impl_->data_[i] *= other.data()[i];
}

void Tensor::mul_(float scalar) {
    for (int i = 0; i < numel(); ++i)
        impl_->data_[i] *= scalar;
}

Tensor Tensor::mul(float scalar) const {
    std::vector<float> result(numel());
    for (int i = 0; i < numel(); ++i)
        result[i] = impl_->data_[i] * scalar;
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

} // namespace znet
