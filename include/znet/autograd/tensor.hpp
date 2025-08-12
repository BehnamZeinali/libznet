// include/znet/tensor.hpp
#pragma once
#include <vector>
#include <memory>
#include <stdexcept>
#include <iostream>

#include "znet/autograd/autograd_meta.hpp"
#include "znet/autograd/autograd.hpp"

namespace znet {

class TensorImpl;

class Tensor {
public:
    // ----- Lightweight row view (assumes dense row-major for this helper) -----
    class TensorSlice {
    public:
        TensorSlice(float* data_ptr, int row_offset, int row_stride)
            : data_ptr_(data_ptr), row_offset_(row_offset), row_stride_(row_stride) {}
        float& operator[](int col) { return data_ptr_[row_offset_ + col]; }
    private:
        float* data_ptr_;
        int row_offset_;
        int row_stride_;
    };

    // ----- Construction -----
    Tensor();
    Tensor(std::vector<int> shape, float val = 0.0f, bool requires_grad = false);
    Tensor(std::vector<int> shape, std::vector<float> values, bool requires_grad = false);
    explicit Tensor(std::shared_ptr<TensorImpl> impl) : impl_(std::move(impl)) {}

    // Factory for SavedTensor::unpack()
    static Tensor from_impl(const std::shared_ptr<TensorImpl>& p) {
        Tensor t;
        t.impl_ = p;
        return t;
    }

    // ----- Impl handle accessors -----
    std::shared_ptr<TensorImpl>& impl() { return impl_; }
    const std::shared_ptr<TensorImpl>& impl() const { return impl_; }

    // ----- Shape / data -----
    const std::vector<int>& shape() const;
    const std::vector<int>& stride() const;
    int numel() const;

    const std::vector<float>& data() const;
    std::vector<float>& data();

    // ----- Pretty print (debug) -----
    void print_shape() const;
    void print_data() const;
    void print() const;

    // ----- Autograd flags / links -----
    bool requires_grad() const;
    void set_requires_grad(bool value);

    std::shared_ptr<Tensor> grad() const;                 // gradient buffer (Tensor)
    void set_grad(std::shared_ptr<Tensor> g);             // set gradient buffer
    void set_grad(const Tensor& g);                       // convenience

    std::shared_ptr<Function> grad_fn() const;
    void set_grad_fn(std::shared_ptr<Function> fn);

    void backward();                                      // seed 1s if needed and run engine

    // ----- Indexing -----
    float operator()(int index) const;
    float& operator()(int index);

    TensorSlice operator[](int row);
    const TensorSlice operator[](int row) const;

    // ----- Basic ops (front API) -----
    Tensor transpose(int dim1, int dim2) const;
    Tensor slice(int dim, int start, int end) const;

    void add_(const Tensor& other);
    void add_(float scalar);

    Tensor mul(const Tensor& other) const;
    void mul_(const Tensor& other);
    void mul_(float scalar);
    Tensor mul(float scalar) const;

    // ----- Grad accumulation -----
    void accumulate_grad(const Tensor& g); // allocates zero grad if null, then += elementwise

private:
    std::shared_ptr<TensorImpl> impl_;
};

// ----- Implementation object (owned via shared_ptr) -----
class TensorImpl {
public:
    TensorImpl(std::vector<int> shape, float val, bool requires_grad);
    TensorImpl(std::vector<int> shape, std::vector<float> values, bool requires_grad);

    void compute_stride();

    std::vector<float> data_;
    std::vector<int>   shape_;
    std::vector<int>   stride_;

    // Autograd meta (requires_grad, grad_fn, grad buffer, etc.)
    std::shared_ptr<AutogradMeta> autograd_ = nullptr;
};

} // namespace znet
