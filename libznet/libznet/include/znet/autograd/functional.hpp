#pragma once
#include <memory>
#include <vector>
#include <znet/autograd/tensor.hpp>

namespace znet {
namespace autograd {

class Function {
public:
    virtual ~Function() = default;

    // Apply backward pass: returns gradient(s) w.r.t. inputs
    virtual std::vector<std::shared_ptr<Tensor>> backward(const std::shared_ptr<Tensor>& grad_output) = 0;

    // Store inputs for use during backward
    std::vector<std::shared_ptr<Tensor>> saved_tensors;

    void save_for_backward(const std::vector<std::shared_ptr<Tensor>>& tensors) {
        saved_tensors = tensors;
    }
};

} // namespace autograd
} // namespace znet
