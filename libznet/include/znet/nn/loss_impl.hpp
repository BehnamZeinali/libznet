#pragma once
#include <znet/nn/module.hpp>
#include <znet/autograd/tensor.hpp>

namespace znet {
namespace nn {

class CrossEntropyLossImpl : public Module {
public:
    CrossEntropyLossImpl() = default;

    // Accepts two inputs: logits and target
    // Tensor forward(const Tensor& logits, const Tensor& target);  // NOTE: overload, not override
    Tensor forward(const Tensor& input, const Tensor& target) override;

    Tensor forward(const Tensor& input) override {
        throw std::runtime_error("CrossEntropyLoss requires both input and target.");
    }

};

} // namespace nn
} // namespace znet
