#pragma once

#include <znet/nn/module.hpp>
#include <znet/autograd/tensor.hpp>

namespace znet {
namespace nn {

class ReLUImpl : public Module {
public:
    ReLUImpl() = default;
    Tensor forward(const Tensor& input) override;
};

} // namespace nn
} // namespace znet
