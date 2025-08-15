#pragma once

#include <znet/nn/module.hpp>
#include <znet/autograd/tensor.hpp>

namespace znet {
namespace nn {

class LinearImpl : public Module {
public:
    LinearImpl(int in_features, int out_features, bool use_bias = true);
    LinearImpl() : LinearImpl(1, 1) {}  // Default dummy

    Tensor forward(const Tensor& input) override;

private:
    int in_features_;
    int out_features_;
    bool use_bias_;

    std::shared_ptr<Tensor> weight_;
    std::shared_ptr<Tensor> bias_;

    
};

} // namespace nn
} // namespace znet
