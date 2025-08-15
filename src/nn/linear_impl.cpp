#include <znet/nn/linear_impl.hpp>
#include <znet/autograd/ops_impl.hpp>  // use *_impl not frontend
#include "znet/autograd/autograd_function.hpp"
#include <random>
#include <cmath>

namespace znet {
namespace nn {

LinearImpl::LinearImpl(int in_features, int out_features, bool use_bias)
    : in_features_(in_features), out_features_(out_features), use_bias_(use_bias) {
    
    float limit = std::sqrt(6.0f / (in_features + out_features));
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(-limit, limit);

    std::vector<float> weight_data(in_features * out_features);
    for (auto& w : weight_data) {
        w = dist(gen);
    }

    Tensor weight({out_features, in_features}, weight_data, /*requires_grad=*/true);
    register_parameter("weight", weight);
    weight_ = parameters_.at("weight");

    if (use_bias_) {
        std::vector<float> bias_data(out_features);
        for (auto& b : bias_data) {
            b = dist(gen);
        }
        Tensor bias({out_features}, bias_data, /*requires_grad=*/true);
        register_parameter("bias", bias);
        bias_ = parameters_.at("bias");
    }
}

Tensor LinearImpl::forward(const Tensor& input) {
    // Transpose weight from [out_features, in_features] -> [in_features, out_features]
    // Tensor weight_T = weight_->transpose(0, 1);

    // matmul: [B, in] x [in, out] = [B, out]
    // Tensor output = matmul_impl_(input, *weight_);
    // // Tensor output = matmul_impl(input, weight_T);

    // if (use_bias_) {
    //     Tensor output_ = add_impl(output, *bias_);
    //     return output_;
        
    // }
    // return output;

    // weight: [out,in]; use a view for weight^T: [in,out], no copy
    Tensor Wt = weight_->transpose_view(0,1);
    Tensor output = matmul_impl(input, Wt); // both flags false inside
     std::cout << "LinearImpl::forward: input shape = " ;// << input.shape() << ", weight shape = " << weight_->shape() << ", output shape = " << output.shape() << std::endl;
    if (use_bias_) output = add_impl(output, *bias_);
    return output;
    

    
}



} // namespace nn
} // namespace znet
