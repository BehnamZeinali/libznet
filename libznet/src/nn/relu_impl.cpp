#include <znet/nn/relu_impl.hpp>
#include <znet/autograd/ops_impl.hpp>  // use *_impl, not frontend ops

namespace znet {
namespace nn {

Tensor ReLUImpl::forward(const Tensor& input) {
    return relu_impl(input);  // backend op with autograd support
}

} // namespace nn
} // namespace znet
