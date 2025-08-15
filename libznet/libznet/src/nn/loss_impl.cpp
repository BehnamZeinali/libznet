#include <znet/nn/loss_impl.hpp>
#include <znet/autograd/ops_impl.hpp>  // Use the backend impl version

namespace znet {
namespace nn {

Tensor CrossEntropyLossImpl::forward(const Tensor& input, const Tensor& target) {
    return cross_entropy_impl(input, target);  // backend op with autograd
}

} // namespace nn
} // namespace znet
