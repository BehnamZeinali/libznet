#pragma once
#include <memory>
#include <znet/nn/loss_impl.hpp>

namespace znet {
namespace nn {

class CrossEntropyLoss {
public:
    CrossEntropyLoss() : impl_(std::make_shared<CrossEntropyLossImpl>()) {}

    Tensor forward(const Tensor& logits, const Tensor& target) const {
        return impl_->forward(logits, target);
    }

    Tensor operator()(const Tensor& logits, const Tensor& target) const {
        return forward(logits, target);
    }

    std::shared_ptr<CrossEntropyLossImpl> ptr() const { return impl_; }

private:
    std::shared_ptr<CrossEntropyLossImpl> impl_;
};

} // namespace nn
} // namespace znet
