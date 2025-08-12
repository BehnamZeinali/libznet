// include/znet/autograd/autograd_meta.hpp
#pragma once

#include <memory>

namespace znet {

class Tensor;
struct Function;

struct AutogradMeta {
    bool requires_grad = false;
    std::shared_ptr<Tensor> grad = nullptr;
    std::shared_ptr<Function> grad_fn = nullptr;
};
}  // namespace znet
