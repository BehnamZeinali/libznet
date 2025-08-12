#include <znet/optim/sgd.hpp>
#include <znet/autograd/grad_mode.hpp>

namespace znet {
namespace optim {

SGD::SGD(std::vector<std::shared_ptr<Tensor>> parameters, float lr)
    : parameters_(std::move(parameters)), lr_(lr) {}

void SGD::step() {
    GradMode::NoGradGuard ng;
    for (auto& param : parameters_) {
        if (param->requires_grad() && param->grad()) {
            auto& data = param->data();
            const auto& grad = param->grad()->data();
            for (size_t i = 0; i < data.size(); ++i) {
                data[i] -= lr_ * grad[i];
            }
        }
    }
}

void SGD::zero_grad() {
    for (auto& param : parameters_) {
        if (param->grad()) {
            std::fill(param->grad()->data().begin(), param->grad()->data().end(), 0.0f);
        }
    }
}

}  // namespace optim
}  // namespace znet
