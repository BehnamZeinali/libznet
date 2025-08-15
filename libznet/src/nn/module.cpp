#include <znet/nn/module.hpp>

namespace znet {
namespace nn {

Tensor Module::forward(const Tensor& input) {
    throw std::runtime_error("This module does not support single-input forward().");
}

Tensor Module::forward(const Tensor& input, const Tensor& target) {
    throw std::runtime_error("This module does not support two-input forward().");
}

void Module::register_parameter(const std::string& name, const Tensor& param) {
    parameters_[name] = std::make_shared<Tensor>(param);
}

std::vector<std::shared_ptr<Tensor>> Module::parameters() {
    std::vector<std::shared_ptr<Tensor>> result;
    for (auto& [_, p] : parameters_) {
        result.push_back(p);
    }
    for (auto& [_, c] : children_) {
        auto child_params = c->parameters();
        result.insert(result.end(), child_params.begin(), child_params.end());
    }
    return result;
}

const std::unordered_map<std::string, std::shared_ptr<Module>>& Module::children() const {
    return children_;
}

const std::unordered_map<std::string, std::shared_ptr<Tensor>>& Module::parameters() const {
    return parameters_;
}

} // namespace nn
} // namespace znet
