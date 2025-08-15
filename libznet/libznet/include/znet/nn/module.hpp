#pragma once

#include <string>
#include <memory>
#include <unordered_map>
#include <vector>
#include "znet/autograd/tensor.hpp"

namespace znet {
namespace nn {

class Module {
public:
    Module() = default;
    virtual ~Module() = default;

    virtual Tensor forward(const Tensor& input);
    virtual Tensor forward(const Tensor& input, const Tensor& target);

    template <typename ModuleType>
    ModuleType register_module(const std::string& name, ModuleType module) {
        children_[name] = module.ptr();
        return module;
    }

    void register_parameter(const std::string& name, const Tensor& param);

    virtual std::vector<std::shared_ptr<Tensor>> parameters();

    virtual void train(bool mode = true) { is_training_ = mode; }
    bool is_training() const { return is_training_; }

    const std::unordered_map<std::string, std::shared_ptr<Module>>& children() const;
    const std::unordered_map<std::string, std::shared_ptr<Tensor>>& parameters() const;

protected:
    std::unordered_map<std::string, std::shared_ptr<Module>> children_;
    std::unordered_map<std::string, std::shared_ptr<Tensor>> parameters_;
    bool is_training_ = true;
};

} // namespace nn
} // namespace znet
