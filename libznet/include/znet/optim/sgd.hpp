#pragma once

#include <vector>
#include <memory>
#include <znet/autograd/tensor.hpp>

namespace znet {
namespace optim {

class SGD {
public:
    SGD(std::vector<std::shared_ptr<Tensor>> parameters, float lr = 0.01f);

    void step();
    void zero_grad();

private:
    std::vector<std::shared_ptr<Tensor>> parameters_;
    float lr_;
};

}  // namespace optim
}  // namespace znet
