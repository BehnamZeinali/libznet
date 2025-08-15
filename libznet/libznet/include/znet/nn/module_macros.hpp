// include/znet/nn/module_macros.hpp
#pragma once

#define ZNET_MODULE(Name)                                        \
    class Name {                                                 \
    public:                                                      \
        template <typename... Args>                              \
        Name(Args&&... args)                                     \
            : impl_(std::make_shared<Name##Impl>(                \
                  std::forward<Args>(args)...)) {}               \
                                                                 \
        znet::Tensor forward(const znet::Tensor& input) const {  \
            return impl_->forward(input);                        \
        }                                                        \
                                                                 \
        znet::Tensor operator()(const znet::Tensor& input) const { \
            return forward(input);                               \
        }                                                        \
                                                                 \
        std::shared_ptr<Name##Impl> ptr() const { return impl_; }\
                                                                 \
    private:                                                     \
        std::shared_ptr<Name##Impl> impl_;                       \
    };
