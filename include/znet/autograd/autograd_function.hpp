// #pragma once
// #include <znet/autograd/tensor.hpp>
// #include <vector>

// namespace znet {

// // Base Function interface for autograd
// struct Function {
//     Function();
//     virtual ~Function();

//     // Backward pass: apply gradient to inputs
//     virtual void backward(Tensor& grad_out) = 0;

//     // Return the input tensors involved in this operation
//     virtual std::vector<Tensor*> inputs() const = 0;

//     virtual const char* name() const { return "Function"; }
// };

// // ==================== Add ====================

// struct AddFunction : public Function {
//     const Tensor* lhs;
//     const Tensor* rhs;
   
//     const char* name() const override { return "AddFunction"; }

//     AddFunction(const Tensor* l, const Tensor* r);
//     void backward(Tensor& grad_out) override;
//     std::vector<Tensor*> inputs() const override;

//     // Tensor lhs;
//     // Tensor rhs;

//     // AddFunction(const Tensor& l, const Tensor& r) ;
//     // const char* name() const override { return "AddFunction"; }
//     // void backward(Tensor& grad_out) override;
//     // std::vector<Tensor*> inputs() const override ; 
// };

// // ==================== Matmul ====================
// struct MatmulFunction : public Function {
//     const Tensor* lhs;
//     const Tensor* rhs;
//     const char* name() const override { return "MatmulFunction"; }
//     MatmulFunction(const Tensor* l, const Tensor* r);
//     void backward(Tensor& grad_out) override;
//     std::vector<Tensor*> inputs() const override;


//     // Tensor lhs;
//     // Tensor rhs;

//     // MatmulFunction(const Tensor& l, const Tensor& r) ;
//     // const char* name() const override { return "MatmulFunction"; }
//     // void backward(Tensor& grad_out) override;
//     // std::vector<Tensor*> inputs() const override;
// };
// // ==================== ReLU ====================
// struct ReLUFunction : public Function {
//     const Tensor* input;
//     const char* name() const override { return "ReLUFunction"; }
//     ReLUFunction(const Tensor* input);
//     void backward(Tensor& grad_out) override;
//     std::vector<Tensor*> inputs() const override;

//     // Tensor input;
//     // ReLUFunction(const Tensor& input);
//     // const char* name() const override { return "ReLUFunction"; }
//     // void backward(Tensor& grad_out) override;
//     // std::vector<Tensor*> inputs() const override;
// };

// // ==================== CrossEntropy ====================
// struct CrossEntropyFunction : public Function {
//     const Tensor* logits;
//     const Tensor* target;
//     const char* name() const override { return "CrossEntropyLossFunction"; }
//     CrossEntropyFunction(const Tensor* l, const Tensor* r);
//     void backward(Tensor& grad_out) override;
//     std::vector<Tensor*> inputs() const override;

//     // Tensor logits;
//     // Tensor target;

//     // CrossEntropyFunction(const Tensor& l, const Tensor& r) ;
//     // const char* name() const override { return "MatmulFunction"; }
//     // void backward(Tensor& grad_out) override;
//     // std::vector<Tensor*> inputs() const override ;
// };

// // ==================== Elementwise Multiply ====================
// // struct MulFunction : public Function {
// //     const Tensor* lhs;
// //     const Tensor* rhs;

// //     const char* name() const override { return "MulFunction"; }
// //     MulFunction(const Tensor* l, const Tensor* r);
// //     void backward(Tensor& grad_out) override;
// //     std::vector<Tensor*> inputs() const override;
// // };
// // // ==================== Softmax ====================
// // struct SoftmaxFunction : public Function {
// //     const Tensor* input;
// //     const char* name() const override { return "SoftmaxFunction"; }
// //     SoftmaxFunction(const Tensor* input);
// //     void backward(Tensor& grad_out) override;
// //     std::vector<Tensor*> inputs() const override;
// // };

// }  // namespace znet

#pragma once
#include <znet/autograd/tensor.hpp>
#include <memory>
#include <vector>

namespace znet {

// -------- Base --------
struct Function {
    Function();
    virtual ~Function();

    virtual void backward(Tensor& grad_out) = 0;
    virtual std::vector<Tensor*> inputs() const = 0;
    virtual const char* name() const { return "Function"; }
};

// -------- SavedTensor (like PyTorch SavedVariable) --------
struct SavedTensor {
    std::shared_ptr<TensorImpl> impl;   // alias exact impl used at forward
    // If you support views, also store sizes/strides/storage_offset here.

    void save(const Tensor& t) {
        impl = t.impl();                 // shared_ptr copy; cheap alias
    }
    Tensor unpack() const {
        // Recreate a handle to the same impl
        return Tensor::from_impl(impl);  // provide this factory in Tensor
    }
};

// ==================== Add ====================
struct AddFunction : public Function {
    SavedTensor a_, b_;        // captured at forward
    Tensor a_view_, b_view_;   // stable views we return from inputs()

    AddFunction(const Tensor& a, const Tensor& b) ;
    const char* name() const override { return "AddFunction"; }
    std::vector<Tensor*> inputs() const override ;
    void backward(Tensor& grad_out) override;
};

// ==================== Matmul ====================
struct MatmulFunction : public Function {
    SavedTensor a_, b_;
    Tensor a_view_, b_view_;

    
    MatmulFunction(const Tensor& a, const Tensor& b) ;

    const char* name() const override { return "MatmulFunction"; }
    std::vector<Tensor*> inputs() const override ;
    void backward(Tensor& grad_out) override;
};

// ==================== ReLU ====================
struct ReLUFunction : public Function {
    SavedTensor x_;
    Tensor x_view_;

   
    // ReLUFunction(const Tensor& x) ;

     // NEW: save the *output* y = relu(x) to build the mask safely
    SavedTensor y_;
    Tensor      y_view_;
    ReLUFunction(const Tensor& x);

    const char* name() const override { return "ReLUFunction"; }
    std::vector<Tensor*> inputs() const override ;
    void backward(Tensor& grad_out) override;
};

// ==================== CrossEntropy ====================
struct CrossEntropyFunction : public Function {
    SavedTensor logits_, target_;
    Tensor logits_view_, target_view_;

    CrossEntropyFunction(const Tensor& l, const Tensor& t) ;

    const char* name() const override { return "CrossEntropyLossFunction"; }
    std::vector<Tensor*> inputs() const override ;
    void backward(Tensor& grad_out) override;
};

} // namespace znet
