#include <iostream>
#include <znet/autograd/tensor.hpp>
#include "znet/autograd/ops.hpp"
#include <znet/nn/linear.hpp>
#include <znet/nn/loss.hpp>

// g++ -std=c++17 -Iinclude src/autograd/tensor.cpp main.cpp -o main
// g++ -std=c++17 -Iinclude src/autograd/tensor.cpp src/autograd/autograd.cpp src/autog
// rad/ops.cpp main.cpp -o main

// g++ -std=c++17 -Iinclude src/autograd/tensor.cpp src/nn/module.cpp src/autograd/autograd_function.cpp src/autograd/ops.cpp src/nn/linear_impl.cpp rc/nn/loss_impl.cpp main.cpp -o main

// cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
// cmake --build build

// cmake -S . -B build_release -DCMAKE_BUILD_TYPE=Release
// cmake --build build_release

// # From your project root:
// mkdir build
// cd build
// cmake ..
// make
int main() {
//    // Create a 2x3 tensor with custom values
//     std::vector<float> values = {1.0, 2.0, 3.0,
//                                  4.0, 5.0, 6.0};
//     znet::Tensor b({2, 3}, values);
//     b.print_shape();  // Output: Shape: (2, 3)
//     b.print_data();   // Output: Data: [1, 2, 3, 4, 5, 6]

//     b.print();  // Output: Shape: (2, 3) Data: [1, 2, 3, 4, 5, 6]

//     std::cout << "t[1][2] = " << b[1][2] << std::endl;  // Should print 6


//     znet::Tensor x({2, 2}, {1.0, 2.0, 3.0, 4.0});
//     x.set_requires_grad(true);

//     std::cout << "x.requires_grad = " << std::boolalpha << x.requires_grad() << std::endl;

//     auto grad = std::make_shared<znet::Tensor>(std::vector<int>{2, 2}, 1.0f);
//     x.set_grad(grad);

//     std::cout << "x.grad = ";
//     x.grad()->print();

//     znet::Tensor a({2}, {1.0, 2.0}, true);  // requires_grad = true
//     znet::Tensor b_({2}, {3.0, 4.0}, true);
//     std::cout << "a requires_grad = " << std::boolalpha << a.requires_grad() << std::endl;
//     auto c = add(a, b_);
//     c.backward();

//     std::cout << "a.grad: ";
//     a.grad()->print();

//     std::cout << "b.grad: ";
//     b_.grad()->print();

//     // std::cout << "b.grad: ";
//     // b_.grad()->print();     // Should be [1, 1]

//     znet::Tensor m({2, 3}, {1, 2, 3, 4, 5, 6}, true);
//     znet::Tensor n({3, 2}, {7, 8, 9, 10, 11, 12}, true);
//     znet::Tensor k = znet::matmul(m, n);

//     k.print();
//     k.backward();

//     m.grad()->print();
//     n.grad()->print();
    znet::nn::Linear linear(2, 3);

    znet::Tensor input_({1, 2}, {0.5f, -1.0f});
    znet::Tensor output = linear(input_);

    std::cout << "Output:\n";
    output.print();

    using namespace znet;
    using namespace znet::nn;

    // Step 1: Create model
    Linear layer1(2, 3);  // Linear 1
    Linear layer2(3, 1);  // Linear 2

    // Step 2: Create input and target
    Tensor input({1, 2}, {0.5f, -1.0f});
    Tensor target({1, 1}, {0.8f});

    // Step 3: Forward pass
    Tensor out1 = layer1(input);
    std::cout << "out1 requires grad: " << std::boolalpha << out1.requires_grad() << "\n";
    std::cout << "out1 has grad_fn: " << std::boolalpha << (out1.grad_fn() != nullptr) << "\n";
    Tensor out2 = layer2(out1);

    // Step 4: Compute loss = MSE: (out2 - target)^2
    Tensor diff = add(out2, Tensor({1, 1}, {-target[0][0]}));  // out2 - target
    Tensor loss = mul(diff, diff);  // squared error

    std::cout << "Loss:\n";
    loss.print();

    // Step 5: Backward
    loss.backward();

    // Step 6: Print gradients
    std::cout << "\nGradients for layer1:\n";
    for (const auto& p : layer1.ptr()->parameters()) {
        std::cout << "- Requires grad? " << std::boolalpha << p->requires_grad() << "\n";
        if (!p->grad()) {
            std::cout << "  → No grad computed.\n";
        } else {
            p->grad()->print();
        }
    }

    std::cout << "\nGradients for layer2:\n";
    for (const auto& p : layer2.ptr()->parameters()) {
        std::cout << "- Requires grad? " << std::boolalpha << p->requires_grad() << "\n";
        if (!p->grad()) {
            std::cout << "  → No grad computed.\n";
        } else {
            p->grad()->print();
        }
    }

    // Simulated logits (batch_size=2, num_classes=3)
    std::vector<float> logits_data = {
        2.0f, 0.5f, 0.1f,   // Sample 1
        0.2f, 1.5f, 0.3f    // Sample 2
    };
    znet::Tensor logits({2, 3}, logits_data);  // shape: (2, 3)
    logits.set_requires_grad(true);  // Enable gradient tracking
    // True class indices for the 2 samples: class 0 and class 1
    std::vector<float> target_data = {0, 1};
    znet::Tensor targets({2}, target_data);    // shape: (2,)

    // Instantiate the loss
    znet::nn::CrossEntropyLoss loss_fn;

    // Compute loss
    znet::Tensor loss_ = loss_fn.forward(logits, targets);

    // Output
    std::cout << "Loss value:\n";
    loss_.print();
    std::cout << "Loss requires grad: " << std::boolalpha << logits.requires_grad() << "\n";
    loss_.backward();  // Backpropagate

    std::cout << "Logits grad:" << std::endl;
    logits.grad()->print();  // Print gradients
    return 0;

}

// # Remove the .git folder (this deletes all Git history)
// rm -rf .git

// # Reinitialize Git
// git init

// # Add your .gitignore file first
// git add .gitignore

// # Add all other files except ignored ones
// git add .

// # Commit them
// git commit -m "Initial commit without dataset and compiled files"

// # Link to your GitHub repo
// git remote add origin https://github.com/<your-username>/<repo-name>.git

// # Push fresh history
// git branch -M main
// git push -u origin main --force
