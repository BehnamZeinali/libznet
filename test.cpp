#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include "znet/autograd/tensor.hpp"
#include "znet/nn/linear.hpp"
#include "znet/nn/relu.hpp"
#include "znet/nn/loss.hpp"
#include "znet/optim/sgd.hpp"

using namespace znet;

// Minimal dummy dataset generator
std::vector<float> generate_dummy_images(int num_samples, int num_features) {
    std::vector<float> data(num_samples * num_features);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (float& x : data) x = dist(gen);
    return data;
}

std::vector<float> generate_dummy_labels(int num_samples, int num_classes = 10) {
    std::vector<float> labels(num_samples);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(0, num_classes - 1);
    for (float& x : labels) x = static_cast<float>(dist(gen));
    return labels;
}

struct MLP : nn::Module {
    MLP() {
        fc1 = register_module("fc1", nn::Linear(784, 256));
        relu = register_module("relu", nn::ReLU());
        fc2 = register_module("fc2", nn::Linear(256, 10));
    }

    Tensor forward(Tensor x) {
        x.set_requires_grad(true);
        x = fc1.forward(x);
        x = relu.forward(x);
        return fc2.forward(x);
    }

    nn::Linear fc1, fc2;
    nn::ReLU relu;
};

int main() {
    const int num_samples = 1024;
    const int input_dim = 784;
    const int num_classes = 10;
    const int batch_size = 256;
    const int epochs = 10;

    std::vector<float> X_data = generate_dummy_images(num_samples, input_dim);
    std::vector<float> y_data = generate_dummy_labels(num_samples, num_classes);

    Tensor X({num_samples, input_dim}, X_data);
    Tensor y({num_samples}, y_data);

    MLP model;
    nn::CrossEntropyLoss loss_fn;
    optim::SGD optimizer(model.parameters(), 0.01f);

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        float total_loss = 0.0f;
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < num_samples; i += batch_size) {
            int end = std::min(i + batch_size, num_samples);
            int cur_batch = end - i;

            Tensor x = X.slice(0, i, end);
            Tensor y_batch = y.slice(0, i, end);

            x.set_requires_grad(true);
            Tensor logits = model.forward(x);
            Tensor loss = loss_fn(logits, y_batch);

            // std::cout << "requires_grad (logits): " << logits.requires_grad() << std::endl;
            // std::cout << "requires_grad (loss): " << loss.requires_grad() << std::endl;
            optimizer.zero_grad();
            loss.backward();

            // std::cout << "Weight before: ";
            // model.fc1.parameters()['weights']  print_data();
           
            optimizer.step();
           
            // std::cout << "Weight after: ";
            // model.fc1.weight.print_data();
            //             optimizer.step();

            total_loss += loss.data()[0];
            std::cout << "Epoch " << epoch << " | Batch " << (i / batch_size)
                      << " | Loss: " << loss.data()[0] << std::endl;
        }

        auto end = std::chrono::high_resolution_clock::now();
        double t = std::chrono::duration<double>(end - start).count();

        std::cout << "Epoch " << epoch << " complete | Avg Loss: "
                  << total_loss / (num_samples / batch_size)
                  << " | Time: " << t << " sec\n";
    }

    return 0;
}
