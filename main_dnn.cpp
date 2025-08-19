#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <algorithm>

#include "znet/autograd/tensor.hpp"
#include <iostream>
#include "znet/autograd/tensor.hpp"
#include "znet/nn/module.hpp"
#include "znet/nn/linear.hpp"
#include "znet/nn/relu.hpp"
#include "znet/nn/loss.hpp"
#include "znet/optim/sgd.hpp"

// std::vector<float> load_binary_file(const std::string& filename, size_t count) {
//     std::vector<float> data(count);
//     std::ifstream file(filename, std::ios::binary);
//     if (!file) throw std::runtime_error("Failed to open " + filename);
//     file.read(reinterpret_cast<char*>(data.data()), count * sizeof(float));
//     return data;
// }


// Load image data: expects float32 array of size 60000 * 784
std::vector<float> load_data(const std::string& filename, size_t num_samples, size_t num_features) {
    size_t total_size = num_samples * num_features;
    std::vector<float> data(total_size);

    FILE* file = fopen(filename.c_str(), "rb");
    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        std::exit(1);
    }

    size_t read_size = fread(data.data(), sizeof(float), total_size, file);
    if (read_size != total_size) {
        std::cerr << "Error reading data: expected " << total_size << " elements, got " << read_size << std::endl;
        std::exit(1);
    }

    fclose(file);
    return data;
}

// Load label data: expects int32 array of size 60000
std::vector<int> load_labels(const std::string& filename, size_t num_samples) {
    std::vector<int> labels(num_samples);

    FILE* file = fopen(filename.c_str(), "rb");
    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        std::exit(1);
    }

    size_t read_size = fread(labels.data(), sizeof(int), num_samples, file);
    if (read_size != num_samples) {
        std::cerr << "Error reading labels: expected " << num_samples << " elements, got " << read_size << std::endl;
        std::exit(1);
    }

    fclose(file);
    return labels;
}

std::vector<float> convert_to_float(const std::vector<int>& labels) {
    std::vector<float> float_labels(labels.size());
    std::transform(labels.begin(), labels.end(), float_labels.begin(),
                   [](int x) { return static_cast<float>(x); });
    return float_labels;
}

struct MLP : znet::nn::Module {
    MLP() {
        fc1 = register_module("fc1", znet::nn::Linear(784, 256));
        relu = register_module("relu", znet::nn::ReLU());
        fc2 = register_module("fc2", znet::nn::Linear(256, 10));
    }

    znet::Tensor forward(znet::Tensor x) {
        x.set_requires_grad(true);
        auto out = fc1.forward(x);
        out = relu.forward(out);
        return fc2.forward(out);
    }

    znet::nn::Linear fc1, fc2;
    znet::nn::ReLU relu;
};

// class MyMLP : public znet::nn::Module {
// public:
//     MyMLP(int in_features, int hidden, int out_features)
//         : fc1(in_features, hidden),
//           relu(),
//           fc2(hidden, out_features) {
//         register_module("fc1", std::make_shared<znet::nn::Linear>(in_features, hidden));
//         register_module("relu", std::make_shared<znet::nn::ReLU>());
//         register_module("fc2", std::make_shared<znet::nn::Linear>(hidden, out_features));
//     }

//     znet::Tensor forward(const znet::Tensor& x) override {
//         auto out = fc1.forward(x);
//         out = relu.forward(out);
//         return fc2.forward(out);
//     }

// private:
//     znet::nn::Linear fc1;
//     znet::nn::ReLU relu;
//     znet::nn::Linear fc2;
// };

int main() {
    // Load binary files
    // auto train_x_data = load_binary_file("mnist_data/train_images.bin", 60000 * 784);
    // auto train_y_data = load_binary_file("mnist_data/train_labels.bin", 60000);
    // auto test_x_data  = load_binary_file("mnist_data/test_images.bin", 10000 * 784);
    // auto test_y_data  = load_binary_file("mnist_data/test_labels.bin", 10000);

    std::vector<float> train_x_data = load_data("mnist_data/X_train.bin", 60000, 784); // size 60000 * 784
    std::vector<float> train_y_data = load_data("mnist_data/y_train.bin", 60000 , 1);              // size 60000
    std::vector<float> test_x_data = load_data("mnist_data/X_test.bin", 10000, 784); // size 60000 * 784
    std::vector<float> test_y_data = load_data("mnist_data/y_test.bin", 10000 , 1);              // size 60000


    for (int i = 0; i < 10; ++i) {
    std::cout << "Label " << i << ": " << test_y_data[i] << std::endl;
}
    // Wrap as znet::Tensor
    znet::Tensor train_x({60000, 784}, train_x_data);
    znet::Tensor train_y({60000 }, train_y_data);
    znet::Tensor test_x({10000, 784}, test_x_data);
    znet::Tensor test_y({10000}, test_y_data);

    std::cout << "Train X_train shape: ";
    train_x.print_shape();
    std::cout << "Train Y_train shape: ";
    train_y.print_shape();

    std::cout << "Train X_test shape: ";
    test_x.print_shape();
    std::cout << "Train Y_test shape: ";
    test_y.print_shape();

    // Next step: define model, loss, optimizer, train loop...
    MLP model;
    znet::nn::CrossEntropyLoss criterion;
    znet::optim::SGD optimizer(model.parameters(), 0.01f);

    int epochs = 50;
    int batch_size = 4;

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        auto start = std::chrono::high_resolution_clock::now();
        float total_loss = 0.0f;

        for (int i = 0; i < 60000; i += batch_size) {
            int end = std::min(i + batch_size, 60000);
            int cur_batch = end - i;

            znet::Tensor x = train_x.slice(0, i, end);
            znet::Tensor y = train_y.slice(0, i, end);

            x.set_requires_grad(true);

            znet::Tensor logits = model.forward(x);
            znet::Tensor loss = criterion(logits, y);
            // std::cout << "loss calculated: " << loss.data()[0] << std::endl;
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            total_loss += loss.data()[0];
            if ((i / batch_size + 1) % 100 == 0 ){
                    std::cout << "Epoch " << epoch << ", Batch " << (i / batch_size + 1)
                      << " | Loss: " << loss.data()[0] << "\n";
            }
            
        }

        auto end = std::chrono::high_resolution_clock::now();
        double epoch_time = std::chrono::duration<double>(end - start).count();

        std::cout << "Epoch " << epoch
                  << " | Loss: " << total_loss / (60000.0f / batch_size)
                  << " | Time: " << epoch_time << " sec\n";
    }

    return 0;
}
