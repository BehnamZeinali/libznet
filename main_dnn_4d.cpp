#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "znet/autograd/tensor.hpp"
#include "znet/nn/module.hpp"
#include "znet/nn/linear.hpp"
#include "znet/nn/relu.hpp"
#include "znet/nn/loss.hpp"
#include "znet/optim/sgd.hpp"

// ------------------------------
// Load a flat float32 .bin file
// Reads exactly total_count float values.
// ------------------------------
static std::vector<float> load_float32_bin(const std::string& filename, std::size_t total_count) {
    std::vector<float> data(total_count);
    FILE* f = std::fopen(filename.c_str(), "rb");
    if (!f) {
        std::perror(("open " + filename).c_str());
        std::exit(1);
    }
    std::size_t nread = std::fread(data.data(), sizeof(float), total_count, f);
    std::fclose(f);
    if (nread != total_count) {
        std::cerr << "Error: expected " << total_count << " float32s in " << filename
                  << ", got " << nread << "\n";
        std::exit(1);
    }
    return data;
}

// ------------------------------
// Simple 2-layer MLP for MNIST
// ------------------------------
struct MLP : znet::nn::Module {
    MLP() {
        fc1 = register_module("fc1", znet::nn::Linear(784, 256));
        relu = register_module("relu", znet::nn::ReLU());
        fc2 = register_module("fc2", znet::nn::Linear(256, 10));
    }

    znet::Tensor forward(znet::Tensor x) {
        // x: [B, 784]
        auto out = fc1.forward(x);
        out = relu.forward(out);
        return fc2.forward(out); // logits [B, 10]
    }

    znet::nn::Linear fc1, fc2;
    znet::nn::ReLU   relu;
};

// ------------------------------
// Argmax across dim=1 for a single row
// logits_row: Tensor view with shape [C] (e.g., 10)
// ------------------------------
static int argmax_1d(const znet::Tensor& logits_row) {
    int best_i = 0;
    float best_v = logits_row.at({0});
    // Safe read via at(); row is small (C=10)
    for (int i = 1; i < static_cast<int>(logits_row.shape()[0]); ++i) {
        float v = logits_row.at({i});
        if (v > best_v) { best_v = v; best_i = i; }
    }
    return best_i;
}

// ------------------------------
// Compute batch accuracy: logits [B, 10], labels [B] (class indices as float)
// Returns number of correct predictions in the batch.
// ------------------------------
static int batch_correct(const znet::Tensor& logits, const znet::Tensor& labels) {
    const auto& shp = logits.shape();          // [B, 10]
    const int64_t B = shp[0];
    int correct = 0;
    for (int64_t b = 0; b < B; ++b) {
        // logits[b] is a view of shape [10]
        znet::Tensor row = logits[b];
        int pred = argmax_1d(row);
        int truth = static_cast<int>(labels.at({b}));
        if (pred == truth) ++correct;
    }
    return correct;
}

// ------------------------------
// Optional normalization (divide by 255)
// ------------------------------
static void normalize_255_inplace(znet::Tensor& x) {
    x.mul_(1.0f / 255.0f);
}

int main() {
    // ---------- Load binaries ----------
    const int64_t Ntrain = 60000;
    const int64_t Ntest  = 10000;
    const int64_t C = 1, H = 28, W = 28, F = H * W; // 784

    // Images as flat float32 in row-major (already normalized? if not, we’ll normalize below)
    std::vector<float> train_x_data = load_float32_bin("mnist_data/X_train.bin", static_cast<std::size_t>(Ntrain * F));
    std::vector<float> test_x_data  = load_float32_bin("mnist_data/X_test.bin",  static_cast<std::size_t>(Ntest * F));

    // Labels as float32 class indices [0..9] (size N)
    std::vector<float> train_y_data = load_float32_bin("mnist_data/y_train.bin", static_cast<std::size_t>(Ntrain));
    std::vector<float> test_y_data  = load_float32_bin("mnist_data/y_test.bin",  static_cast<std::size_t>(Ntest));

    // ---------- Wrap into Tensors ----------
    // Images as 4D NCHW
    znet::Tensor train_x_4d({Ntrain, C, H, W}, std::move(train_x_data));
    znet::Tensor test_x_4d ({Ntest,  C, H, W}, std::move(test_x_data));

    // Flatten to [N, 784] for the MLP
    znet::Tensor train_x = train_x_4d.view({Ntrain, F});
    znet::Tensor test_x  = test_x_4d.view({Ntest,  F});

    // Labels as [N]
    znet::Tensor train_y({Ntrain}, std::move(train_y_data));
    znet::Tensor test_y ({Ntest},  std::move(test_y_data));

    // Optional: normalize images to [0,1]
    // normalize_255_inplace(train_x);
    // normalize_255_inplace(test_x);

    std::cout << "Train X shape: "; train_x.print_shape();   // [60000, 784]
    std::cout << "Train y shape: "; train_y.print_shape();   // [60000]
    std::cout << "Test  X shape: "; test_x.print_shape();    // [10000, 784]
    std::cout << "Test  y shape: "; test_y.print_shape();    // [10000]

    // ---------- Model / loss / optimizer ----------
    MLP model;
    znet::nn::CrossEntropyLoss criterion;    // expects logits [B,10] + class indices [B]
    znet::optim::SGD optimizer(model.parameters(), 0.01f);

    const int    epochs     = 10;
    const int    batch_size = 4;
    const int64_t num_train_batches = (Ntrain + batch_size - 1) / batch_size;

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        auto start = std::chrono::high_resolution_clock::now();
        float epoch_loss = 0.0f;
        int   epoch_correct = 0;
        int64_t seen = 0;

        for (int64_t i = 0; i < Ntrain; i += batch_size) {
            const int64_t end = std::min(i + (int64_t)batch_size, Ntrain);
            const int64_t curB = end - i;

            // Mini-batches (materialized slice copies are OK here)
            znet::Tensor x = train_x.slice(0, static_cast<int>(i), static_cast<int>(end)); // [B,784]
            znet::Tensor y = train_y.slice(0, static_cast<int>(i), static_cast<int>(end)); // [B]

            // Forward
            znet::Tensor logits = model.forward(x);           // [B,10]
            znet::Tensor loss   = criterion(logits, y);       // scalar {}

            // Accuracy (before backward)
            epoch_correct += batch_correct(logits, y);
            seen += static_cast<int>(curB);

            // Backward + update
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            // Accumulate loss
            epoch_loss += loss.item();

            // Logging
            const int batch_idx = static_cast<int>(i / batch_size) + 1;
            if (batch_idx % 100 == 0) {
                std::cout << "Epoch " << epoch
                          << "  Batch " << batch_idx << "/" << num_train_batches
                          << "  Loss " << loss.item()
                          << "  Acc "  << (100.0 * epoch_correct / double(seen)) << "%\n";
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        double epoch_time = std::chrono::duration<double>(end - start).count();

        double epoch_acc = (seen > 0) ? (100.0 * epoch_correct / double(seen)) : 0.0;
        std::cout << "[Train] Epoch " << epoch
                  << " | Loss: " << (epoch_loss / num_train_batches)
                  << " | Acc: "  << epoch_acc << "%"
                  << " | Time: " << epoch_time << "s\n";

        // --------- Evaluation on test set ---------
        float test_loss = 0.0f;
        int   test_correct = 0;
        int64_t test_seen = 0;

        for (int64_t i = 0; i < Ntest; i += batch_size) {
            const int64_t end = std::min(i + (int64_t)batch_size, Ntest);
            znet::Tensor x = test_x.slice(0, static_cast<int>(i), static_cast<int>(end));
            znet::Tensor y = test_y.slice(0, static_cast<int>(i), static_cast<int>(end));
            znet::Tensor logits = model.forward(x);
            znet::Tensor loss   = criterion(logits, y);
            test_loss += loss.item();
            test_correct += batch_correct(logits, y);
            test_seen += static_cast<int>(end - i);
        }

        std::cout << "[Test ] Epoch " << epoch
                  << " | Loss: " << (test_loss / ((Ntest + batch_size - 1) / batch_size))
                  << " | Acc: "  << (100.0 * test_correct / double(test_seen)) << "%\n";
    }

    return 0;
}
