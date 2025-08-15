#include <znet/autograd/kernels.hpp>
#include <stdexcept>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <numeric>

namespace znet {

Tensor add_kernel(const Tensor& a, const Tensor& b) {
    const auto& as = a.shape();
    const auto& bs = b.shape();

    // Case 1: exact same shape
    if (as == bs) {
        std::vector<float> out(a.numel());
        for (int i = 0; i < a.numel(); ++i)
            out[i] = a.data()[i] + b.data()[i];
        // IMPORTANT: kernels return no-grad tensors
        return Tensor(as, std::move(out), /*requires_grad=*/false);
    }

    // Case 2: (B,F) + (F) row-broadcast
    if (as.size() == 2 && bs.size() == 1 && as[1] == bs[0]) {
        int B = as[0], F = as[1];
        std::vector<float> out(a.numel());
        for (int i = 0; i < B; ++i)
            for (int j = 0; j < F; ++j)
                out[i * F + j] = a.data()[i * F + j] + b.data()[j];
        return Tensor(as, std::move(out), /*requires_grad=*/false);
    }

    throw std::runtime_error("add_kernel: unsupported broadcast pattern");
}



Tensor matmul_kernel(const Tensor& a, const Tensor& b) {
    const auto& a_shape = a.shape();
    const auto& b_shape = b.shape();

    if (a_shape.size() != 2 || b_shape.size() != 2)
        throw std::runtime_error("matmul: only 2D tensors are supported for now");

    int m = a_shape[0];
    int k1 = a_shape[1];
    int k2 = b_shape[0];
    int n = b_shape[1];

    if (k1 != k2)
        throw std::runtime_error("matmul: inner dimensions must match");

    Tensor out({m, n}, std::vector<float>(m * n, 0.0f));

    const auto& a_data = a.data();
    const auto& b_data = b.data();
    auto& out_data = out.data();

    const auto& a_stride = a.stride();
    const auto& b_stride = b.stride();
    const auto& out_stride = out.stride();

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < k1; ++k) {
                int a_idx = i * a_stride[0] + k * a_stride[1];
                int b_idx = k * b_stride[0] + j * b_stride[1];
                sum += a_data[a_idx] * b_data[b_idx];
            }
            int out_idx = i * out_stride[0] + j * out_stride[1];
            out_data[out_idx] = sum;
        }
    }

    return out;
}

Tensor relu_kernel(const Tensor& input) {
    std::vector<float> result(input.numel());
    for (int i = 0; i < input.numel(); ++i)
        result[i] = std::max(0.0f, input.data()[i]);

    bool requires_grad = input.requires_grad();
    Tensor out(input.shape(), result, requires_grad);
    
    return out;
}

Tensor cross_entropy_kernel(const Tensor& logits, const Tensor& target) {
    const auto& log_data = logits.data();
    const auto& target_data = target.data();

    if (logits.shape().size() != 2 || target.shape().size() != 1)
        throw std::runtime_error("CrossEntropy expects 2D logits and 1D target");

    int batch_size = logits.shape()[0];
    int num_classes = logits.shape()[1];

    std::vector<float> loss_values(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        float max_logit = -std::numeric_limits<float>::infinity();
        for (int c = 0; c < num_classes; ++c)
            max_logit = std::max(max_logit, log_data[i * num_classes + c]);

        float sum_exp = 0.0f;
        for (int c = 0; c < num_classes; ++c)
            sum_exp += std::exp(log_data[i * num_classes + c] - max_logit);

        int label = static_cast<int>(target_data[i]);
        float log_prob = log_data[i * num_classes + label] - max_logit - std::log(sum_exp);
        loss_values[i] = -log_prob;
    }

    float mean_loss = std::accumulate(loss_values.begin(), loss_values.end(), 0.0f) / batch_size;

    Tensor out({}, {mean_loss}, logits.requires_grad());
    
    return out;
}


// C = A^T @ B
// A: [M, K], B: [M, N]  ->  C: [K, N]
Tensor matmul_AT_B_kernel(const Tensor& A, const Tensor& B) {
    const auto& as = A.shape();
    const auto& bs = B.shape();
    if (as.size() != 2 || bs.size() != 2) {
        throw std::runtime_error("matmul_AT_B_kernel: inputs must be 2D");
    }
    int M = as[0], K = as[1];
    if (bs[0] != M) {
        throw std::runtime_error("matmul_AT_B_kernel: A.rows must equal B.rows");
    }
    int N = bs[1];

    const auto& ad = A.data();
    const auto& bd = B.data();
    std::vector<float> cd(K * N, 0.0f);

    // C[k, n] = sum_m A[m, k] * B[m, n]
    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int m = 0; m < M; ++m) {
                acc += ad[m * K + k] * bd[m * N + n];
            }
            cd[k * N + n] = acc;
        }
    }
    return Tensor({K, N}, std::move(cd), /*requires_grad=*/false);
}

// C = A @ B^T
// A: [M, K], B: [N, K]  ->  C: [M, N]
Tensor matmul_A_BT_kernel(const Tensor& A, const Tensor& B) {
    const auto& as = A.shape();
    const auto& bs = B.shape();
    if (as.size() != 2 || bs.size() != 2) {
        throw std::runtime_error("matmul_A_BT_kernel: inputs must be 2D");
    }
    int M = as[0], K = as[1];
    if (bs[1] != K) {
        throw std::runtime_error("matmul_A_BT_kernel: A.cols must equal B.cols");
    }
    int N = bs[0];

    const auto& ad = A.data();
    const auto& bd = B.data();
    std::vector<float> cd(M * N, 0.0f);

    // C[m, n] = sum_k A[m, k] * B[n, k]
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += ad[m * K + k] * bd[n * K + k];
            }
            cd[m * N + n] = acc;
        }
    }
    return Tensor({M, N}, std::move(cd), /*requires_grad=*/false);
}


} // namespace znet
