// main_test.cpp
#include <iostream>
#include <vector>

#include <znet/autograd/tensor.hpp>
#include <znet/autograd/kernels.hpp>     // matmul_strided_batched_kernel
#include <znet/autograd/shape_ops.hpp>   // compute_matmul_out_shape_view, prod

using namespace znet;

static void run_case(const char* title,
                     const Tensor& A,
                     const Tensor& B,
                     bool transA,
                     bool transB)
{
    std::cout << "\n=== " << title << " ===\n";
    // Compute output shape using your helper
    std::vector<int> out_shape = compute_matmul_out_shape_view(A, B);

    // Allocate output (contiguous)
    Tensor C(out_shape, 0.0f, /*requires_grad=*/false);

    // Run the kernel
    matmul_strided_batched_kernel(A, B, C, transA, transB);

    // Print kernel output
    std::cout << "Kernel output:\n";
    C.print();
}

int main() {
    // -------- Case 1: A(4x3) @ B(3x5)  => (4x5), flags: A=F, B=F --------
    // A has 12 elems (real numbers)
    std::vector<float> A1_vals = {
        1.0f,  2.0f,  3.0f,
        0.5f, -1.0f,  2.0f,
        4.0f,  0.0f, -0.5f,
        1.5f,  2.5f,  3.5f
    };
    Tensor A1({4,3}, A1_vals, /*requires_grad=*/false);

    // B has 15 elems (real numbers)
    std::vector<float> B1_vals = {
         2.0f,  0.0f, -1.0f,  1.0f,  0.5f,
         1.0f,  3.0f,  0.5f, -2.0f,  1.5f,
         0.0f, -1.0f,  2.0f,  0.5f, -0.5f
    };
    Tensor B1({3,5}, B1_vals, /*requires_grad=*/false);

    // Expected (do not compute in code):
    // C1 = A1 @ B1  (4x5)
    // [
    //   [ 4.00,  3.00,  6.00, -1.50,  2.00],
    //   [ 0.00, -5.00,  3.00,  3.50, -2.25],
    //   [ 8.00,  0.50, -5.00,  3.75,  2.25],
    //   [ 5.50,  4.00,  6.75, -1.75,  2.75]
    // ]
    run_case("Case 1: A[4x3] @ B[3x5] (A:F, B:F)", A1, B1, /*transA=*/false, /*transB=*/false);


    // -------- Case 2: A(4x3) @ B(5x3)^T => (4x5), flags: A=F, B=T --------
    // A (same 12 elems, still 4x3)
    Tensor A2 = A1;

    // B2 has 15 elems (5x3); we will multiply with B2^T inside the kernel
    std::vector<float> B2_vals = {
        2.0f,  0.0f, -1.0f,
        1.0f,  3.0f,  0.5f,
        0.0f, -1.0f,  2.0f,
        1.0f, -2.0f,  0.5f,
        0.5f,  1.5f, -0.5f
    };
    Tensor B2({5,3}, B2_vals, /*requires_grad=*/false);

    // Expected (do not compute in code):
    // C2 = A2 @ (B2^T)  (4x5)
    // [
    //   [-1.00,  8.50,  4.00, -1.50,  2.00],
    //   [-1.00, -1.50,  5.00,  3.50, -2.25],
    //   [ 8.50,  3.75, -1.00,  3.75,  2.25],
    //   [-0.50, 10.75,  4.50, -1.75,  2.75]
    // ]
    run_case("Case 2: A[4x3] @ B[5x3]^T (A:F, B:T)", A2, B2, /*transA=*/false, /*transB=*/true);


    // -------- Case 3: A(3x4)^T @ B(3x5) => (4x5), flags: A=T, B=F --------
    // A3 has 12 elems but shaped 3x4 (same flat values as A1, reinterpreted)
    std::vector<float> A3_vals = {
        1.0f,  2.0f,  3.0f,  0.5f,
       -1.0f,  2.0f,  4.0f,  0.0f,
       -0.5f,  1.5f,  2.5f,  3.5f
    };
    Tensor A3({3,4}, A3_vals, /*requires_grad=*/false);

    // B3 keep 3x5 (same as B1 values)
    Tensor B3 = B1;

    // Expected (do not compute in code):
    // C3 = (A3^T) @ B3  (4x5)
    // [
    //   [ 1.00, -2.50, -2.50,  2.75, -0.75],
    //   [ 6.00,  4.50,  2.00, -1.25,  3.25],
    //   [10.00,  9.50,  4.00, -3.75,  6.25],
    //   [ 1.00, -3.50,  6.50,  2.25, -1.50]
    // ]
    run_case("Case 3: A[3x4]^T @ B[3x5] (A:T, B:F)", A3, B3, /*transA=*/true, /*transB=*/false);


    // -------- Case 4: A(3x4)^T @ B(5x3)^T => (4x5), flags: A=T, B=T --------
    // A4 (same as A3), B4 (same as B2)
    Tensor A4 = A3;
    Tensor B4 = B2;

    // Expected (do not compute in code):
    // C4 = (A4^T) @ (B4^T)  (4x5)
    // [
    //   [ 2.50, -2.25,  0.00,  2.75, -0.75],
    //   [ 2.50,  8.75,  1.00, -1.25,  3.25],
    //   [ 3.50, 16.25,  1.00, -3.75,  6.25],
    //   [-2.50,  2.25,  7.00,  2.25, -1.50]
    // ]
    run_case("Case 4: A[3x4]^T @ B[5x3]^T (A:T, B:T)", A4, B4, /*transA=*/true, /*transB=*/true);

    return 0;
}
