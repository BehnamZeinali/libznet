# libznet
LibZnet is a minimalist deep learning framework in C++ and Cuda

### What’s new in v0.2
- Up to **5-D** tensors, **64-bit** shapes/strides.
- Views: `as_strided`, `transpose_view`, `slice_view`, `select`.
- Python-style indexing: `a[0][1]` → chain to rank-0, read via `.item()`.
- `reshape`, `view`, `contiguous` added.

```cpp
using znet::Tensor;

// Construct
Tensor a({2,3,4}, 0.0f, false);

// Indexing / select
a.at({1,2,3}) = 7.0f;
std::cout << a[1][2][3].item() << "\n";     // 7

// Views
auto aT = a.transpose_view(0,1);            // metadata-only
auto a01 = a.select(0,0).select(0,1);       // shape [4]

// Reshape
Tensor x4d({60000,1,28,28}, ...);
auto x2d = x4d.view({60000, 28*28});        // zero-copy if contiguous
