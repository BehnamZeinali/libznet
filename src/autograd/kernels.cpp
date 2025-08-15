#include <znet/autograd/kernels.hpp>
#include <znet/autograd/tensor_iter.hpp>
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
        // result[i] = std::max(0.0f, input.data()[i]);
        result[i] = ( input.data()[i] > 0.0f) ? input.data()[i] : 0.0f;

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


static inline int64_t numel_from_sizes(const std::vector<int>& sizes) {
    int64_t n = 1;
    for (int s : sizes) n *= s;
    return n;
}

void add_kernel_strided(Tensor& out, const Tensor& a, const Tensor& b) {
    // Build iterator from inputs and the *out* logical shape
    auto it = make_iter2_for_binary(a, b, out.shape());

    const float* a_base = a.data_ptr();
    const float* b_base = b.data_ptr();
    float* out_base     = out.data_ptr();

    const int ndim = it.ndim;
    const int64_t total = numel_from_sizes(it.sizes);

    // Fast path: fully contiguous, no broadcast (optional)
    bool a_contig = true, b_contig = true;
    for (int d = 0; d < ndim; ++d) {
        // contiguous if stride matches rowmajor and no broadcast (stride!=0)
        a_contig = a_contig && (it.a_strides[d] == it.out_strides[d]);
        b_contig = b_contig && (it.b_strides[d] == it.out_strides[d]);
    }
    if (a_contig && b_contig && it.out_offset == 0) {
        for (int64_t i = 0; i < total; ++i) {
            out_base[i] = a_base[i + it.a_offset] + b_base[i + it.b_offset];
        }
        return;
    }

    // General path: multi-index with carry
    std::vector<int> idx(ndim, 0);
    for (int64_t linear = 0; linear < total; ++linear) {
        int64_t a_off = it.a_offset;
        int64_t b_off = it.b_offset;
        int64_t o_off = it.out_offset;

        for (int d = 0; d < ndim; ++d) {
            a_off += static_cast<int64_t>(idx[d]) * it.a_strides[d];
            b_off += static_cast<int64_t>(idx[d]) * it.b_strides[d];
            o_off += static_cast<int64_t>(idx[d]) * it.out_strides[d];
        }

        out_base[o_off] = a_base[a_off] + b_base[b_off];

        // increment idx with carry from last dim
        for (int d = ndim - 1; d >= 0; --d) {
            if (++idx[d] < it.sizes[d]) break;
            idx[d] = 0;
        }
    }
}


void relu_kernel_strided(Tensor& out, const Tensor& x) {
    // Output shape == input shape for ReLU
    auto it = make_iter1_for_unary(x, out.shape());

    const float* in_base = x.data_ptr();
    float* out_base      = out.data_ptr();

    const int ndim = it.ndim;
    const int64_t total = numel_from_sizes(it.sizes);

    // Fast path: input contiguous and aligned with output (common case)
    bool in_contig = true;
    for (int d = 0; d < ndim; ++d) {
        in_contig = in_contig && (it.in_strides[d] == it.out_strides[d]);
    }
    if (in_contig && it.out_offset == 0) {
        const int64_t in_off = it.in_offset;
        for (int64_t i = 0; i < total; ++i) {
            float v = in_base[in_off + i];
            out_base[i] = (v > 0.0f) ? v : 0.0f;
        }
        return;
    }

    // General path
    std::vector<int> idx(ndim, 0);
    for (int64_t linear = 0; linear < total; ++linear) {
        int64_t in_off  = it.in_offset;
        int64_t out_off = it.out_offset;

        for (int d = 0; d < ndim; ++d) {
            in_off  += static_cast<int64_t>(idx[d]) * it.in_strides[d];
            out_off += static_cast<int64_t>(idx[d]) * it.out_strides[d];
        }

        float v = in_base[in_off];
        out_base[out_off] = (v > 0.0f) ? v : 0.0f;

        // increment idx with carry
        for (int d = ndim - 1; d >= 0; --d) {
            if (++idx[d] < it.sizes[d]) break;
            idx[d] = 0;
        }
    }
}

static inline std::vector<int64_t> rowmajor_strides(const std::vector<int>& sizes) {
    std::vector<int64_t> s(sizes.size());
    int64_t stride = 1;
    for (int i = static_cast<int>(sizes.size()) - 1; i >= 0; --i) {
        s[i] = stride;
        stride *= sizes[i];
    }
    return s;
}

static inline bool is_contiguous_like(const Tensor& t,
                                      const std::vector<int>& sizes) {
    if (t.shape() != sizes) return false;
    if (t.numel() == 0) return true;
    if constexpr (true) {
        // storage_offset()==0 is the common fast case in your codebase
        if (t.storage_offset() != 0) return false;
    }
    auto want = rowmajor_strides(sizes);
    const auto& have = t.stride();
    if (have.size() != want.size()) return false;
    for (size_t d = 0; d < have.size(); ++d) {
        if (static_cast<int64_t>(have[d]) != want[d]) return false;
    }
    return true;
}

// grad_in += (x > 0) ? grad_out : 0
void relu_backward_kernel_accum(Tensor& grad_in, const Tensor& x, const Tensor& grad_out) {
    const auto& sizes = x.shape();
    if (grad_out.shape() != sizes || grad_in.shape() != sizes) {
        throw std::runtime_error("relu_backward: shape mismatch");
    }

    const int64_t total = numel_from_sizes(sizes);

    // Try the fully-contiguous fast path (all three contiguous, offset 0)
    const bool x_contig  = is_contiguous_like(x, sizes);
    const bool go_contig = is_contiguous_like(grad_out, sizes);
    const bool gi_contig = is_contiguous_like(grad_in, sizes);

    if (x_contig && go_contig && gi_contig) {
        const float* __restrict xptr  = x.data_ptr();
        const float* __restrict goptr = grad_out.data_ptr();
        float* __restrict giptr       = grad_in.data_ptr();
        // Note: x.storage_offset()==0 etc. enforced by is_contiguous_like
        for (int64_t i = 0; i < total; ++i) {
            const float v  = xptr[i];
            const float go = goptr[i];
            // branchless mask (usually vectorized)
            giptr[i] += (v > 0.0f ? go : 0.0f);
        }
        return;
    }

    // Generic strided path
    // Collect/pad strides and offsets
    const int ndim = static_cast<int>(sizes.size());

    // Left-pad strides to ndim
    auto pad64 = [&](const std::vector<int>& v)->std::vector<int64_t>{
        std::vector<int64_t> r(ndim - static_cast<int>(v.size()), 0);
        r.insert(r.end(), v.begin(), v.end());
        return r;
    };

    const auto x_str  = pad64(x.stride());
    const auto go_str = pad64(grad_out.stride());
    const auto gi_str = pad64(grad_in.stride()); // grad_in is typically row-major

    const int64_t x_off  = x.storage_offset();
    const int64_t go_off = grad_out.storage_offset();
    const int64_t gi_off = grad_in.storage_offset();

    // Multi-dimensional counter
    std::vector<int> idx(ndim, 0);

    const float* __restrict xbase  = x.data_ptr();
    const float* __restrict gobase = grad_out.data_ptr();
    float* __restrict gibase       = grad_in.data_ptr();

    for (int64_t lin = 0; lin < total; ++lin) {
        int64_t xo = x_off, goo = go_off, gio = gi_off;
        // map logical idx -> storage offsets
        for (int d = 0; d < ndim; ++d) {
            const int64_t step = static_cast<int64_t>(idx[d]);
            xo  += step * x_str[d];
            goo += step * go_str[d];
            gio += step * gi_str[d];
        }

        const float v  = xbase[xo];
        const float go = gobase[goo];
        gibase[gio] += (v > 0.0f ? go : 0.0f);

        // bump idx with carry
        for (int d = ndim - 1; d >= 0; --d) {
            if (++idx[d] < sizes[d]) break;
            idx[d] = 0;
        }
    }
}

struct MatRef {
    // logical matrix sizes (per batch)
    int M, N, K;
    // strides (in elements) for walking the two axes inside the matrix
    int64_t s_row; // stride to advance the "row" index
    int64_t s_col; // stride to advance the "col" index
    // batch stride (elements) to jump to the next matrix in the batch
    int64_t s_batch;
    // base pointer and starting offset (elements)
    const float* base; // or float* for destination
    int64_t offset;
};

// Extract a matrix reference from a Tensor for its last-2 dims.
// If trans==false: rows=last-2, cols=last-1. If trans==true: swap roles logically by swapping strides/sizes.
MatRef as_matrix_ref(const Tensor& T, bool trans=false) {
    const auto& sh = T.shape();
    const auto& st = T.stride();
    const int D = (int)sh.size();
    if (D < 2) throw std::runtime_error("matmul: rank must be >= 2");

    const int dR = D - 2;   // row dim (pre-trans)
    const int dC = D - 1;   // col dim (pre-trans)

    int M = sh[dR], N = sh[dC];
    int64_t sR = st[dR], sC = st[dC];

    if (trans) { std::swap(M, N); std::swap(sR, sC); }

    MatRef r;
    r.M = M;
    r.N = N;
    r.K = -1;                 // not used; set to sentinel to avoid accidental use
    r.s_row = sR;
    r.s_col = sC;
    r.s_batch = 0;            // you’ll fill this when adding batched support
    r.base = T.data().data(); // your Tensor gives access to std::vector<float>
    r.offset = T.storage_offset(); // use your accessor; units: elements
    return r;
}

// Compute broadcasted leading shape (align right)
std::vector<int> broadcast_leading(const std::vector<int>& a, const std::vector<int>& b) {
    const int da=(int)a.size(), db=(int)b.size();
    const int d=std::max(da,db);
    std::vector<int> out(d,1);
    for (int i=0;i<d;++i){
        int ai=(i<d-da)?1:a[i-(d-da)];
        int bi=(i<d-db)?1:b[i-(d-db)];
        if (ai!=bi && ai!=1 && bi!=1) throw std::runtime_error("matmul: broadcast mismatch");
        out[i]=std::max(ai,bi);
    }
    return out;
}

inline int64_t prod(const std::vector<int>& v){ int64_t p=1; for(int x:v)p*=x; return p; }

std::vector<int> compute_matmul_out_shape_view(const Tensor& A, const Tensor& B) {
    const auto& As = A.shape();
    A.print_shape();
    const auto& Bs = B.shape();
    B.print_shape();
    if (As.size() < 2 || Bs.size() < 2)
        throw std::runtime_error("matmul: rank must be >= 2");

    // last-2 dims define the matrix multiply (views already encode transposes)
    int M  = As[As.size()-2];
    int K  = As[As.size()-1];
    int K2 = Bs[Bs.size()-2];
    int N  = Bs[Bs.size()-1];
    // std::cout << "matmul: M=" << M << ", K=" << K << ", K2=" << K2 << ", N=" << N << std::endl;
    if (K != K2){
        std::cout << "matmul: M=" << M << ", K=" << K << ", K2=" << K2 << ", N=" << N << std::endl;
        throw std::runtime_error("matmul: inner dims do not match___");
    }
        

    std::vector<int> Alead(As.begin(), As.end()-2);
    std::vector<int> Blead(Bs.begin(), Bs.end()-2);

    std::vector<int> L = broadcast_leading(Alead, Blead);
    std::vector<int> out = L;
    out.push_back(M);
    out.push_back(N);
    return out;
}

// Compute out shape for a batched matmul with logical transposes:
// out = (transA? A^T : A) @ (transB ? B^T : B)
// i.e., take the last-two dims after transposes as (M,K) and (K,N)
std::vector<int> compute_mm_out_shape_flags(const Tensor& A, const Tensor& B,
                                                   bool transA, bool transB) {
    const auto& As = A.shape();
    const auto& Bs = B.shape();
    if (As.size() < 2 || Bs.size() < 2)
        throw std::runtime_error("matmul backward: rank must be >= 2");

    const int A_M = transA ? As[As.size()-1] : As[As.size()-2];
    const int A_K = transA ? As[As.size()-2] : As[As.size()-1];
    const int B_K = transB ? Bs[Bs.size()-1] : Bs[Bs.size()-2];
    const int B_N = transB ? Bs[Bs.size()-2] : Bs[Bs.size()-1];
    if (A_K != B_K)
        throw std::runtime_error("matmul backward: inner dim mismatch");

    std::vector<int> Alead(As.begin(), As.end()-2);
    std::vector<int> Blead(Bs.begin(), Bs.end()-2);
    std::vector<int> L = broadcast_leading(Alead, Blead);

    std::vector<int> out = L;
    out.push_back(A_M);
    out.push_back(B_N);
    return out;
}

// C[...] = A[...] @ B[...]
// A_mat: (M,K) via strides; B_mat: (K,N) via strides; C is contiguous MxN per batch
void matmul_strided_batched_kernel(const Tensor& A, const Tensor& B, Tensor& C,
                                          bool A_logical_trans, bool B_logical_trans) {
    const auto& Ash=A.shape(); const auto& Bsh=B.shape();
    if (Ash.size()<2 || Bsh.size()<2) throw std::runtime_error("matmul: rank>=2");

    // Build refs to last-2 dims with logical transpose via strides
    MatRef Ar = as_matrix_ref(A, A_logical_trans);
    MatRef Br = as_matrix_ref(B, B_logical_trans);
    std::cout << "matmul: A=" << Ar.M << "x" << Ar.K << ", B=" << Br.K << "x" << Br.N << std::endl;
    if (Ar.K != Br.K) throw std::runtime_error("matmul: inner dim mismatch");

    // Broadcast leading dims
    std::vector<int> Alead(Ash.begin(), Ash.end()-2);
    std::vector<int> Blead(Bsh.begin(), Bsh.end()-2);
    std::vector<int> L = broadcast_leading(Alead, Blead);

    // Output shape = L + [M,N] (already allocated by caller)
    const int M = Ar.M, N = Br.N, K = Ar.K;
    const int64_t BATCH = L.empty()? 1 : prod(L);

    // Build multipliers to decode a flat batch index into per-tensor batch offsets (for broadcast)
    const int d = (int)L.size();
    std::vector<int64_t> Lmul(d,1), Amul(d,0), Bmul(d,0);
    for (int i=d-2;i>=0;--i) Lmul[i]=Lmul[i+1]*L[i+1];

    // For your current row-major contiguous case: if a dim equals 1, that tensor “broadcasts” ⇒ offset multiplier 0.
    // Otherwise, step size is the product of the sizes of the trailing dims (here we reuse s_batch per matrix).
    {
        int64_t a_step = Ar.s_batch; // elements to jump one A matrix
        int64_t b_step = Br.s_batch;
        for (int i=d-1, ia=(int)Alead.size()-1; i>=0; --i, --ia) {
            int asz = (ia>=0)? Alead[ia] : 1;
            Amul[i] = (asz==1)? 0 : a_step;
            if (asz!=1) a_step *= asz; // for completeness; with contiguous it equals product of higher dims
        }
        for (int i=d-1, ib=(int)Blead.size()-1; i>=0; --i, --ib) {
            int bsz = (ib>=0)? Blead[ib] : 1;
            Bmul[i] = (bsz==1)? 0 : b_step;
            if (bsz!=1) b_step *= bsz;
        }
    }

    const float* Abase = Ar.base;
    const float* Bbase = Br.base;
    float*       Cbase = C.data_ptr();
    const int64_t Cstride = (int64_t)M * N;

    for (int64_t b = 0; b < BATCH; ++b) {
        // decode batch b into per-tensor element offsets
        int64_t aoff = Ar.offset, boff = Br.offset;
        int64_t t = b;
        for (int i=0;i<d;++i) {
            const int idx = (d==0)? 0 : (int)(t / Lmul[i]) % L[i];
            t = (d==0)? 0 : (t % Lmul[i]);
            aoff += idx * Amul[i];
            boff += idx * Bmul[i];
        }
        const float* A0 = Abase + aoff;
        const float* B0 = Bbase + boff;
        float*       C0 = Cbase + b * Cstride;

        // Plain 3-loop GEMM using strides (no materialization)
        for (int i=0;i<M;++i) {
            float* Crow = C0 + (int64_t)i * N;
            const float* Ai0 = A0 + (int64_t)i * Ar.s_row;
            for (int j=0;j<N;++j) {
                const float* Bj0 = B0 + (int64_t)j * Br.s_col;
                float acc = 0.0f;
                const float* Ai = Ai0;
                const float* Bj = Bj0;
                for (int k=0;k<K;++k) {
                    acc += *Ai * *Bj;
                    Ai += Ar.s_col; // advance along K in A
                    Bj += Br.s_row; // advance along K in B
                }
                Crow[j] = acc;
            }
        }
    }
}

} // namespace znet
