#include <cassert>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <cmath>

#include "znet/autograd/tensor.hpp"

using znet::Tensor;

static void expect(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "FAIL: " << msg << "\n";
        std::abort();
    }
}
static void expect_eq(int64_t a, int64_t b, const char* msg) {
    if (a != b) {
        std::cerr << "FAIL: " << msg << "  got=" << a << " expected=" << b << "\n";
        std::abort();
    }
}
static void expect_near(float a, float b, float eps, const char* msg) {
    if (std::fabs(a - b) > eps) {
        std::cerr << "FAIL: " << msg << "  got=" << a << " expected=" << b << "\n";
        std::abort();
    }
}
template <typename Fn>
static void expect_throw(Fn&& f, const char* msg) {
    bool threw = false;
    try { f(); } catch (...) { threw = true; }
    if (!threw) {
        std::cerr << "FAIL (no throw): " << msg << "\n";
        std::abort();
    }
}

int main() {
    // ---------- Step 1/2: 64-bit shape, row-major strides ----------
    {
        Tensor t({2,3,4,2,5}, 1.0f, false);
        expect_eq(t.numel(), 2*3*4*2*5, "numel 5D");
        const auto& s = t.stride(); // [3*4*2*5, 4*2*5, 2*5, 5, 1] => [120, 40, 10, 5, 1]
        expect(s.size() == 5, "stride rank==5");
        expect_eq(s[0], 120, "stride[0]");
        expect_eq(s[1], 40 , "stride[1]");
        expect_eq(s[2], 10 , "stride[2]");
        expect_eq(s[3], 5  , "stride[3]");
        expect_eq(s[4], 1  , "stride[4]");
        expect(t.is_contiguous(), "fresh contiguous");
    }

    // ---------- Step 3: norm_dim, offset_of, at() negative indices ----------
    {
        Tensor t({2,3,4}, 0.0f, false);
        // fill with linear i
        int v = 0;
        for (int i=0;i<2;i++) for (int j=0;j<3;j++) for (int k=0;k<4;k++) t.at({i,j,k}) = float(v++);
        // negative index -> last element
        expect_near(t.at({-1,-1,-1}), t.at({1,2,3}), 1e-6f, "negative indices");
        // flat offset sanity via at()
        expect_near(t.at({0,0,0}), 0.0f, 1e-6f, "offset 0");
        expect_near(t.at({0,0,1}), 1.0f, 1e-6f, "offset 1");
        expect_near(t.at({0,1,0}), 4.0f, 1e-6f, "offset stride hop");
    }

    // ---------- Step 4: as_strided basic view ----------
    {
        Tensor base({3,4}, 0.0f, false);
        int idx = 0;
        for (int i=0;i<3;i++) for (int j=0;j<4;j++) base.at({i,j}) = float(idx++);

        // Column-major view over the same storage (same shape, different strides)
        Tensor col_major = base.as_strided({3,4}, /*strides=*/{1,3}, /*off=*/0);
        expect(!col_major.is_contiguous(), "as_strided non-contig");
        // Check a few positions
        expect_near(col_major.at({0,0}), base.at({0,0}), 1e-6f, "as_strided(0,0)");
        expect_near(col_major.at({1,0}), base.at({0,1}), 1e-6f, "as_strided map (1,0)->(0,1)");
        expect_near(col_major.at({2,3}), base.at({2,3}), 1e-6f, "as_strided corner"); // We'll not over-constrain
    }

    // ---------- Step 5: transpose_view ----------
    {
        Tensor t({2,3,4}, 0.0f, false);
        int v=0;
        for (int i=0;i<2;i++) for (int j=0;j<3;j++) for (int k=0;k<4;k++) t.at({i,j,k}) = float(v++);
        auto tv = t.transpose_view(0,1); // shape [3,2,4]
        expect(tv.shape()[0]==3 && tv.shape()[1]==2 && tv.shape()[2]==4, "transpose_view shape");
        // spot-check mapping: tv[j,i,k] == t[i,j,k]
        expect_near(tv.at({1,0,2}), t.at({0,1,2}), 1e-6f, "transpose_view value");
    }

    // ---------- Step 6: N-D at() read/write ----------
    {
        Tensor t({2,2,2}, 0.0f, false);
        t.at({1,1,1}) = 42.0f;
        expect_near(t.at({-1,-1,-1}), 42.0f, 1e-6f, "at negative write");
    }

    // ---------- Step 7: print_limited smoke ----------
    {
        Tensor t({2,3,4}, 1.23f, false);
        t.print_limited(2); // should not crash; visual inspect optional
    }

    // ---------- Step 8: add_/mul_ (contiguous fast path + generic path) ----------
    {
        // contiguous path
        Tensor a({2,3}, 1.0f, false);
        Tensor b({2,3}, 2.0f, false);
        a.add_(b); // now 3s
        a.mul_(2.0f); // now 6s
        for (int i=0;i<2;i++) for (int j=0;j<3;j++) expect_near(a.at({i,j}), 6.0f, 1e-6f, "in-place contig");

        // generic path: make 'other' non-contiguous with as_strided but same logical shape
        Tensor x({2,2}, 10.0f, false);
        Tensor y({2,2}, 1.0f, false);
        // column-major view of y
        Tensor y_view = y.as_strided({2,2}, /*strides*/{1,2}, /*off*/0);
        expect(!y_view.is_contiguous(), "y_view non-contig");
        x.add_(y_view); // increments via generic path
        for (int i=0;i<2;i++) for (int j=0;j<2;j++) expect_near(x.at({i,j}), 11.0f, 1e-6f, "in-place generic");
    }

    // ---------- Step 9: materialized_transpose ----------
    {
        Tensor m({3,4}, 0.0f, false);
        int v=0;
        for (int i=0;i<3;i++) for (int j=0;j<4;j++) m.at({i,j}) = float(v++);
        Tensor mt = m.materialized_transpose(0,1); // shape [4,3]
        expect(mt.shape()[0]==4 && mt.shape()[1]==3, "materialized_transpose shape");
        // spot-check: mt[j,i] == m[i,j]
        expect_near(mt.at({2,1}), m.at({1,2}), 1e-6f, "materialized_transpose value");
        expect(mt.is_contiguous(), "materialized_transpose contiguous");
    }

    // ---------- Step 10: slice (copy path) ----------
    {
        Tensor t({2,3,5}, 0.0f, false);
        int v=0; for (int i=0;i<2;i++) for (int j=0;j<3;j++) for (int k=0;k<5;k++) t.at({i,j,k}) = float(v++);
        Tensor s = t.slice(/*dim=*/1, /*start=*/1, /*end=*/3); // shape [2,2,5]
        expect(s.shape()[0]==2 && s.shape()[1]==2 && s.shape()[2]==5, "slice shape");
        expect_near(s.at({1,1,4}), t.at({1,2,4}), 1e-6f, "slice value");
        // invalid range throws
        expect_throw([&]{ (void)t.slice(1, 2, 1); }, "slice invalid range");
    }

    // ---------- Step 11: transpose returns view semantics ----------
    {
        Tensor t({2,3}, 0.0f, false);
        int v=0; for (int i=0;i<2;i++) for (int j=0;j<3;j++) t.at({i,j}) = float(v++);
        Tensor tv = t.transpose(0,1);       // view
        tv.at({1,0}) = 999.0f;              // write through view
        expect_near(t.at({0,1}), 999.0f, 1e-6f, "transpose view writes through");
    }

    // ---------- Step 12: operator[] (2D row helper) ----------
    {
        Tensor t({3,4}, 0.0f, false);
        int v=0; for (int i=0;i<3;i++) for (int j=0;j<4;j++) t.at({i,j}) = float(v++);

        // row1 is a Tensor view of shape [4]
        Tensor row1 = t[1];
        expect(row1.shape().size() == 1 && row1.shape()[0] == 4, "row1 is shape [4]");

        // Read via .item()
        expect_near(row1[0].item(), t.at({1,0}), 1e-6f, "row view element read");

        // Write via .item_ref()
        row1[2].item_ref() = -5.0f;
        expect_near(t.at({1,2}), -5.0f, 1e-6f, "row view write-through");

        // Non-contiguous: now allowed; returns a view, not an error
        Tensor tv = t.transpose(0,1);     // shape [4,3], non-contiguous
        Tensor tv0 = tv[0];               // shape [3]
        expect(tv0.shape().size()==1 && tv0.shape()[0]==3, "tv[0] is a valid [3] view");

        // Rank!=2: also allowed; returns a view
        Tensor u({2,2,2}, 0.0f, false);
        Tensor u0 = u[0];                 // shape [2,2]
        expect(u0.shape().size()==2 && u0.shape()[0]==2 && u0.shape()[1]==2, "u[0] is [2,2]");
    }

    // ---------- Step 13: contiguous(), reshape(), view() ----------
    {
        Tensor t({2,3,4}, 0.0f, false);
        int v=0; for (int i=0;i<2;i++) for (int j=0;j<3;j++) for (int k=0;k<4;k++) t.at({i,j,k}) = float(v++);
        Tensor tv = t.transpose_view(0,1);      // non-contiguous view
        expect(!tv.is_contiguous(), "transpose_view non-contig");

        Tensor tc = tv.contiguous();            // materialize
        expect(tc.is_contiguous(), "contiguous() returns contiguous");
        expect(tc.shape() == tv.shape(), "contiguous() shape preserved");
        // data preserved
        for (int i=0;i<3;i++) for (int j=0;j<2;j++) for (int k=0;k<4;k++)
            expect_near(tc.at({i,j,k}), tv.at({i,j,k}), 1e-6f, "contiguous data");

        // reshape with -1
        Tensor r = t.reshape({-1, 4}); // (2*3) x 4
        expect(r.shape()[0]==6 && r.shape()[1]==4, "reshape -1 inferred");
        expect(r.is_contiguous(), "reshape of contiguous stays view");

        // view requires contiguous
        Tensor v2 = t.transpose_view(1,2);
        expect_throw([&]{ (void)v2.view({6,4}); }, "view on non-contig throws");
        // view on contiguous is ok
        Tensor v3 = t.view({6,4});
        expect(v3.is_contiguous(), "view returns view (contig)");
    }

    {

        Tensor a({2,3,4}, 0.0f, false);
        for (int i=0;i<2;i++)
        for (int j=0;j<3;j++)
            for (int k=0;k<4;k++)
            a.at({i,j,k}) = 100*i + 10*j + k;

        // Print a vector/matrix like Python:
        a[0].print();      // prints shape [3,4]
        a[0][1].print();   // prints shape [4]

        // Print a single element:
        std::cout << a[0][0][1].item() << "\n";  // 1

        // Write a single element via chain:
        a[1][2][3].item_ref() = 999.0f;
        std::cout << a.at({1,2,3}) << "\n";      // 999

        // Still available: direct N-D access
        a.at({0,2,3}) = 42.0f;
    }

    std::cout << "OK\n";
    return 0;
}

// g++ -std=c++17 -Iinclude src/autograd/tensor.cpp src/autograd/grad_mode.cpp multi_dim.cpp -o multi_dim