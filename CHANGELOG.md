## [0.2.0] — 2025-08-19

### Added
- **Up to 5-D tensors** with **64-bit** sizes/strides (`int64_t`): safer for large arrays.
- **View ops**: `as_strided`, `transpose_view`, `slice_view`, and `select(dim, index)`.
- **Python-like indexing**: `operator[](i)` returns a view (chainable: `a[0][1]`), ending at a scalar view.
- **Scalar support** (rank-0): `is_scalar()`, `item()`, `item_ref()`; `item()` now also works for any 1-element tensor (e.g., shapes `{}`, `{1}`, `{1,1}`, …).
- **Reshape family**: `view(new_shape)` (no copy, contiguous required), `reshape(new_shape)` (copies if needed), `contiguous()`.
- **Pretty printing**: `print_limited(max_per_dim)` for compact multi-D dumps.
- **Shape utils**: `sum_to_shape` migrated to 64-bit and supports scalar targets.
- **Constructors**: `initializer_list<int64_t>` + forwarding `initializer_list<int>` so `{2,3,4}` “just works”.
- **MNIST example** updated: loads as `[N,1,28,28]`, flattens to `[N,784]`, prints **loss + accuracy**.

### Changed / Breaking
- `operator[]` no longer returns the old 2-D row helper; it now returns a **Tensor view** for Python-style chaining.
  - To mutate a single element from a chain, use `.item_ref()` (e.g., `a[1][2][3].item_ref() = 9;`).
  - For the previous 2-D row helper behavior, use `select(0,i)` (or add `row_slice(i)` as a dedicated API).
- Constructors prefer `initializer_list` overloads; brace-init now maps to 64-bit shapes.

### Fixed / Internal
- `backward()` & gradient seeding now work for **scalar outputs** (rank-0).
- Iteration helpers and stride builders handle **rank-0** and large shapes.
