# Performance Optimization Roadmap for BLAS Operations

## Issue: Apply Memory-Bound Operation Optimizations to Remaining BLAS Functions

### Background

Recent optimizations to `dot` and `ger` operations demonstrated significant performance improvements (3-4x) through:
- Multi-accumulator unrolling to reduce dependency chains
- Software prefetching to hide memory latency
- Lower multi-threading thresholds for memory-bound operations
- Bug fixes in kernel implementations

**Current Performance Baseline:**
- **dot**: Improved from ~0.3-0.7 GFlops to ~0.8-1.4 GFlops (out-of-cache)
- **ger**: Improved from ~0.24-0.35 GFlops to ~1.04-1.42 GFlops

These operations are fundamentally memory-bandwidth limited, achieving ~3-8 GB/s sustained throughput, which is reasonable given single-threaded sequential access patterns and the low arithmetic intensity of these operations.

### Operations Requiring Optimization

The following operations should receive similar treatment:

#### Level 1 (Vector-Vector Operations)
- [x] `dot` - **COMPLETED** (4-way accumulators, prefetching, MT threshold: 32768)
- [x] `copy` - **COMPLETED** (prefetching in kernels, MT support, MT threshold: 16384)
- [x] `axpy` - **COMPLETED** (4-way accumulators, prefetching, MT threshold: 32768, multi-threading enabled)
- [ ] `scal` - MT support present, but missing prefetching in kernels
- [x] `swap` - **COMPLETED** (prefetching in kernels, MT support)
- [x] `asum` - **COMPLETED** (4-way accumulators, prefetching in kernels)
- [x] `nrm2` - **COMPLETED** (prefetching in kernels, MT support)
- [x] `rot` - **COMPLETED** (prefetching in kernels, MT support)

#### Level 2 (Matrix-Vector Operations)
- [x] `ger` - **COMPLETED** (prefetching, MT threshold: 2048, bug fixes)
- [x] `gemv` - **COMPLETED** (prefetching in kernels, MT support, MT threshold: 4096)

#### Level 3 (Matrix-Matrix Operations)
- [x] `gemm` - **COMPLETED** (MT threshold tuned to 4096, already has blocking and SIMD)

### Optimization Techniques to Apply

#### 1. Multi-Accumulator Unrolling
**Purpose:** Reduce dependency chains, improve instruction-level parallelism

**Implementation:**
```c
// Before: Single accumulator
for (i = 0; i < n; i += 8) {
    sum = _mm256_fmadd_ps(x, y, sum);
}

// After: 4 independent accumulators
for (i = 0; i < n; i += 32) {
    sum0 = _mm256_fmadd_ps(x0, y0, sum0);
    sum1 = _mm256_fmadd_ps(x1, y1, sum1);
    sum2 = _mm256_fmadd_ps(x2, y2, sum2);
    sum3 = _mm256_fmadd_ps(x3, y3, sum3);
}
// Combine accumulators at end
```

**Target operations:** dot, asum, nrm2, axpy, copy, scal, gemv

#### 2. Software Prefetching
**Purpose:** Hide memory latency by prefetching data ahead of use

**Implementation:**
```c
#if defined(CBLAS_PREFETCH)
    const CBLAS_INDEX prefetch_distance = 64; // 256 bytes ahead
    
    if (i + prefetch_distance < n) {
        CBLAS_PREFETCH(x + i + prefetch_distance, 0, 3);  // read
        CBLAS_PREFETCH(y + i + prefetch_distance, 1, 3);  // write
    }
#endif
```

**Parameters:**
- Read-only: `CBLAS_PREFETCH(addr, 0, 3)` - locality level 3 (stay in cache)
- Write: `CBLAS_PREFETCH(addr, 1, 3)` - prepare for write
- Distance: 64-128 elements (256-512 bytes) ahead

**Target operations:** All Level 1 and Level 2 operations

#### 3. Multi-Threading Threshold Tuning
**Purpose:** Enable MT earlier for memory-bound operations where overhead is justified

**Current thresholds:**
```c
#define CBLAS_MT_DOT    32768   // Optimized (was 100000)
#define CBLAS_MT_COPY   10000   // Review needed
#define CBLAS_MT_GER    2048    // Optimized (was 10000)
#define CBLAS_MT_GEMM   10000   // Review needed
#define CBLAS_MT_GEMV   10000   // Review needed
```

**Recommended approach:**
1. Create threshold tuning test for each operation
2. Measure performance at various sizes (4KB to 16MB)
3. Find crossover point where MT overhead < MT benefit
4. Set threshold to ~0.5-0.75x of crossover point

**Target operations:** copy, axpy, scal, gemv (lower thresholds likely beneficial)

#### 4. Bug Fixes and Code Review
**Purpose:** Identify and fix correctness/performance bugs

**Areas to review:**
- Pointer arithmetic in loop increments
- Off-by-one errors in leftover element handling
- Correct use of `incx`/`incy` strides
- Alignment requirements for SIMD loads/stores
- Cache line boundary handling

**Example from ger fix:**
```c
// Bug: xr pointer not incremented correctly
xr = &X(row + i);  // Wrong: sets to row+i each iteration

// Fix: Simple increment
xr++;  // Correct: advances to next element
```

#### 5. Performance Test Harness Updates
**Purpose:** Measure and report realistic performance metrics

**Required changes for each operation:**
1. Initialize test arrays with non-zero values (avoid zero-page optimizations)
2. Report GFlops **and** GB/s bandwidth
3. Test range from L1 cache (4KB) to out-of-cache (16MB+)
4. Add operation-specific notes about arithmetic intensity

**Template:**
```c
// Initialize with non-zero values
for (CBLAS_INDEX i = 0; i < n; i++) {
    x[i] = (float)(i % 100) / 100.0f + 1.0f;
}

// Calculate metrics
float gflops = (float)(flops) / dt / 1e9;
float bytes = /* operation-specific memory traffic */;
float gbytes_per_sec = bytes / dt / 1e9;

printf("%10s %10s %12s %12s\n", "Size", "GFlops", "GB/s", "Time(s)");
printf("%10d %10.2f %12.2f %12.6f\n", size, gflops, gbytes_per_sec, dt);
```

### Implementation Plan

#### Phase 1: Level 1 Memory-Bound Operations (Copy, AXPY, SCAL)
**Priority:** High (simple, high-impact)
**Status:** ✅ MOSTLY COMPLETE (scal missing prefetching)

**Completed:**
- [x] `copy` - Prefetching added to kernels, MT threshold: 16384
- [x] `axpy` - 4-way unrolling, prefetching, MT threshold: 32768
- [ ] `scal` - MT support present, prefetching still needed

**Expected improvements:** 2-3x for out-of-cache workloads

#### Phase 2: Level 1 Reduction Operations (ASUM, NRM2)
**Priority:** Medium (reduction patterns need careful handling)
**Status:** ✅ COMPLETE

**Completed:**
- [x] `asum` - 4-way accumulators in scalar path, prefetching in SIMD kernels
- [x] `nrm2` - Prefetching in kernels, MT support added

**Expected improvements:** 1.5-2x for large vectors

#### Phase 3: Level 2 Operations (GEMV)
**Priority:** High (matrix-vector is common operation)
**Status:** ✅ COMPLETE

**Completed:**
- [x] Prefetching added to gemv kernels (gemv_k_avx.c, gemv_k_neon.c)
- [x] MT threshold lowered to 4096
- [x] Both transpose and non-transpose cases supported

**Expected improvements:** 2-3x, especially for cache-unfriendly access patterns

#### Phase 4: Level 3 Review (GEMM)
**Priority:** Medium (already has blocking and SIMD)
**Status:** ✅ COMPLETE

**Completed:**
- [x] MT threshold tuned to 4096
- [x] Blocking and SIMD optimizations already in place

**Expected improvements:** 10-20% (already well-optimized)

### Testing Requirements

#### Correctness Tests
- [x] All existing unit tests must pass after optimizations (ctest 100% pass rate)
- [ ] Add edge case tests for new code paths (prefetching, unrolling)
- [ ] Verify numerical accuracy is maintained (especially for reductions)
- [ ] Test with various strides (incx=2, incy=3, etc.)
- [ ] Validate alignment handling for unaligned arrays

#### Performance Tests
- [ ] Create/update `<op>_perf.c` for each optimized operation
- [ ] Test size range: 4, 8, 16, ..., up to 16M elements
- [ ] Report both GFlops and GB/s
- [ ] Run on both in-cache and out-of-cache datasets
- [ ] Measure improvement vs baseline (save baseline results)

#### Threshold Tuning Tests
- [ ] Create `<op>_threshold_tuning.c` for operations with MT
- [ ] Test threshold range from 1K to 1M elements
- [ ] Measure single-thread vs multi-thread performance
- [ ] Identify crossover point
- [ ] Update threshold tests to validate new values

#### Regression Tests
- [ ] Run full test suite before and after changes
- [ ] Use CTest to automate test execution
- [ ] Check that no performance regressions occur for small sizes
- [ ] Verify MT overhead is acceptable at threshold boundaries

### Acceptance Criteria

#### Minimum Requirements
- [x] All correctness tests pass (ctest shows 100%)
- [x] Performance improvements documented with before/after numbers
- [ ] No performance regressions for small sizes (< 1KB)
- [ ] Code follows existing patterns and style
- [ ] Comments explain optimization techniques used
- [ ] MT thresholds justified by empirical testing

#### Performance Targets

**Level 1 Operations (per-operation basis):**
- Out-of-cache bandwidth: ≥3 GB/s (memory-limited operations)
- In-cache throughput: Approach L2 bandwidth (~20-30 GB/s)
- Minimum 1.5x improvement over baseline for n > 100K
- MT activation provides measurable benefit at tuned threshold

**Level 2 Operations (GEMV):**
- Matrix-vector multiply: ≥5 GFlops for large matrices (m,n > 4096)
- Cache-friendly access: ≥15 GB/s effective bandwidth
- Minimum 2x improvement over baseline

**Level 3 Operations (GEMM):**
- Peak performance: ≥7 GFlops (already achieved)
- No regressions in existing benchmarks
- Any improvements are bonus (already well-optimized)

#### Documentation
- [ ] Update `.github/copilot-instructions.md` with optimization patterns
- [ ] Document new MT thresholds and rationale in `cblas.h` comments
- [ ] Add performance notes to README.md
- [ ] Create or update `docs/PERFORMANCE.md` with benchmark results

### Priority Ranking

1. **High Priority** (immediate impact, low risk): ✅ ALL COMPLETE
   - `copy` - ✅ Completed (prefetching, MT threshold 16384)
   - `axpy` - ✅ Completed (4-way accumulators, prefetching, MT threshold 32768)
   - `gemv` - ✅ Completed (prefetching, MT threshold 4096)

2. **Medium Priority** (moderate impact, some complexity): ✅ MOSTLY COMPLETE
   - `scal` - ⚠️ In progress (MT support present, missing prefetching)
   - `asum` - ✅ Completed (4-way accumulators, prefetching)
   - `nrm2` - ✅ Completed (prefetching, MT support)

3. **Low Priority** (specialized or already optimized): ✅ ALL COMPLETE
   - `swap` - ✅ Completed (prefetching, MT support)
   - `rot` - ✅ Completed (prefetching, MT support)
   - `gemm` - ✅ Completed (MT threshold 4096, blocking, SIMD)

### References

**Completed Optimizations:**
- `dot.c` - 4-way accumulator unrolling, prefetching (lines 13-107)
- `ger.c` - Prefetching, MT threshold tuning, bug fixes (lines 638-695)
- `axpy.c` - Multi-threading support, 4-way accumulator unrolling, prefetching, MT threshold: 32768
- `copy.c` - Prefetching in kernels (copy_k_avx.c, copy_k_neon.c), MT threshold: 16384
- `swap.c` - Prefetching in kernels (swap_k_avx.c, swap_k_neon.c), MT support
- `asum.c` - 4-way accumulators in scalar path, prefetching in SIMD kernels
- `nrm2.c` - Prefetching in kernels (nrm2_k_neon.c, nrm2_k_sse.c), MT support
- `rot.c` - Prefetching in kernels (rot_k_sse.c, rot_k_neon.c), MT support
- `gemv.c` - Prefetching in kernels (gemv_k_avx.c, gemv_k_neon.c), MT threshold: 4096
- `gemm.c` - MT threshold tuned to 4096
- `cblas.h` - Updated MT thresholds (lines 63-68)
- `test_dot_threshold.c` - Updated threshold validation (lines 128-141)

**Performance Results:**
- See `dot_perf` output: ~1.4 GFlops, ~5.6 GB/s at 8192 elements
- See `ger_perf` output: ~1.35 GFlops, ~5.4 GB/s at 8192×8192
- `axpy_perf` output: ~5.5-5.7 GFlops, ~33-34 GB/s for large vectors (> 1M elements)
  - Multi-threading activates for n > 32768
  - Performance improvements observed: ~2-3x for out-of-cache workloads with MT enabled

**Configuration:**
- Prefetch distance: 128 elements (512 bytes) - `cblas.h:71`
- Prefetch threshold: 100K elements - `cblas.h:70`
- Prefetch macro: `CBLAS_PREFETCH(addr, rw, locality)` - `cblas.h:75-80`

---

**Status:** Mostly Complete (scal missing prefetching)
**Assignee:** TBD
**Labels:** performance, optimization, enhancement
**Milestone:** v0.26
