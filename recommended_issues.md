# Performance Optimization Recommendations

**Priority: HIGH = 10%+ gain, MEDIUM = 2-10%, LOW = <2%**  
**Risk: HIGH = algorithmic changes, MEDIUM = ISA-specific, LOW = incremental additions**

---

## 🔥 PRIORITY 1: High-Impact SIMD Opportunities (HIGH priority, MEDIUM risk)

### 1.1 Add SIMD to Level-1 Operations (15-40% speedup)
**Impact:** HIGH | **Risk:** MEDIUM | **Effort:** 2-3 days

**Missing SIMD implementations:**
- `axpy.c` - No SIMD at all (scalar loop only)
- `scal.c` - No SIMD at all (scalar loop only)
- `dot.c` - No SIMD (only 4-way unrolling)
- `swap.c` - Likely no SIMD (not examined, but probable)

**Why this matters:**
- AXPY and SCAL are commonly used in higher-level algorithms
- 4-8x throughput improvement with SSE/AVX on x86
- 4x throughput improvement with NEON on ARM
- Low risk: straightforward vectorization patterns

**Implementation approach:**
```c
// Example for saxpy with AVX2 (8 floats at once)
for (; i + 8 <= n; i += 8) {
    __m256 xvec = _mm256_loadu_ps(&x[i]);
    __m256 yvec = _mm256_loadu_ps(&y[i]);
    __m256 alphavec = _mm256_set1_ps(alpha);
    yvec = _mm256_fmadd_ps(alphavec, xvec, yvec);  // y = alpha*x + y
    _mm256_storeu_ps(&y[i], yvec);
}
```

**Files to modify:**
- [axpy.c](axpy.c) - Add SSE/AVX/NEON paths
- [scal.c](scal.c) - Add SSE/AVX/NEON paths  
- [dot.c](dot.c) - Add SSE/AVX horizontal sum reduction
- [swap.c](swap.c) - Add SIMD if not present

**Expected gains:**
- **x86 SSE:** 4x throughput (128-bit registers, 4 floats)
- **x86 AVX2:** 8x throughput (256-bit registers, 8 floats)
- **ARM NEON:** 4x throughput (128-bit registers, 4 floats)

---

### 1.2 Add AVX-512 Support (20-50% over AVX2)
**Impact:** HIGH | **Risk:** MEDIUM | **Effort:** 3-5 days

**Current state:** AVX512 detected in [cpuid_x64.c](cpuid_x64.c#L177) but never used

**What to add:**
- `_mm512_*` intrinsics for 16 floats/8 doubles per operation
- Kernel variants: `AddDot8x8_AVX512`, `AddProd16x4_AVX512`
- Runtime dispatch in [util.c](util.c) based on `CPU_AVX512` flag

**Target operations:**
1. **GEMM** - 16-float micro-kernels (8x8 or 16x4 blocks)
2. **GER** - 16-column outer products
3. **DOT** - 16-way parallel multiply-accumulate
4. **AXPY/SCAL** - 16-element vector ops

**Why this is valuable:**
- Modern servers (Ice Lake+, Zen 4+) have AVX-512
- Doubles register width vs AVX2 (512-bit vs 256-bit)
- Better for large problems (>1000 elements)

**Implementation notes:**
- Use `_mm512_loadu_ps()` for unaligned loads
- Use `_mm512_reduce_add_ps()` for horizontal sum in DOT
- Guard with `#if defined(__AVX512F__)`

---

### 1.3 Optimize Double-Precision Paths (10-30% speedup)
**Impact:** MEDIUM | **Risk:** LOW | **Effort:** 2 days

**Current state:** Most `d*` functions (ddot, dgemm, etc.) have **no SIMD**

**Missing:**
- `ddot` - Scalar loop only ([dot.c](dot.c#L174-L191))
- `dgemm` - No implementation at all
- `dger` - Likely scalar
- `daxpy` - Scalar ([axpy.c](axpy.c#L44-L67))

**Quick wins:**
```c
// ddot with AVX2 (4 doubles at once)
__m256d sum_vec = _mm256_setzero_pd();
for (; i + 4 <= n; i += 4) {
    __m256d xvec = _mm256_loadu_pd(&x[i]);
    __m256d yvec = _mm256_loadu_pd(&y[i]);
    sum_vec = _mm256_fmadd_pd(xvec, yvec, sum_vec);  // sum += x * y
}
// Horizontal reduction at end
```

**Why this matters:**
- Scientific computing often uses double precision
- 4x speedup with AVX2, 8x with AVX-512
- Similar effort to single-precision paths

---

## 🎯 PRIORITY 2: Cache Blocking Tuning (5-15% gain, LOW risk)

### 2.1 Profile and Optimize GEMM Block Sizes
**Impact:** MEDIUM | **Risk:** LOW | **Effort:** 1-2 days

**Current parameters** in [gemm.c](gemm.c#L19-L21):
```c
#define mc 256   // Rows of A to pack
#define kc 128   // Inner dimension
#define nb 1024  // Columns of B to pack
```

**Issues:**
1. **Hardcoded for unknown CPU** - No tuning per architecture
2. **L2 cache not considered** - Current sizes designed for L1 only
3. **No validation** - May not fit in cache on all platforms

**Optimal sizing calculation:**
```
L1d cache (typical): 32 KB per core
L2 cache (typical): 256-512 KB per core

Packed A: mc × kc × 4 bytes = 256 × 128 × 4 = 128 KB
Packed B: kc × nb × 4 bytes = 128 × 1024 × 4 = 512 KB
Total: 640 KB (exceeds L1, fits in L2)
```

**Recommendations:**
1. **Query L1/L2 cache sizes** at runtime ([cpuid_x64.c](cpuid_x64.c#L98-L117) already has this)
2. **Adjust block sizes** based on cache:
   ```c
   // For 32KB L1 data cache:
   mc = 128, kc = 256, nb = 256  // ~256KB total for L2
   
   // For 64KB L1 data cache:
   mc = 256, kc = 256, nb = 512  // ~512KB total for L2
   ```
3. **Profile on target CPUs:**
   - Intel Core (32KB L1d): Test mc=128, kc=256
   - AMD Zen (32KB L1d): Test mc=128, kc=256
   - Apple M-series (128KB L1d): Test mc=512, kc=256
   - ARM Cortex (64KB L1d): Test mc=256, kc=256

**Expected gains:**
- 5-15% on large matrices (>1000x1000)
- 20-30% on Apple Silicon due to large L1
- Minimal gain on small matrices (<512x512)

---

### 2.2 Add Cache-Aware Level-2 Blocking
**Impact:** MEDIUM | **Risk:** LOW | **Effort:** 1 day

**Current state:** GER and GEMV don't use cache blocking

**Recommendation for GEMV:**
```c
// Current: processes entire column at once (poor locality)
// Better: block into cache-friendly chunks
#define GEMV_BLOCK_SIZE 256  // Fits in L1

for (CBLAS_INDEX i = 0; i < m; i += GEMV_BLOCK_SIZE) {
    CBLAS_INDEX ib = MIN(GEMV_BLOCK_SIZE, m - i);
    // Process ib rows at a time
}
```

**Expected gain:** 5-10% on large matrices (>2000 elements)

---

## ⚡ PRIORITY 3: Memory Prefetching (2-8% gain, LOW risk)

### 3.1 Enable Prefetching in GEMM
**Impact:** MEDIUM | **Risk:** LOW | **Effort:** 1 day

**Current state:** Prefetch code exists but is **commented out** in [gemm.c](gemm.c#L162-L167):
```c
// __builtin_prefetch(&C(0,0), 0);
// __builtin_prefetch(&C(0,1), 0);
// ...
```

**Why it's disabled:** Likely not benchmarked or caused regression on some CPU

**Action items:**
1. **Re-enable and test** on modern CPUs (2020+)
2. **Add prefetch distance tuning:**
   ```c
   #define PREFETCH_DISTANCE 8  // Prefetch 8 iterations ahead
   
   for (CBLAS_INDEX p = 0; p < k; p++) {
       if (p + PREFETCH_DISTANCE < k) {
           __builtin_prefetch(&packedA[(p + PREFETCH_DISTANCE) * 4], 0, 3);
           __builtin_prefetch(&packedB[(p + PREFETCH_DISTANCE) * 4], 0, 3);
       }
       // ... compute ...
   }
   ```
3. **Prefetch C matrix before write:**
   ```c
   __builtin_prefetch(&C(0,0), 1, 3);  // Write prefetch
   ```

**Expected gains:**
- 3-8% on large GEMM (>512x512)
- Minimal on small matrices (data already in cache)
- Best on CPUs with weak hardware prefetchers

---

### 3.2 Add Prefetching to Level-1 Operations
**Impact:** LOW | **Risk:** LOW | **Effort:** 0.5 days

**Target:** DOT, AXPY, COPY for large vectors (>100K elements)

```c
// Example for AXPY
for (CBLAS_INDEX i = 0; i < n; i++) {
    if (i + 64 < n) {  // Prefetch 64 elements ahead
        __builtin_prefetch(&x[i + 64], 0, 0);
        __builtin_prefetch(&y[i + 64], 1, 0);
    }
    y[i] = alpha * x[i] + y[i];
}
```

**Expected gain:** 2-5% on very large vectors, negligible on small

---

## 🏗️ PRIORITY 4: Architecture Improvements (MEDIUM priority, MEDIUM risk)

### 4.1 Add Auto-Tuning Infrastructure
**Impact:** MEDIUM | **Risk:** MEDIUM | **Effort:** 3-5 days

**Problem:** Hardcoded thresholds don't adapt to different CPUs

**Current thresholds** in [cblas.h](cblas.h#L64-L68):
```c
#define CBLAS_MT_DOT    10000
#define CBLAS_MT_COPY   10000
#define CBLAS_MT_GER    10000
#define CBLAS_MT_GEMM   10000
```

**Solution:** Runtime calibration
1. On first init, run micro-benchmarks
2. Measure single-thread vs multi-thread crossover
3. Store in static variables or config file
4. Adjust based on actual core count

**Expected gain:** 5-10% by avoiding unnecessary threading overhead

---

### 4.2 Improve GEMM Kernel Structure
**Impact:** MEDIUM | **Risk:** HIGH | **Effort:** 5-7 days

**Current limitations:**
1. **Fixed 4x4 micro-kernel** - Not optimal for all CPUs
2. **No register blocking** beyond 4x4
3. **Packed buffers on stack** - Limited by stack size

**Recommendations:**
1. **Platform-specific kernels:**
   - x86 AVX2: 6x16 or 8x8 kernels
   - x86 AVX-512: 8x16 or 16x8 kernels
   - ARM NEON: 8x8 or 4x12 kernels
2. **Multiple kernel sizes** - dispatch based on problem size
3. **Heap allocation for large packed buffers**

**Expected gain:** 20-40% on large GEMM, but HIGH RISK of regression

---

## 📊 Summary Table

| Optimization | Impact | Effort | Risk | Priority | Est. Gain |
|--------------|--------|--------|------|----------|-----------|
| Add SIMD to AXPY/SCAL | HIGH | 2-3 days | MEDIUM | 1 | 15-40% |
| AVX-512 support | HIGH | 3-5 days | MEDIUM | 1 | 20-50% |
| Double-precision SIMD | MEDIUM | 2 days | LOW | 1 | 10-30% |
| GEMM block tuning | MEDIUM | 1-2 days | LOW | 2 | 5-15% |
| Enable prefetching | MEDIUM | 1 day | LOW | 3 | 3-8% |
| Auto-tuning thresholds | MEDIUM | 3-5 days | MEDIUM | 4 | 5-10% |
| Advanced GEMM kernels | MEDIUM | 5-7 days | HIGH | 4 | 20-40% |

---

## 🚀 Recommended Implementation Order

### Phase 1 (Week 1): Quick Wins
1. Add SIMD to AXPY and SCAL (2 days)
2. Add SIMD to DOT (1 day)
3. Re-enable and tune GEMM prefetching (1 day)
4. Test suite validation (1 day)

**Expected combined gain:** 20-50% on Level-1 operations

### Phase 2 (Week 2): Double Precision
1. Add SIMD to ddot, daxpy (2 days)
2. Implement dgemm with SIMD (3 days)

**Expected gain:** 15-35% on double-precision workloads

### Phase 3 (Week 3): Advanced Features
1. AVX-512 kernels for GEMM (3 days)
2. Cache blocking tuning (2 days)

**Expected gain:** 10-25% on large matrices with AVX-512 CPUs

### Phase 4 (Optional): Infrastructure
1. Auto-tuning system (5 days)
2. Advanced GEMM micro-kernels (7 days)

**Expected gain:** 15-30% but higher risk

---

## 🔍 Validation Strategy

After each change:
1. Run `make test` - ensure correctness
2. Run `./gemm_perf` - measure GEMM performance
3. Run `./dot_perf` - measure Level-1 performance
4. Compare against baseline on multiple CPUs
5. Check for regressions on edge cases (n=1, n=3, strided access)

---

## 📝 Notes

**Why not focus on Level-2 (GER/GEMV)?**
- Already have some SIMD ([ger.c](ger.c#L127-L248) has 4x4 SIMD kernel)
- Less commonly used than GEMM or Level-1
- Gains would be 5-15% but more complex

**Why prioritize single-precision over double?**
- Current codebase emphasis (more SP kernels exist)
- ML/AI workloads increasingly use SP
- But DP is still important for science - hence Phase 2

**Why is AVX-512 MEDIUM risk?**
- Some CPUs downclock when using AVX-512
- May hurt more than help on older Skylake-X
- Need careful benchmarking per CPU generation
- But on Ice Lake+/Zen 4+, it's a clear win
