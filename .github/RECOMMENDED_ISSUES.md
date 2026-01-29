# Recommended Issues for CBLAS

This file contains tracking issues for codebase improvements. Copy each issue to GitHub Issues as needed.

---

## 🧪 Testing Improvements

### Issue #1: Add numerical accuracy tests
**Priority:** Medium  
**Labels:** testing, quality

**Description:**
Current tests only check exact equality. Need tests comparing against reference implementations for:
- Accumulated rounding errors in large operations
- Edge cases (denormals, infinities, NaN)
- Comparison vs OpenBLAS or reference BLAS

**Implementation:**
Create `test_accuracy.c` with:
- Large matrix operations (n > 10000)
- Known difficult cases (ill-conditioned matrices)
- Comparison thresholds based on operation complexity

**Acceptance Criteria:**
- [ ] Create test_accuracy.c
- [ ] Define acceptable error bounds (e.g., n * epsilon)
- [ ] Test GEMM accuracy for various matrix sizes
- [ ] Test DOT accuracy with large vectors
- [ ] Document expected error characteristics

---

### Issue #2: Add thread safety tests
**Priority:** Medium  
**Labels:** testing, threading, concurrency

**Description:**
No tests verify thread safety when multiple threads call BLAS functions simultaneously.

**Test Scenarios:**
- Multiple threads calling different BLAS operations
- Multiple threads modifying cblas_set_num_threads()
- Race conditions in work queue management
- Thread local storage issues

**Implementation:**
Create `test_concurrent.c` with:
- Spawn N threads, each calling BLAS operations
- Stress test cblas_init()/cblas_shutdown() cycles
- Test thread safety of cblas_set_num_threads()

**Acceptance Criteria:**
- [ ] Create test_concurrent.c
- [ ] Run with ThreadSanitizer (TSAN)
- [ ] Test with 2, 4, 8, 16 concurrent threads
- [ ] No data races detected

---

### Issue #3: Fix memory leaks in test suite
**Priority:** Medium  
**Labels:** testing, memory-leak

**Description:**
test.c and test_stress.c allocate large arrays (sbig_ones, dbig_ones) but never call cblas_shutdown() or free memory.

**Leaks:**
```c
sbig_ones = svec_fill(BIG_ARRAY, 1.0f);   // Never freed
sbig_zeroes = svec_fill(BIG_ARRAY, 0.0f); // Never freed
```

**Fix:**
Add cleanup in test teardown and call cblas_shutdown().

**Acceptance Criteria:**
- [ ] Add cleanup functions to test.c
- [ ] Call cblas_shutdown() in main()
- [ ] Run Valgrind with --leak-check=full
- [ ] Zero reported leaks

---

## 🏗️ Code Quality

### Issue #4: Separate platform-specific code
**Priority:** Low  
**Labels:** refactoring, architecture

**Description:**
Platform-specific code is scattered throughout the codebase. Create abstraction layer.

**Proposed Structure:**
```
platform/
  threading.h   - Abstract pthread/Win32 threading
  simd.h        - SIMD intrinsic wrappers
  cpuid.h       - CPU detection interface
```

**Benefits:**
- Easier to add new platforms (WebAssembly, etc.)
- Cleaner conditional compilation
- Better testability

**Acceptance Criteria:**
- [ ] Create platform/ directory structure
- [ ] Move threading code to platform/threading.h
- [ ] Move SIMD code to platform/simd.h
- [ ] Update build systems
- [ ] All platforms build successfully

---

## 📚 Documentation

### Issue #5: Document cache-blocking strategy
**Priority:** Low  
**Labels:** documentation, performance

**Description:**
gemm.c uses specific tile sizes (mc=256, kc=128, nb=1024) with no explanation.

**Add Comments Explaining:**
- Why these specific sizes?
- Relationship to L1/L2 cache sizes
- How to tune for different architectures
- Tradeoffs between tile sizes

**Acceptance Criteria:**
- [ ] Add inline comments to gemm.c
- [ ] Document cache blocking algorithm
- [ ] Provide tuning guidelines
- [ ] Reference GotoBLAS paper

---

## 🔧 Build System

### Issue #6: Add CMake configuration options
**Priority:** Medium  
**Labels:** enhancement, build

**Description:**
Configuration flags are hardcoded #defines in cblas.h. Make them CMake options.

**Proposed Options:**
```cmake
option(CBLAS_ENABLE_MT "Enable multi-threading" ON)
option(CBLAS_USE_SIMD "Enable SIMD optimizations" ON)
option(CBLAS_CHECK_INPUTS "Enable input validation" ON)
option(CBLAS_USE_STATIC_BUFFERS "Use static buffers" ON)
set(CBLAS_MAX_THREADS "64" CACHE STRING "Maximum threads")
```

**Generate:** `cblas_config.h` from CMake

**Acceptance Criteria:**
- [ ] Add CMake options
- [ ] Generate cblas_config.h
- [ ] Update cblas.h to include config
- [ ] Document options in README
- [ ] Test various configurations

---

### Issue #7: Add CI performance benchmarks
**Priority:** Low  
**Labels:** ci, performance

**Description:**
GitHub Actions only builds and tests. Add performance regression detection.

**Implementation:**
- Run gemm_perf, dot_perf, ger_perf in CI
- Store results as artifacts
- Compare against baseline
- Fail CI if performance drops >10%

**Acceptance Criteria:**
- [ ] Add perf benchmarks to CI workflow
- [ ] Store baseline performance data
- [ ] Implement regression detection
- [ ] Report GFlops in CI logs

---

### Issue #8: Remove disabled code blocks
**Priority:** Low  
**Labels:** code-quality, cleanup

**Description:**
Several files have `#if 0` blocks with old SIMD implementations that are confusing.

**Files:**
- ger.c:32-86 - Disabled SIMD code that's reimplemented below

**Action:**
Delete or document why code is disabled. If experimental, move to separate branch.

**Acceptance Criteria:**
- [ ] Review all #if 0 blocks
- [ ] Delete obsolete code
- [ ] Document reasons if kept
- [ ] No functional changes

---

## 📊 Observability

### Issue #9: Add performance counters
**Priority:** Low  
**Labels:** enhancement, observability

**Description:**
Add runtime statistics for monitoring BLAS usage patterns.

**API Design:**
```c
typedef struct {
    uint64_t total_calls;
    uint64_t total_elements;
    uint64_t mt_activations;
    double total_time_sec;
} cblas_stats_t;

cblas_stats_t* cblas_get_stats(const char* operation);
void cblas_reset_stats();
void cblas_print_stats();
```

**Use Cases:**
- Profile which operations dominate
- Detect inefficient usage patterns
- Verify MT is activating appropriately

**Acceptance Criteria:**
- [ ] Add stats tracking structure
- [ ] Instrument all BLAS operations
- [ ] Add query API
- [ ] Add stats to cblas_print_configuration()
- [ ] Minimal performance overhead (<1%)

---

### Issue #10: Improve MT debug output
**Priority:** Low  
**Labels:** enhancement, debugging, threading

**Description:**
Current MT_DEBUG only prints basic info. Enhance for troubleshooting.

**Add:**
- Thread ID tracking
- Work queue depth monitoring
- Load imbalance detection
- Execution time per thread
- JSON output mode for parsing

**Implementation:**
```c
MT_TRACE_THREAD(tid, "Processing work item %d/%d", current, total);
MT_TRACE_TIMING(tid, operation, duration_us);
MT_TRACE_QUEUE_DEPTH(depth);
```

**Acceptance Criteria:**
- [ ] Add structured tracing macros
- [ ] Track per-thread timing
- [ ] Detect load imbalance (>20% variance)
- [ ] Add JSON output option
- [ ] Document debug workflow

---

## Summary

**High Priority:** Issue #1  
**Medium Priority:** Issues #2, #3, #6  
**Low Priority:** Issues #4, #5, #7, #8, #9, #10
