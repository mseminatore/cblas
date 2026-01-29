# CBLAS Threading Architecture

## Overview

CBLAS uses a work queue-based multi-threading system to parallelize large BLAS operations. When an operation exceeds a predefined threshold, work is automatically distributed across worker threads to improve performance on multi-core systems.

## Threading Components

### 1. Thread Server

The thread server manages a pool of persistent worker threads that process BLAS operations in parallel. The server is initialized automatically when `cblas_init()` is called with more than one thread.

**Key Functions:**
- `cblas_init(int threads)` - Initialize the library with specified thread count (use `CBLAS_DEFAULT_THREADS` for auto-detection)
- `cblas_shutdown()` - Gracefully terminate all worker threads and cleanup resources
- `cblas_set_num_threads(int threads)` - Dynamically adjust the number of active threads
- `cblas_get_num_threads()` - Query the current number of active threads

**Platform Implementations:**
- **POSIX systems** (Linux, macOS): `server.c` using pthreads
- **Windows**: `server_win32.c` using Win32 threads

### 2. Work Queue System

Operations are parallelized by dividing work into partitions and dispatching them to worker threads via a work queue.

**Key Structures:**

```c
// Work queue item representing a single task
typedef struct work_queue_t {
    struct work_queue_t* next;  // Next task in queue
    cblas_args_t* args;         // Parameters for kernel function
    int type;                   // Type of call
    kernel_function kernel;      // Function to execute
    volatile int finished;       // Completion flag
    int thread_num, tid;        // Thread tracking
} work_queue_t;

// Arguments passed to kernel functions
typedef struct {
    CBLAS_INDEX m, n, k, incx, incy, lda, ldb, ldc, ib, pb;
    void *x, *y, *c, *alpha, *beta, *a, *b;
} cblas_args_t;
```

**Queue Operations:**
- `cblas_execute(items, queue)` - Synchronously execute a work queue (main thread participates)
- `cblas_execute_async(items, queue)` - Submit tasks to worker threads asynchronously
- `cblas_execute_async_join(items, queue)` - Wait for asynchronous tasks to complete

### 3. Work Dispatch

BLAS operations use dispatch functions to partition work across threads:

- `cblas_level1_exec()` - Dispatches Level-1 operations (vector-vector)
- `cblas_level1_exec_result()` - Dispatches Level-1 operations that return a value (e.g., dot product)

**Work Partitioning Algorithm:**

For N elements distributed across T threads, each thread receives approximately N/T elements. The partitioning uses a round-robin approach that balances work evenly:

```c
for (i = 0; i < thread_count; i++) {
    partition_size = (n + thread_count - i - 1) / (thread_count - i);
    // Assign partition_size elements to thread i
    n -= partition_size;
}
```

This ensures that when N is not evenly divisible by T, the first few threads get one extra element.

## Multi-Threading Thresholds

Operations only activate multi-threading when the problem size exceeds a threshold. This avoids thread overhead for small operations where sequential execution is faster.

**Default Thresholds** (defined in `cblas.h`):

```c
#define CBLAS_MT_DOT    10000  // cblas_sdot, cblas_ddot
#define CBLAS_MT_COPY   10000  // cblas_scopy, cblas_dcopy, etc.
#define CBLAS_MT_GER    10000  // cblas_sger, cblas_dger
#define CBLAS_MT_GEMM   10000  // cblas_sgemm, cblas_dgemm
#define CBLAS_MT_GEMV   10000  // cblas_sgemv, cblas_dgemv
```

**Tuning Guidelines:**

- **Increase thresholds** if you see slowdowns on small operations (reduce threading overhead)
- **Decrease thresholds** if you have very fast cores and want earlier parallelization
- Optimal thresholds depend on:
  - CPU core performance
  - Memory bandwidth
  - Cache hierarchy
  - Thread synchronization overhead

**Example threshold calculation for dot product:**

Threading overhead is typically 1-5 microseconds. For single-precision dot product:
- Throughput: ~4 GFlops/core (2 ops/element: multiply + add)
- Cost per element: ~0.5 nanoseconds
- Break-even point: overhead / cost_per_element = 2000-10000 elements

Thresholds can be modified by editing `cblas.h` and recompiling.

## Performance Characteristics

### Thread Overhead

Multi-threading introduces overhead from:
1. **Work queue management** (~1-2 microseconds)
2. **Thread synchronization** (~1-3 microseconds per barrier)
3. **Cache effects** (false sharing, cache line migration)

For operations below the MT threshold, sequential execution avoids this overhead.

### Scaling Efficiency

Performance scaling depends on:
- **Problem size**: Larger problems scale better (less overhead, more work per thread)
- **Memory bandwidth**: Level-1 operations (copy, axpy, dot) are often memory-bound
- **Cache efficiency**: Level-3 operations (gemm) use cache blocking to reduce bandwidth pressure
- **Core count**: Diminishing returns beyond available memory bandwidth

**Expected Speedup:**

| Operation | Problem Size | Threads | Typical Speedup |
|-----------|--------------|---------|-----------------|
| dot       | 100K         | 4       | 2.5-3.5x        |
| dot       | 1M           | 4       | 3.0-3.8x        |
| gemm      | 512x512      | 4       | 3.5-3.9x        |
| gemm      | 2048x2048    | 4       | 3.8-4.0x        |

Speedups are relative to single-threaded execution. Memory-bound operations (dot, copy) scale worse than compute-bound operations (gemm).

### Cache Blocking (Level-3 Operations)

GEMM uses cache-aware blocking to maximize data reuse:

```c
#define mc 256    // Rows of A in L2 cache
#define kc 128    // Columns of A / Rows of B
#define nb 1024   // Columns of B in L2 cache
```

These tile sizes are tuned for typical L1/L2 cache hierarchies. Each thread processes independent tiles to minimize cache interference.

## Thread Safety

### Thread-Safe Operations

All BLAS functions are **thread-safe** when:
- Different threads operate on **non-overlapping data**
- No simultaneous calls to `cblas_set_num_threads()` during computation

**Safe concurrent usage:**
```c
// Safe: Different threads, different matrices
#pragma omp parallel sections
{
    #pragma omp section
    cblas_sgemm(..., A1, B1, C1);
    
    #pragma omp section
    cblas_sgemm(..., A2, B2, C2);
}
```

### NOT Thread-Safe

- **Overlapping data**: Multiple threads writing to the same output array
- **Configuration changes**: Calling `cblas_set_num_threads()` while operations are in progress
- **Initialization/shutdown**: Calling `cblas_init()` or `cblas_shutdown()` during computation

**Unsafe concurrent usage:**
```c
// UNSAFE: Both threads write to same output C
cblas_sgemm(..., A1, B1, C);  // Thread 1
cblas_sgemm(..., A2, B2, C);  // Thread 2 - RACE CONDITION!
```

### Internal Thread Safety

The work queue uses platform-specific synchronization:
- **POSIX**: `pthread_mutex_t` and `pthread_cond_t`
- **Windows**: `CRITICAL_SECTION` and Events

Worker threads safely dequeue tasks and update completion flags using:
- Mutex-protected queue operations
- Volatile completion flags with busy-wait polling
- Memory barriers (implicit in synchronization primitives)

## Disabling Multi-Threading

### Compile-Time Disable

Comment out the `MT_ENABLED` flag in `cblas.h` and recompile:

```c
// Disable multi-threading
// #define MT_ENABLED
```

This completely removes threading code from the library.

### Runtime Disable

Force single-threaded execution without recompiling:

```c
// Option 1: Initialize with 1 thread
cblas_init(1);

// Option 2: Set threads to 1 after initialization
cblas_init(CBLAS_DEFAULT_THREADS);
cblas_set_num_threads(1);
```

This keeps the threading infrastructure but forces all operations to run sequentially on the main thread.

### Per-Operation Disable

Modify thresholds to extremely large values to disable threading for specific operations:

```c
// In cblas.h
#define CBLAS_MT_DOT    INT_MAX  // Never use MT for dot product
#define CBLAS_MT_GEMM   10000    // Still use MT for gemm
```

## Debugging Threading Issues

### Enable Debug Tracing

Uncomment `MT_DEBUG` in `cblas.h` and recompile:

```c
#define MT_DEBUG
```

This enables detailed logging with enhanced diagnostics:
```
set threads = 4
[Thread 0] created.
[Thread 1] created.
[Thread 2] created.
adding 3 items to the queue.
[Queue] Depth: 3
waking worker threads.
[Thread 1] executing a task.
[Thread 0] executing a task.
[Thread 2] executing a task.
[Thread 1] task took 245.32 us
[Thread 1] task completed.
[Thread 0] task took 248.15 us
[Thread 0] task completed.
[Thread 2] task took 251.78 us
[Thread 2] task completed.
[Load Balance] OK: 2.6% variance (min=245.32us, max=251.78us, avg=248.42us)
...
```

### Enhanced Debug Macros

When `MT_DEBUG` is enabled, the following structured tracing macros are available:

- **MT_TRACE_THREAD(tid, ...)** - Thread-specific logging with thread ID prefix
- **MT_TRACE_TIMING(tid, op, duration_us)** - Per-thread execution time tracking (microseconds)
- **MT_TRACE_QUEUE_DEPTH(depth)** - Work queue depth monitoring
- **MT_TRACE_LOAD_BALANCE(thread_count, times)** - Automatic load imbalance detection

**Load Balance Detection:**
- Calculates variance between fastest and slowest thread
- Warns if variance exceeds 20% (indicates imbalanced work distribution)
- Reports min/max/average execution times

**Example Output:**
```
[Load Balance] WARNING: 45.2% variance (min=100.50us, max=300.25us, avg=206.84us)
[Load Balance] OK: 8.3% variance (min=195.12us, max=212.45us, avg=203.91us)
```

### Common Issues

**1. Slowdown with small matrices**
- Cause: Threading overhead exceeds computation cost
- Solution: Increase MT thresholds or reduce thread count

**2. Poor scaling beyond 4-8 threads**
- Cause: Memory bandwidth saturation
- Solution: Expected for memory-bound operations; use larger problems or compute-bound operations

**3. Hangs during shutdown**
- Cause: Improper thread cleanup
- Solution: Always call `cblas_shutdown()` before program exit

**4. Incorrect results**
- Cause: Possible stride bug or race condition
- Solution: Verify with single-threaded execution (`cblas_set_num_threads(1)`) and enable debug tracing

**5. High load imbalance**
- Cause: Work partition sizes vary significantly across threads
- Detection: Check for `WARNING` messages in load balance output
- Solution: Adjust problem size or thread count; investigate work distribution algorithm

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      CBLAS Application                       │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    BLAS API Function                         │
│              (e.g., cblas_sgemm, cblas_sdot)                 │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │  Size Check   │
                    │ n > threshold?│
                    └───────┬───────┘
                            │
              ┌─────────────┴─────────────┐
              │ YES                       │ NO
              ▼                           ▼
    ┌──────────────────┐        ┌─────────────────┐
    │ cblas_level1_exec│        │  Direct Kernel  │
    │  (Multi-threaded)│        │    Execution    │
    └────────┬─────────┘        └─────────────────┘
             │
             ▼
    ┌──────────────────┐
    │ Partition Work   │
    │ into N chunks    │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │  Build Work      │
    │  Queue Items     │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │ cblas_execute()  │
    │  - Main thread   │
    │    runs chunk 0  │
    │  - Submit chunks │
    │    1..N to queue │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────────────────────────────────┐
    │           Work Queue (Mutex-Protected)       │
    │  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐     │
    │  │Task 1│→ │Task 2│→ │Task 3│→ │ NULL │     │
    │  └──────┘  └──────┘  └──────┘  └──────┘     │
    └────────┬──────────────────────────────────┬──┘
             │                                  │
             ▼                                  ▼
    ┌─────────────────┐              ┌─────────────────┐
    │ Worker Thread 0 │              │ Worker Thread N │
    │                 │              │                 │
    │ 1. Dequeue task │              │ 1. Dequeue task │
    │ 2. Execute      │     ...      │ 2. Execute      │
    │    kernel(args) │              │    kernel(args) │
    │ 3. Mark done    │              │ 3. Mark done    │
    └─────────────────┘              └─────────────────┘
             │                                  │
             └──────────────┬───────────────────┘
                            ▼
                  ┌──────────────────┐
                  │ cblas_execute_   │
                  │   async_join()   │
                  │ Wait for all     │
                  │ tasks finished   │
                  └──────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │    Return     │
                    └───────────────┘
```

## Best Practices

1. **Initialize once**: Call `cblas_init()` at program startup and `cblas_shutdown()` at exit
2. **Right-size thread count**: Use `CBLAS_DEFAULT_THREADS` or match your CPU core count
3. **Batch operations**: Group multiple BLAS calls together to amortize threading overhead
4. **Profile first**: Measure performance before tuning thresholds
5. **Use large problems**: Multi-threading benefits diminish for small matrices/vectors
6. **Avoid dynamic changes**: Don't call `cblas_set_num_threads()` during heavy computation

## Example Usage

```c
#include "cblas.h"

int main() {
    // Initialize with automatic core detection
    cblas_init(CBLAS_DEFAULT_THREADS);
    
    // Print configuration
    cblas_print_configuration();
    
    // Allocate large vectors
    size_t n = 1000000;
    float *x = malloc(n * sizeof(float));
    float *y = malloc(n * sizeof(float));
    
    // Initialize data...
    
    // This will use multi-threading (n > CBLAS_MT_DOT)
    float result = cblas_sdot(n, x, 1, y, 1);
    
    // For small operations, threading is automatically disabled
    n = 100;  // Below threshold
    result = cblas_sdot(n, x, 1, y, 1);  // Single-threaded
    
    // Adjust thread count dynamically
    cblas_set_num_threads(2);
    
    // Cleanup
    free(x);
    free(y);
    cblas_shutdown();
    
    return 0;
}
```

## References

- Source files: `server.c` (POSIX), `server_win32.c` (Windows)
- Work dispatch: `util.c` (functions: `cblas_level1_exec`, `cblas_level1_exec_result`)
- Configuration: `cblas.h` (thresholds: lines 68-72)
- Thread management API: `cblas.h` (lines 268-276)
