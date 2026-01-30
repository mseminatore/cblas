# Multi-Threading Debug Guide

This guide explains how to use the enhanced MT debug features in CBLAS to troubleshoot multi-threading issues and analyze performance.

## Overview

CBLAS includes comprehensive debugging support for multi-threaded operations, providing insights into:

- Thread activity and lifecycle
- Work queue depth monitoring
- Per-thread execution timing
- Load imbalance detection
- Operation-specific performance tracking

## Enabling MT Debug

### Basic Debug Mode

To enable basic debug output, define `MT_DEBUG` when building:

```bash
# Using Make
CFLAGS="-DMT_DEBUG" make

# Using CMake
cmake -DCMAKE_C_FLAGS="-DMT_DEBUG" ..
cmake --build .
```

This produces human-readable debug output to stderr.

### JSON Debug Mode

For machine-parseable output suitable for log analysis tools, enable both `MT_DEBUG` and `MT_DEBUG_JSON`:

```bash
# Using Make
CFLAGS="-DMT_DEBUG -DMT_DEBUG_JSON" make

# Using CMake
cmake -DCMAKE_C_FLAGS="-DMT_DEBUG -DMT_DEBUG_JSON" ..
cmake --build .
```

## Debug Output Format

### Human-Readable Mode (MT_DEBUG)

Example output:

```
set threads = 4
[Thread 0] created.
[Thread 0] waits.
adding 4 items to the queue.
[Queue] Depth: 4
waking worker threads.
[Thread 0] is awake.
[Thread 0] executing a task.
[Thread 0] SDOT took 3.45 us
[Thread 0] task completed.
[Load Balance] WARNING: 25.3% variance (min=3.20us, max=4.81us, avg=3.84us)
```

### JSON Mode (MT_DEBUG + MT_DEBUG_JSON)

Example output (one JSON object per line):

```json
{"type":"trace","message":"set threads = 4\n"}
{"type":"thread","tid":0,"message":"created.\n"}
{"type":"queue","depth":4}
{"type":"timing","tid":0,"operation":"SDOT","duration_us":3.45}
{"type":"load_balance","thread_count":4,"variance_pct":25.3,"min_us":3.20,"max_us":4.81,"avg_us":3.84,"warning":true}
```

## Debug Event Types

### Thread Events
Tracks thread lifecycle and state changes.

**Human-readable:**
```
[Thread 2] created.
[Thread 2] waits.
[Thread 2] is awake.
[Thread 2] executing a task.
[Thread 2] task completed.
[Thread 2] exiting.
```

**JSON:**
```json
{"type":"thread","tid":2,"message":"created.\n"}
{"type":"thread","tid":2,"message":"executing a task.\n"}
```

### Timing Events
Records per-thread execution time with operation names.

**Human-readable:**
```
[Thread 1] SDOT took 12.34 us
[Thread 2] SCOPY took 45.67 us
[Thread 0] GEMM took 123.45 us
```

**JSON:**
```json
{"type":"timing","tid":1,"operation":"SDOT","duration_us":12.34}
{"type":"timing","tid":2,"operation":"SCOPY","duration_us":45.67}
{"type":"timing","tid":0,"operation":"GEMM","duration_us":123.45}
```

### Queue Depth Events
Monitors work queue size to detect queuing issues.

**Human-readable:**
```
[Queue] Depth: 8
```

**JSON:**
```json
{"type":"queue","depth":8}
```

### Load Balance Events
Detects load imbalance across threads (triggers on >20% variance).

**Human-readable:**
```
[Load Balance] WARNING: 45.2% variance (min=10.20us, max=18.41us, avg=12.67us)
[Load Balance] OK: 12.5% variance (min=15.30us, max=17.62us, avg=16.18us)
```

**JSON:**
```json
{"type":"load_balance","thread_count":4,"variance_pct":45.2,"min_us":10.20,"max_us":18.41,"avg_us":12.67,"warning":true}
{"type":"load_balance","thread_count":4,"variance_pct":12.5,"min_us":15.30,"max_us":17.62,"avg_us":16.18,"warning":false}
```

## Supported Operations

The following BLAS operations include operation-name tracking in MT debug:

### Level 1
- `SDOT` - Single-precision dot product
- `SCOPY` - Single-precision vector copy
- `SSWAP` - Single-precision vector swap
- `DCOPY` - Double-precision vector copy
- `DSWAP` - Double-precision vector swap

Additional operations will show as "task" or "UNKNOWN" in timing output.

## Performance Considerations

**Important:** MT_DEBUG adds significant overhead and should **NEVER** be enabled in production builds:

1. **Queue depth monitoring** is O(n) and runs inside critical sections
2. **Timing measurements** require system calls on every task
3. **Output operations** can block on stderr writes
4. **Load balance analysis** scans all work items after each operation

For production performance monitoring, use `CBLAS_ENABLE_STATS` instead of `MT_DEBUG`.

## Analyzing Debug Output

### Detecting Load Imbalance

Look for "WARNING" in load balance output. High variance indicates:
- Uneven work distribution
- Cache effects
- Thread migration overhead
- OS scheduler interference

**Remediation:**
- Adjust partition sizes
- Increase problem size to amortize overhead
- Pin threads to cores (external to CBLAS)

### Monitoring Queue Depth

Normal queue depth should be close to the number of worker threads. Higher values indicate:
- Work items being added faster than consumed
- Thread starvation
- Insufficient worker threads

### Timing Analysis

Compare operation timing across threads:
- Similar times = good load balance
- Wide variation = investigate partitioning
- Very short times (< 1us) = overhead dominates useful work

## Example Usage

```c
#include "cblas.h"

int main(void)
{
    // Initialize with 4 threads
    cblas_init(4);
    
    const size_t n = 100000;  // Large enough to trigger MT
    float *x = malloc(n * sizeof(float));
    float *y = malloc(n * sizeof(float));
    
    // Initialize vectors...
    
    // Perform operation - debug output will show:
    // - Thread creation
    // - Work queue depth
    // - Per-thread timing
    // - Load balance analysis
    float result = cblas_sdot(n, x, 1, y, 1);
    
    free(x);
    free(y);
    cblas_shutdown();
    
    return 0;
}
```

Build and run with debug:

```bash
# Human-readable output
CFLAGS="-DMT_DEBUG" make my_program
./my_program 2>debug.log

# JSON output for analysis
CFLAGS="-DMT_DEBUG -DMT_DEBUG_JSON" make my_program
./my_program 2>debug.json

# Parse JSON with jq
./my_program 2>&1 | grep timing | jq -s 'group_by(.operation) | map({op: .[0].operation, count: length, avg: (map(.duration_us) | add / length)})'
```

## Troubleshooting

### No Debug Output

- Verify `MT_DEBUG` is defined at compile time
- Check that operations exceed MT thresholds (e.g., `CBLAS_MT_DOT = 10000`)
- Confirm debug output goes to stderr (use `2>&1` or `2>file`)

### Malformed JSON

- Ensure both `MT_DEBUG` and `MT_DEBUG_JSON` are defined
- Some trace messages may contain embedded newlines - use line-delimited JSON parsers

### Performance Degradation

- MT_DEBUG adds substantial overhead - this is expected
- Never enable in production
- For production monitoring, use `CBLAS_ENABLE_STATS`

## See Also

- [THREADING.md](THREADING.md) - General threading architecture
- [cblas.h](../cblas.h) - MT_DEBUG macro definitions (lines 78-125)
- [test_mt_debug.c](../test_mt_debug.c) - Example debug usage
