# MT Debug Output Improvements - Summary

## Issue #21 Implementation

This document summarizes the implementation of enhanced MT (Multi-Threading) debug output as specified in RECOMMENDED_ISSUES.md item #21.

## Changes Made

### 1. Enhanced Debug Macros (cblas.h)

Added four new structured debug macros:

```c
MT_TRACE_THREAD(tid, ...)       // Thread-specific logging with ID prefix
MT_TRACE_TIMING(tid, op, dur)   // Per-thread execution time tracking
MT_TRACE_QUEUE_DEPTH(depth)     // Work queue depth monitoring
MT_TRACE_LOAD_BALANCE(cnt, arr) // Automatic load imbalance detection
```

### 2. Cross-Platform Timing Support

Added `mt_get_time_us()` helper function for microsecond-precision timing:
- Windows: Uses `QueryPerformanceCounter()`
- POSIX: Uses `clock_gettime(CLOCK_MONOTONIC, ...)`

### 3. Timing Fields in work_queue_t

Extended the work_queue_t structure with debug timing fields:
```c
#ifdef MT_DEBUG
    double start_time_us;  // Task start timestamp
    double end_time_us;    // Task completion timestamp
#endif
```

### 4. Instrumented Thread Servers

Both `server.c` (POSIX) and `server_win32.c` (Windows) now:
- Track execution time for each work item
- Log thread-specific events with thread IDs
- Monitor work queue depth on task submission
- Detect and report load imbalance after task completion

### 5. Load Imbalance Detection

The `MT_TRACE_LOAD_BALANCE` macro automatically:
- Calculates min/max/average execution times across threads
- Computes variance percentage
- Warns if variance exceeds 20% (indicates poor load distribution)
- Safely handles edge cases (zero times, empty arrays)

## Example Output

When `MT_DEBUG` is enabled, the following output is generated:

```
set threads = 4
[Thread 0] created.
[Thread 0] waits.
Testing DOT product with 500000 elements (MT threshold: 10000)...
adding 1 items to the queue.
[Queue] Depth: 1
waking worker threads.
[Thread 0] is awake.
[Thread 0] executing a task.
[Thread 0] task took 145.93 us
[Thread 0] task completed.
queued tasks finished.
```

With multiple threads and load imbalance:
```
[Load Balance] WARNING: 45.2% variance (min=100.50us, max=300.25us, avg=206.84us)
[Load Balance] OK: 8.3% variance (min=195.12us, max=212.45us, avg=203.91us)
```

## Performance Impact

- **MT_DEBUG disabled**: Zero overhead - all macros compile to empty statements
- **MT_DEBUG enabled**: Debug overhead includes:
  - Queue depth counting (O(n) in critical section)
  - Microsecond timing calls per task
  - Load balance calculation (O(n) where n = thread count)
  
This overhead is acceptable for debugging but should not be used in production.

## Safety Improvements

Based on code review feedback, the implementation includes:

1. **Division-by-zero protection** - Checks avg_time > 0 before calculating variance
2. **Array bounds checking** - Verifies thread_count > 0 before accessing arrays
3. **Variable shadowing prevention** - Uses `_mt_idx` instead of generic `i` in macros
4. **Lifetime safety comments** - Documents that queue items are stack-allocated and accessed before scope exit
5. **Performance warnings** - Documents O(n) queue traversal in critical section

## Testing

Created `test_mt_debug.c` to demonstrate the enhanced debug output:
- Initializes with 4 threads
- Performs large vector operations (500K elements)
- Triggers multi-threading (exceeds MT_DOT and MT_COPY thresholds)
- Shows thread creation, work distribution, timing, and shutdown

## Documentation

Updated `docs/THREADING.md` with:
- Description of new debug macros
- Example output showing all features
- Load balance detection explanation
- Performance impact notes
- Additional troubleshooting tip (#5) for high load imbalance

## Files Modified

1. `cblas.h` - Added debug macros, timing helper, work_queue_t fields
2. `server.c` - Instrumented with timing and monitoring
3. `server_win32.c` - Instrumented with timing and monitoring
4. `docs/THREADING.md` - Updated debugging section
5. `test_mt_debug.c` - New demo/test program
6. `.gitignore` - Added test_mt_debug binary

## Benefits

1. **Better observability** - Clear view into thread behavior and performance
2. **Load imbalance detection** - Automatic warnings for poor work distribution
3. **Performance troubleshooting** - Per-thread timing helps identify bottlenecks
4. **Zero production cost** - No overhead when MT_DEBUG is disabled
5. **Cross-platform** - Works on both Windows and POSIX systems

## Future Enhancements (Not Implemented)

The RECOMMENDED_ISSUES.md suggested but not implemented in this PR:
- JSON output mode for machine parsing
- Per-operation statistics tracking
- More granular timing (e.g., queue wait time vs execution time)

These can be added in future work if needed.
