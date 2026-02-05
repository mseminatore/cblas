//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cblas.h"

//------------------------------------------------------
// state variables
//------------------------------------------------------
volatile int cblas_max_threads  = MAX_THREADS;  // max system supported threads
static int cblas_server_alive   = CBLAS_FALSE;  // has thread server been initialized
kernels_t blas_kernels;

//------------------------------------------------------
// Runtime MT threshold variables (initialized to defaults)
//------------------------------------------------------
CBLAS_INDEX cblas_mt_dot_threshold = CBLAS_MT_DOT_DEFAULT;
CBLAS_INDEX cblas_mt_axpy_threshold = CBLAS_MT_AXPY_DEFAULT;
CBLAS_INDEX cblas_mt_copy_threshold = CBLAS_MT_COPY_DEFAULT;
CBLAS_INDEX cblas_mt_ger_threshold = CBLAS_MT_GER_DEFAULT;
CBLAS_INDEX cblas_mt_gemm_threshold = CBLAS_MT_GEMM_DEFAULT;
CBLAS_INDEX cblas_mt_gemv_threshold = CBLAS_MT_GEMV_DEFAULT;

//------------------------------------------------------
// GEMM cache-aware block sizes (initialized at runtime)
// Default values match 32KB L1d configuration (Intel/AMD)
//------------------------------------------------------
CBLAS_INDEX cblas_gemm_mc = 128;   // Rows of A to pack
CBLAS_INDEX cblas_gemm_kc = 256;   // Inner dimension
CBLAS_INDEX cblas_gemm_nb = 256;   // Columns of B to pack

//------------------------------------------------------
// Performance counters - track stats per operation
//------------------------------------------------------
#define STATS_TABLE_SIZE 32

typedef struct {
    char name[32];
    cblas_stats_t stats;
} stats_entry_t;

static stats_entry_t stats_table[STATS_TABLE_SIZE];
static int stats_initialized = 0;

#ifdef MT_ENABLED
#ifdef _WIN32
#include <windows.h>
static CRITICAL_SECTION stats_mutex;
static int stats_mutex_initialized = 0;
#define STATS_LOCK() EnterCriticalSection(&stats_mutex)
#define STATS_UNLOCK() LeaveCriticalSection(&stats_mutex)
#else
#include <pthread.h>
static pthread_mutex_t stats_mutex = PTHREAD_MUTEX_INITIALIZER;
#define STATS_LOCK() pthread_mutex_lock(&stats_mutex)
#define STATS_UNLOCK() pthread_mutex_unlock(&stats_mutex)
#endif
#else
#define STATS_LOCK()
#define STATS_UNLOCK()
#endif

//------------------------------------------------------
// Initialize stats table
//------------------------------------------------------
static void init_stats_table(void)
{
    if (stats_initialized)
        return;
    
#if defined(MT_ENABLED) && defined(_WIN32)
    if (!stats_mutex_initialized) {
        InitializeCriticalSection(&stats_mutex);
        stats_mutex_initialized = 1;
    }
#endif
    
    STATS_LOCK();
    if (!stats_initialized) {
        memset(stats_table, 0, sizeof(stats_table));
        stats_initialized = 1;
    }
    STATS_UNLOCK();
}

//------------------------------------------------------
// Find or create stats entry for operation
//------------------------------------------------------
static cblas_stats_t* get_stats_entry(const char* operation)
{
    if (!stats_initialized)
        init_stats_table();
    
    STATS_LOCK();
    
    // Look for existing entry
    for (int i = 0; i < STATS_TABLE_SIZE; i++) {
        if (stats_table[i].name[0] == 0) {
            // Empty slot - create new entry
#ifdef _MSC_VER
            strncpy_s(stats_table[i].name, sizeof(stats_table[i].name), operation, _TRUNCATE);
#else
            strncpy(stats_table[i].name, operation, sizeof(stats_table[i].name) - 1);
            stats_table[i].name[sizeof(stats_table[i].name) - 1] = 0;
#endif
            STATS_UNLOCK();
            return &stats_table[i].stats;
        }
        if (strcmp(stats_table[i].name, operation) == 0) {
            // Found existing entry
            STATS_UNLOCK();
            return &stats_table[i].stats;
        }
    }
    
    STATS_UNLOCK();
    return NULL; // Table full
}

//------------------------------------------------------
// Record a BLAS operation
//------------------------------------------------------
void cblas_record_operation(const char* operation, uint64_t elements, int mt_used, double time_sec)
{
    cblas_stats_t* stats = get_stats_entry(operation);
    if (!stats)
        return;
    
    STATS_LOCK();
    stats->total_calls++;
    stats->total_elements += elements;
    if (mt_used)
        stats->mt_activations++;
    stats->total_time_sec += time_sec;
    STATS_UNLOCK();
}

//------------------------------------------------------
// Get stats for a specific operation
//------------------------------------------------------
const cblas_stats_t* cblas_get_stats(const char* operation)
{
    if (!stats_initialized)
        init_stats_table();
    
    STATS_LOCK();
    for (int i = 0; i < STATS_TABLE_SIZE; i++) {
        if (strcmp(stats_table[i].name, operation) == 0) {
            STATS_UNLOCK();
            return &stats_table[i].stats;
        }
    }
    STATS_UNLOCK();
    
    return NULL;
}

//------------------------------------------------------
// Reset all performance counters
//------------------------------------------------------
void cblas_reset_stats(void)
{
    if (!stats_initialized)
        init_stats_table();
    
    STATS_LOCK();
    for (int i = 0; i < STATS_TABLE_SIZE; i++) {
        if (stats_table[i].name[0] != 0) {
            memset(&stats_table[i].stats, 0, sizeof(cblas_stats_t));
        }
    }
    STATS_UNLOCK();
}

//------------------------------------------------------
// Print performance statistics
//------------------------------------------------------
void cblas_print_stats(void)
{
    if (!stats_initialized)
        init_stats_table();
    
    printf("\n=== CBLAS Performance Statistics ===\n");
    printf("%-12s %10s %15s %10s %12s %12s\n", 
           "Operation", "Calls", "Elements", "MT Uses", "Time (s)", "Avg (us)");
    printf("------------------------------------------------------------------------\n");
    
    STATS_LOCK();
    int found_any = 0;
    for (int i = 0; i < STATS_TABLE_SIZE; i++) {
        if (stats_table[i].name[0] != 0 && stats_table[i].stats.total_calls > 0) {
            found_any = 1;
            cblas_stats_t* s = &stats_table[i].stats;
            double avg_us = (s->total_calls > 0) ? (s->total_time_sec * 1e6 / s->total_calls) : 0.0;
            printf("%-12s %10llu %15llu %10llu %12.6f %12.3f\n",
                   stats_table[i].name,
                   (unsigned long long)s->total_calls,
                   (unsigned long long)s->total_elements,
                   (unsigned long long)s->mt_activations,
                   s->total_time_sec,
                   avg_us);
        }
    }
    STATS_UNLOCK();
    
    if (!found_any) {
        printf("(no operations recorded)\n");
    }
    printf("\n");
}

//------------------------------------------------------
// Cleanup stats resources
//------------------------------------------------------
void cblas_cleanup_stats(void)
{
#if defined(MT_ENABLED) && defined(_WIN32)
    if (stats_mutex_initialized) {
        DeleteCriticalSection(&stats_mutex);
        stats_mutex_initialized = 0;
    }
#endif
    stats_initialized = 0;
}

//------------------------------------------------------
// return the current server status
//------------------------------------------------------
int cblas_is_server_alive(void)
{
    return cblas_server_alive;
}

//------------------------------------------------------
// mark the server as dead/alive
//------------------------------------------------------
void cblas_set_server_alive(int yesno)
{
    cblas_server_alive = yesno;
}

//------------------------------------------------------
// use secure run-time calls if available
//------------------------------------------------------
#ifdef _WIN32
#   define strcat(s1, s2) strcat_s((s1), sizeof(s1), (s2))
#endif

//------------------------------------------------------
// standard BLAS error handler
//
// srcname - name of the function that called xerbla
// info - position of the invalid parameter in the parameter list
// len - length of the name in bytes
//------------------------------------------------------
void xerbla(const char *srcname, int info, size_t len)
{
    (void)len;
    printf("\nCBLAS error: parameter %d was invalid in call to %s()\n",info, srcname);
}

//------------------------------------------------------
// return the current config
//------------------------------------------------------
const char *cblas_get_config(void)
{
    static char buf[CBLAS_SMALL_BUF];

#ifdef _WIN32
    sprintf_s(buf, sizeof(buf), "\nCBLAS %s %s MAX_THREADS=%d", CBLAS_VERSION, cpu_get_core_name(), MAX_THREADS);
#else
    sprintf(buf, "\nCBLAS %s %s MAX_THREADS=%d", CBLAS_VERSION, cpu_get_core_name(), MAX_THREADS);
#endif
    
    return buf;
}

//------------------------------------------------------
// return bit flags defining available ISA extensions 
//------------------------------------------------------
const char *cblas_get_isa_features(void)
{
    static char buf[CBLAS_SMALL_BUF];

    unsigned int cpu = cpu_get_features();

    char *pos = buf;
    int remaining = CBLAS_SMALL_BUF;
    int written = 0;
    int first = 1;

    buf[0] = 0;

#if defined(__APPLE__) || defined(__aarch64__)
    if (cpu & CPU_NEON) {
        written = snprintf(pos, remaining, "NEON");
        if (written > 0 && written < remaining) {
            pos += written;
            remaining -= written;
            first = 0;
        }
    }

    if (cpu & CPU_NEON_FMA) {
        written = snprintf(pos, remaining, "%sFMA", first ? "" : ", ");
        if (written > 0 && written < remaining) {
            pos += written;
            remaining -= written;
            first = 0;
        }
    }
#endif

#if defined(__x86_64__) || defined(_M_X64)
    if (cpu & CPU_SSE) {
        written = snprintf(pos, remaining, "SSE");
        if (written > 0 && written < remaining) {
            pos += written;
            remaining -= written;
            first = 0;
        }
    }
    if (cpu & CPU_AVX) {
        written = snprintf(pos, remaining, "%sAVX", first ? "" : ", ");
        if (written > 0 && written < remaining) {
            pos += written;
            remaining -= written;
            first = 0;
        }
    }
    if (cpu & CPU_AVX2) {
        written = snprintf(pos, remaining, "%sAVX2", first ? "" : ", ");
        if (written > 0 && written < remaining) {
            pos += written;
            remaining -= written;
            first = 0;
        }
    }
    if (cpu & CPU_x64_FMA3) {
        written = snprintf(pos, remaining, "%sFMA", first ? "" : ", ");
        if (written > 0 && written < remaining) {
            pos += written;
            remaining -= written;
            first = 0;
        }
    }
#endif

    return buf;
}

//------------------------------------------------------
// return the name of the host CPU
//------------------------------------------------------
const char *cblas_get_corename(void)
{
    return cpu_get_core_name();
}

//------------------------------------------------------
// return number of CPUs on the host system
//------------------------------------------------------
int cblas_get_num_procs(void)
{
    return cpu_get_core_count();
}

//------------------------------------------------------
// standard configuration print banner
//------------------------------------------------------
void cblas_print_configuration(void)
{
    printf("%s\n", cblas_get_config());
    printf("     CPU uArch: %s\n", cblas_get_corename());
	printf("ISA Extensions: %s\n", cblas_get_isa_features());
    
    // Show hybrid architecture info if available
    if (cpu_is_hybrid())
    {
        printf(" Cores/Threads: %d/%d (Hybrid: %dP + %dE)\n", 
               cblas_get_num_procs(), cblas_get_num_threads(),
               cpu_get_p_core_count(), cpu_get_e_core_count());
    }
    else
    {
        printf(" Cores/Threads: %d/%d\n", cblas_get_num_procs(), cblas_get_num_threads());
    }
    
    printf(" L1$ line size: %d bytes\n", cpu_get_cacheline_size());
    printf(" L1$ data size: %d Kbytes\n", cpu_get_l1_data_cache_size());
    printf("      L2$ size: %d Kbytes\n", cpu_get_l2_cache_size());
    printf(" GEMM block sz: mc=%zu, kc=%zu, nb=%zu\n\n", cblas_gemm_mc, cblas_gemm_kc, cblas_gemm_nb);
}

//------------------------------------------------------
// Print active kernel configuration
//------------------------------------------------------
void cblas_print_kernels(void)
{
    unsigned int cpu = cpu_get_features();
    
    printf("=== Active CBLAS Kernels ===\n");
    
    // Determine kernel variant based on CPU features and build config
    #if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)
        if (cpu & CPU_AVX2) {
            printf("Architecture: x86_64 with AVX2\n");
            printf("SIMD Width:   256-bit (8 floats, 4 doubles)\n");
            if (cpu & CPU_x64_FMA3) {
                printf("FMA Support:  Yes (FMA3)\n");
            } else {
                printf("FMA Support:  No\n");
            }
        } else if (cpu & CPU_AVX) {
            printf("Architecture: x86_64 with AVX\n");
            printf("SIMD Width:   256-bit (8 floats, 4 doubles)\n");
            printf("FMA Support:  No\n");
        } else if (cpu & CPU_SSE) {
            printf("Architecture: x86_64 with SSE\n");
            printf("SIMD Width:   128-bit (4 floats, 2 doubles)\n");
            printf("FMA Support:  No\n");
        } else {
            printf("Architecture: x86_64 (scalar fallback)\n");
            printf("SIMD Width:   None\n");
        }
        // Report correct kernel variant based on FMA support
        if (cpu & CPU_x64_FMA3) {
            printf("Kernel Variant: SSE/AVX optimized with FMA (_k_noinc_fma)\n");
        } else {
            printf("Kernel Variant: SSE/AVX optimized (_k_noinc_sse)\n");
        }
    #elif defined(__aarch64__) && defined(__ARM_NEON)
        printf("Architecture: ARM64 with NEON\n");
        printf("SIMD Width:   128-bit (4 floats, 2 doubles)\n");
        if (cpu & CPU_NEON_FMA) {
            printf("FMA Support:  Yes (NEON FMA)\n");
        } else {
            printf("FMA Support:  No\n");
        }
        printf("Kernel Variant: NEON optimized (_k_noinc_neon)\n");
    #else
        printf("Architecture: Generic (SIMD disabled)\n");
        printf("Kernel Variant: Scalar fallback\n");
    #endif

    printf("\nOptimizations:\n");
    printf("  - 4-way accumulator unrolling: ");
    printf("Yes (32 floats/16 doubles per iteration)\n");
    
    printf("  - FMA instructions (runtime):  ");
#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)
    if (cpu & CPU_x64_FMA3) {
        printf("Yes (Intel FMA3 detected, using _fma kernels)\n");
    } else {
        printf("No (CPU lacks FMA3, using mul+add)\n");
    }
#elif defined(__aarch64__) && defined(__ARM_NEON)
    if (cpu & CPU_NEON_FMA) {
        printf("Yes (ARM NEON FMA)\n");
    } else {
        printf("No (separate mul+add)\n");
    }
#else
    printf("No (separate mul+add)\n");
#endif
    
    printf("  - Software prefetching:       ");
#if defined(__GNUC__) || defined(__clang__) || defined(_MSC_VER)
    printf("Yes (%d elements ahead)\n", CBLAS_PREFETCH_DISTANCE);
#else
    printf("No\n");
#endif

    printf("  - Multi-threading:            ");
#ifdef MT_ENABLED
    printf("Yes (%d threads)\n", cblas_get_num_threads());
#else
    printf("No\n");
#endif

    printf("\nMT Thresholds:\n");
    printf("  DOT:  %lu   AXPY: %lu   COPY: %lu\n", 
           (unsigned long)cblas_mt_dot_threshold, 
           (unsigned long)cblas_mt_axpy_threshold, 
           (unsigned long)cblas_mt_copy_threshold);
    printf("  GER:  %lu   GEMM: %lu   GEMV: %lu\n\n", 
           (unsigned long)cblas_mt_ger_threshold, 
           (unsigned long)cblas_mt_gemm_threshold, 
           (unsigned long)cblas_mt_gemv_threshold);
}

//------------------------------------------------------
// level 1 dispatch
//------------------------------------------------------
void cblas_level1_exec(CBLAS_INDEX byte_stride, kernel_function kernel, CBLAS_INDEX n, void *x, CBLAS_INDEX incx, void *y, CBLAS_INDEX incy, void *alpha, void *beta, const char* op_name)
{
    work_queue_t queue[MAX_THREADS];
    cblas_args_t args[MAX_THREADS];

    CBLAS_INDEX thread_count = CLAMP(cblas_get_num_threads(), 1, MAX_THREADS);

    for (CBLAS_INDEX i = 0; i < thread_count; i++)
    {
        args[i].incx = incx;
        args[i].incy = incy;
        args[i].alpha = alpha;
        args[i].beta = beta;

        // compute partition starts based on remaining task size and remaining threads
        CBLAS_INDEX partition_size = (n + thread_count - i - 1) / (thread_count - i);

        args[i].n = partition_size;
        args[i].x = x;
        args[i].y = y;

        n -= partition_size;

        // Advance pointers by partition size accounting for stride
        x = (char*)x + partition_size * incx * byte_stride;
        if (y) y = (char*)y + partition_size * incy * byte_stride;

        queue[i].finished   = 0;
        queue[i].args       = &args[i];
        queue[i].kernel     = kernel;
        queue[i].next       = &queue[i + 1];
#ifdef MT_DEBUG
        queue[i].operation  = op_name ? op_name : "UNKNOWN";
#else
        (void)op_name;  // Suppress unused parameter warning
#endif
    }

    // mark end of task queue
//    if (thread_count)
        queue[thread_count - 1].next = NULL;

    // synchronously execute task queue
    cblas_execute(thread_count, queue);
}

//------------------------------------------------------
// level 1 dispatch
//------------------------------------------------------
void cblas_level1_exec_result(CBLAS_INDEX byte_stride, kernel_function kernel, CBLAS_INDEX n, void* x, CBLAS_INDEX incx, void* y, CBLAS_INDEX incy, void *c, const char* op_name)
{
    work_queue_t queue[MAX_THREADS];
    cblas_args_t args[MAX_THREADS];

    CBLAS_INDEX thread_count = CLAMP(cblas_get_num_threads(), 1, MAX_THREADS);

    for (CBLAS_INDEX i = 0; i < thread_count; i++)
    {
        args[i].incx = incx;
        args[i].incy = incy;

        // compute partition starts based on remaining task size and remaining threads
        CBLAS_INDEX partition_size = (n + thread_count - i - 1) / (thread_count - i);

        args[i].n = partition_size;
        args[i].x = x;
        args[i].y = y;
        args[i].c = (char*)c + i * byte_stride;

        n -= partition_size;

        // Advance pointers by partition size accounting for stride
        x = (char*)x + partition_size * incx * byte_stride;
        y = (char*)y + partition_size * incy * byte_stride;

        queue[i].finished = 0;
        queue[i].args = &args[i];
        queue[i].kernel = kernel;
        queue[i].next = &queue[i + 1];
#ifdef MT_DEBUG
        queue[i].operation = op_name ? op_name : "UNKNOWN";
#else
        (void)op_name;  // Suppress unused parameter warning
#endif
    }

    // mark end of task queue
//    if (thread_count)
    queue[thread_count - 1].next = NULL;

    // synchronously execute task queue
    cblas_execute(thread_count, queue);
}

//------------------------------------------------------
// level 2 dispatch - partitions rows across threads
//------------------------------------------------------
void cblas_level2_exec(CBLAS_INDEX element_size, kernel_function kernel, cblas_level2_partition_t part_mode,
                       CBLAS_INDEX m, CBLAS_INDEX n, void* a, CBLAS_INDEX lda,
                       void* x, CBLAS_INDEX incx, void* y, CBLAS_INDEX incy,
                       void* alpha, void* beta, const char* op_name)
{
    work_queue_t queue[MAX_THREADS];
    cblas_args_t args[MAX_THREADS];

    CBLAS_INDEX thread_count = CLAMP(cblas_get_num_threads(), 1, MAX_THREADS);
    CBLAS_INDEX rows_remaining = m;
    CBLAS_INDEX row_offset = 0;

    for (CBLAS_INDEX i = 0; i < thread_count; i++)
    {
        // Compute partition size: distribute rows evenly
        CBLAS_INDEX rows_per_thread = (rows_remaining + thread_count - i - 1) / (thread_count - i);

        args[i].m = rows_per_thread;
        args[i].n = n;
        args[i].lda = lda;
        args[i].incx = incx;
        args[i].incy = incy;
        args[i].alpha = alpha;
        args[i].beta = beta;

        // Matrix A is always partitioned by rows
        args[i].a = (char*)a + row_offset * lda * element_size;

        // Partition either X or Y based on mode
        if (part_mode == CBLAS_PART_X)
        {
            // GER mode: X is partitioned with rows, Y is shared
            args[i].x = (char*)x + row_offset * incx * element_size;
            args[i].y = y;
        }
        else
        {
            // GEMV mode: Y is partitioned with rows, X is shared
            args[i].x = x;
            args[i].y = (char*)y + row_offset * incy * element_size;
        }

        queue[i].finished = 0;
        queue[i].args = &args[i];
        queue[i].kernel = kernel;
        queue[i].next = &queue[i + 1];
#ifdef MT_DEBUG
        queue[i].operation = op_name ? op_name : "UNKNOWN";
#else
        (void)op_name;  // Suppress unused parameter warning
#endif

        row_offset += rows_per_thread;
        rows_remaining -= rows_per_thread;
    }

    // mark end of task queue
    queue[thread_count - 1].next = NULL;

    // synchronously execute task queue
    cblas_execute(thread_count, queue);
}

//------------------------------------------------------
// Calculate optimal GEMM block sizes based on cache
//------------------------------------------------------
static void cblas_compute_gemm_block_sizes(void)
{
    int l1_cache_kb = cpu_get_l1_data_cache_size();
    int l2_cache_kb = cpu_get_l2_cache_size();
    
    // Default values if cache detection fails
    if (l1_cache_kb == 0) l1_cache_kb = 32;
    if (l2_cache_kb == 0) l2_cache_kb = 256;
    
    // Target: fit packed A and B in L2 cache
    // Packed A: mc × kc × 4 bytes
    // Packed B: kc × nb × 4 bytes
    // Total: (mc × kc + kc × nb) × 4 bytes should fit in L2
    
    // Adjust block sizes based on L1 cache size
    if (l1_cache_kb >= 128) {
        // Apple M-series (128KB L1d): larger blocks
        cblas_gemm_mc = 512;
        cblas_gemm_kc = 256;
        cblas_gemm_nb = 512;
    } else if (l1_cache_kb >= 64) {
        // ARM Cortex (64KB L1d): medium blocks
        cblas_gemm_mc = 256;
        cblas_gemm_kc = 256;
        cblas_gemm_nb = 512;
    } else if (l2_cache_kb >= 512) {
        // Intel/AMD with 512KB+ L2: optimize for L2 cache
        // mc=192, kc=256, nb=384: Total = (192*256 + 256*384)*4 = 577 KB
        cblas_gemm_mc = 192;
        cblas_gemm_kc = 256;
        cblas_gemm_nb = 384;
    } else if (l2_cache_kb >= 256) {
        // Smaller L2 cache (256KB)
        cblas_gemm_mc = 128;
        cblas_gemm_kc = 192;
        cblas_gemm_nb = 256;
    } else {
        // Very small L2 cache: conservative defaults
        cblas_gemm_mc = 128;
        cblas_gemm_kc = 256;
        cblas_gemm_nb = 256;
    }
    
    // Verify total size fits in L2 cache (use 85% of L2 for better utilization)
    size_t target_kb = (l2_cache_kb * 85) / 100;
    size_t total_kb = ((cblas_gemm_mc * cblas_gemm_kc +
                     cblas_gemm_kc * cblas_gemm_nb) * 4) / 1024;
    
    // If too large, scale down proportionally while maintaining ratios
    if (total_kb > target_kb) {
        // Scale down by reducing kc first (inner dimension)
        // This maintains mc and nb which affect outer loop efficiency
        while (total_kb > target_kb && cblas_gemm_kc > 64) {
            cblas_gemm_kc -= 16;  // Reduce by cache line increments
            total_kb = ((cblas_gemm_mc * cblas_gemm_kc +
                        cblas_gemm_kc * cblas_gemm_nb) * 4) / 1024;
        }
        
        // If still too large, reduce mc and nb proportionally
        if (total_kb > target_kb) {
            float scale = (float)target_kb / (float)total_kb;
            cblas_gemm_mc = (CBLAS_INDEX)(cblas_gemm_mc * scale);
            cblas_gemm_nb = (CBLAS_INDEX)(cblas_gemm_nb * scale);
            
            // Round down to multiples of 16 for cache alignment
            cblas_gemm_mc = (cblas_gemm_mc / 16) * 16;
            cblas_gemm_nb = (cblas_gemm_nb / 16) * 16;
        }
        
        // Ensure minimum sizes (must be at least 64)
        cblas_gemm_mc = MAX(cblas_gemm_mc, 64);
        cblas_gemm_kc = MAX(cblas_gemm_kc, 64);
        cblas_gemm_nb = MAX(cblas_gemm_nb, 64);
    }
    
#ifdef USE_STATIC_BUFFERS
    // Clamp to maximum static buffer sizes
    cblas_gemm_mc = MIN(cblas_gemm_mc, 512);  // MAX_MC
    cblas_gemm_kc = MIN(cblas_gemm_kc, 256);  // MAX_KC
    cblas_gemm_nb = MIN(cblas_gemm_nb, 1024); // MAX_NB
#endif
}

//------------------------------------------------------
// initialize the CBLAS library
//------------------------------------------------------
void cblas_init(int threads)
{
    if (CBLAS_DEFAULT_THREADS == threads)
        threads = cpu_get_core_count();

    char *s_env_threads = getenv("CBLAS_THREADS");
    int env_threads = threads;

    if (s_env_threads)
        env_threads = atoi(s_env_threads);

    // make sure env_threads is valid;
    env_threads = MAX(env_threads, 1);

    threads = MIN(threads, env_threads);

#ifndef MT_ENABLED
    threads = 1;
#endif

    cblas_set_num_threads(threads);

    // Compute optimal GEMM block sizes based on cache
    cblas_compute_gemm_block_sizes();
    
    // Detect CPU features for kernel dispatch
    cpu_get_features();

    // start thread server
#ifdef MT_ENABLED
    if (!cblas_server_alive && cblas_init_server())
        cblas_server_alive = 1;
#endif

    // Auto-tune thresholds if requested via environment variable
    char *s_autotune = getenv("CBLAS_AUTO_TUNE");
    if (s_autotune && atoi(s_autotune) != 0) {
        cblas_autotune_thresholds();
    } else {
        // Use default thresholds
        cblas_reset_thresholds();
    }
}

//------------------------------------------------------
// return the active number of cblas threads
//------------------------------------------------------
int cblas_get_num_threads(void)
{
    return cblas_max_threads;
}

//------------------------------------------------------
//
//------------------------------------------------------
void cbu_timer_get_time(struct cblas_timer* t)
{
#ifdef _WIN32
    QueryPerformanceCounter(&t->t);
#else
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t->t);
#endif
}

//------------------------------------------------------
//
//------------------------------------------------------
float cbu_timer_get_delta(struct cblas_timer* t1, struct cblas_timer* t2)
{
    float dt;

#ifdef _WIN32
    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);

    dt = (t2->t.QuadPart - t1->t.QuadPart) / (float)freq.QuadPart;

#else
    int seconds = (int)(t2->t.tv_sec - t1->t.tv_sec);
    long long ns = t2->t.tv_nsec - t1->t.tv_nsec;
    dt = (float)seconds + (float)ns / (1000000000);
#endif

    return dt;
}

//------------------------------------------------------
// set given matrix to identity
//------------------------------------------------------
void cbu_sge_set_identity(float *mtx, CBLAS_INDEX cols, CBLAS_INDEX rows)
{
    for (CBLAS_INDEX row = 0; row < rows; row++)
        for (CBLAS_INDEX col = 0; col < cols; col++)
            mtx[row * cols + col] = (row == col) ? 1.0f : 0.0f;
}

void cbu_dge_set_identity(double *mtx, CBLAS_INDEX cols, CBLAS_INDEX rows)
{
    for (CBLAS_INDEX row = 0; row < rows; row++)
        for (CBLAS_INDEX col = 0; col < cols; col++)
            mtx[row * cols + col] = (row == col) ? 1.0 : 0.0;
}

//------------------------------------------------------
// test given matrix for identity
//------------------------------------------------------
int cbu_sge_is_identity(float* mtx, CBLAS_INDEX cols, CBLAS_INDEX rows)
{
    float val;

    for (CBLAS_INDEX row = 0; row < rows; row++)
        for (CBLAS_INDEX col = 0; col < cols; col++)
        {
            val = mtx[row * cols + col];
            
            if (row == col)
            {
                if (val != 1.0f)
                    return CBLAS_FALSE;
            }
            else
            {
                if (val != 0.0f)
                    return CBLAS_FALSE;
            }
        }

    return CBLAS_TRUE;
}

int cbu_dge_is_identity(double* mtx, CBLAS_INDEX cols, CBLAS_INDEX rows)
{
    double val;

    for (CBLAS_INDEX row = 0; row < rows; row++)
        for (CBLAS_INDEX col = 0; col < cols; col++)
        {
            val = mtx[row * cols + col];

            if (row == col)
            {
                if (val != 1.0)
                    return CBLAS_FALSE;
            }
            else
            {
                if (val != 0.0)
                    return CBLAS_FALSE;
            }
        }

    return CBLAS_TRUE;
}

//------------------------------------------------------
// allocate and fill an identity matrix
//------------------------------------------------------
float *cbu_sge_make_identity(int cols, int rows)
{
	float *mtx = malloc(cols * rows * sizeof(float));

	if (!mtx)
		return mtx;

	for (int row = 0; row < rows; row++)
		for (int col = 0; col < cols; cols++)
			mtx[row * cols + col] = (row == col) ? 1.0f : 0.0f;

	return mtx;
}

double* cbu_dge_make_identity(int cols, int rows)
{
    double* mtx = malloc(cols * rows * sizeof(double));

    if (!mtx)
        return mtx;

    for (int row = 0; row < rows; row++)
        for (int col = 0; col < cols; cols++)
            mtx[row * cols + col] = (row == col) ? 1.0 : 0.0;

    return mtx;
}

//------------------------------------------------------
// Auto-tuning infrastructure
//------------------------------------------------------

// Calibration constants
#define AUTOTUNE_MIN_SIZE 1000
#define AUTOTUNE_MAX_SIZE 1000000
#define AUTOTUNE_NUM_WARMUP 2
#define AUTOTUNE_NUM_ITERATIONS 5
#define AUTOTUNE_SPEEDUP_THRESHOLD 1.10f  // 10% speedup to enable MT

/**
 * @brief Benchmark an operation at a given size
 * @param operation_func Function pointer to the operation to benchmark
 * @param n Problem size
 * @param use_mt Whether to enable multi-threading for this benchmark
 * @return Execution time in seconds
 */
static double benchmark_operation(void (*operation_func)(CBLAS_INDEX, void*, void*), 
                                  CBLAS_INDEX n, int use_mt)
{
    struct cblas_timer t1, t2;
    double min_time = 1e9;
    
    // Allocate test vectors
    float *x = malloc(n * sizeof(float));
    float *y = NULL;  // Initialize to NULL for safe error handling
    
    // Only allocate y if x succeeded
    if (x) {
        y = malloc(n * sizeof(float));
        if (!y) {
            free(x);
            return -1.0;
        }
    } else {
        return -1.0;
    }
    
    // Initialize vectors
    for (CBLAS_INDEX i = 0; i < n; i++) {
        x[i] = (float)(i % 100) / 100.0f;
        y[i] = (float)((i + 1) % 100) / 100.0f;
    }
    
    // Save original thread count
    int orig_threads = cblas_get_num_threads();
    
    // Set threads for this benchmark
    if (use_mt) {
        cblas_set_num_threads(orig_threads);
    } else {
        cblas_set_num_threads(1);
    }
    
    // Warmup
    for (int i = 0; i < AUTOTUNE_NUM_WARMUP; i++) {
        operation_func(n, x, y);
    }
    
    // Measure
    for (int i = 0; i < AUTOTUNE_NUM_ITERATIONS; i++) {
        cbu_timer_get_time(&t1);
        operation_func(n, x, y);
        cbu_timer_get_time(&t2);
        
        double dt = cbu_timer_get_delta(&t1, &t2);
        if (dt < min_time) {
            min_time = dt;
        }
    }
    
    // Restore thread count
    cblas_set_num_threads(orig_threads);
    
    free(x);
    free(y);
    
    return min_time;
}

// Wrapper functions for benchmarking different operations
static void bench_sdot(CBLAS_INDEX n, void *x, void *y)
{
    cblas_sdot(n, (float*)x, 1, (float*)y, 1);
}

static void bench_scopy(CBLAS_INDEX n, void *x, void *y)
{
    cblas_scopy(n, (float*)x, 1, (float*)y, 1);
}

static void bench_saxpy(CBLAS_INDEX n, void *x, void *y)
{
    cblas_saxpy(n, 2.0f, (float*)x, 1, (float*)y, 1);
}

/**
 * @brief Find optimal MT threshold for a single operation
 * @param operation_func Function pointer to the operation
 * @param min_size Minimum problem size to test
 * @param max_size Maximum problem size to test
 * @return Optimal threshold (or max_size if MT never provides speedup)
 */
static CBLAS_INDEX find_mt_threshold(void (*operation_func)(CBLAS_INDEX, void*, void*),
                                     CBLAS_INDEX min_size,
                                     CBLAS_INDEX max_size)
{
    CBLAS_INDEX optimal_threshold = max_size;
    
    // Binary search for crossover point
    CBLAS_INDEX low = min_size;
    CBLAS_INDEX high = max_size;
    
    while (low < high) {
        CBLAS_INDEX mid = low + (high - low) / 2;
        
        // Round to nearest power of 2 for cleaner thresholds
        CBLAS_INDEX test_size = 1;
        while (test_size < mid) {
            test_size *= 2;
        }
        
        if (test_size > max_size) {
            test_size = max_size;
        }
        
        // Benchmark single-threaded
        double time_st = benchmark_operation(operation_func, test_size, 0);
        
        // Benchmark multi-threaded
        double time_mt = benchmark_operation(operation_func, test_size, 1);
        
        if (time_st < 0 || time_mt < 0) {
            // Benchmark failed, use default
            break;
        }
        
        // Calculate speedup
        float speedup = (float)time_st / (float)time_mt;
        
        if (speedup >= AUTOTUNE_SPEEDUP_THRESHOLD) {
            // MT is beneficial at this size, try smaller
            optimal_threshold = test_size;
            high = mid - 1;
        } else {
            // MT not beneficial yet, try larger
            low = mid + 1;
        }
    }
    
    return optimal_threshold;
}

/**
 * @brief Reset all MT thresholds to default compile-time values
 */
void cblas_reset_thresholds(void)
{
    cblas_mt_dot_threshold = CBLAS_MT_DOT_DEFAULT;
    cblas_mt_axpy_threshold = CBLAS_MT_AXPY_DEFAULT;
    cblas_mt_copy_threshold = CBLAS_MT_COPY_DEFAULT;
    cblas_mt_ger_threshold = CBLAS_MT_GER_DEFAULT;
    cblas_mt_gemm_threshold = CBLAS_MT_GEMM_DEFAULT;
    cblas_mt_gemv_threshold = CBLAS_MT_GEMV_DEFAULT;
}

/**
 * @brief Auto-tune multi-threading thresholds based on runtime benchmarks
 */
void cblas_autotune_thresholds(void)
{
#ifdef MT_ENABLED
    int num_threads = cblas_get_num_threads();
    
    // Only tune if we have more than 1 thread
    if (num_threads <= 1) {
        cblas_reset_thresholds();
        return;
    }
    
    printf("CBLAS: Auto-tuning MT thresholds for %d threads...\n", num_threads);
    
    // Tune Level-1 operations (vector-vector)
    printf("  Calibrating DOT threshold... ");
    fflush(stdout);
    cblas_mt_dot_threshold = find_mt_threshold(bench_sdot, AUTOTUNE_MIN_SIZE, AUTOTUNE_MAX_SIZE);
    printf("%lu\n", (unsigned long)cblas_mt_dot_threshold);
    
    printf("  Calibrating COPY threshold... ");
    fflush(stdout);
    cblas_mt_copy_threshold = find_mt_threshold(bench_scopy, AUTOTUNE_MIN_SIZE, AUTOTUNE_MAX_SIZE);
    printf("%lu\n", (unsigned long)cblas_mt_copy_threshold);
    
    printf("  Calibrating AXPY threshold... ");
    fflush(stdout);
    cblas_mt_axpy_threshold = find_mt_threshold(bench_saxpy, AUTOTUNE_MIN_SIZE, AUTOTUNE_MAX_SIZE);
    printf("%lu\n", (unsigned long)cblas_mt_axpy_threshold);
    
    // For Level-2 and Level-3, use heuristics based on Level-1 results
    // GER is matrix-level, use lower threshold (1/8 of COPY threshold)
    // Apply minimum threshold of max(CBLAS_MT_GER_DEFAULT/2, 1024)
    cblas_mt_ger_threshold = cblas_mt_copy_threshold / 8;
    CBLAS_INDEX min_ger_threshold = (CBLAS_MT_GER_DEFAULT / 2 > 1024) ? 
                                    CBLAS_MT_GER_DEFAULT / 2 : 1024;
    if (cblas_mt_ger_threshold < min_ger_threshold) {
        cblas_mt_ger_threshold = min_ger_threshold;
    }
    
    // GEMV and GEMM are compute-intensive, use lower thresholds
    cblas_mt_gemv_threshold = cblas_mt_ger_threshold;
    cblas_mt_gemm_threshold = cblas_mt_ger_threshold;
    
    printf("  GER/GEMV/GEMM thresholds (heuristic): %lu\n", 
           (unsigned long)cblas_mt_ger_threshold);
    
    printf("CBLAS: Auto-tuning complete.\n");
#else
    cblas_reset_thresholds();
#endif
}
