//------------------------------------------------------
//
// Copyright 2023-2026 Mark Seminatore. All rights reserved.
//
// copy_perf.c - Performance testing for cblas_scopy/dcopy
//
//------------------------------------------------------

#include "cblas.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Timer helper
static double get_time(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

#define ITERATIONS 100

int main(void)
{
    printf("CBLAS COPY Performance Test\n");
    printf("============================\n\n");

    cblas_init(CBLAS_DEFAULT_THREADS);

    // Print table header
    printf("%-10s  %-12s  %-12s  %-12s\n", "Size", "Time (s)", "Bandwidth", "Ops/s");
    printf("%-10s  %-12s  %-12s  %-12s\n", "----------", "------------", "------------", "------------");

    // Test various sizes
    const int sizes[] = {
        4, 8, 16, 32, 64, 128, 256, 512,
        1024, 2048, 4096, 8192, 16384,
        32768, 65536, 131072, 262144,
        524288, 1048576, 2097152, 4194304
    };
    const int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int s = 0; s < num_sizes; s++)
    {
        int n = sizes[s];
        
        // Allocate and initialize vectors
        // aligned_alloc requires size to be multiple of alignment
        size_t alloc_size = ((n * sizeof(float) + 63) / 64) * 64;
        float *x = (float *)aligned_alloc(64, alloc_size);
        float *y = (float *)aligned_alloc(64, alloc_size);
        
        if (!x || !y) {
            fprintf(stderr, "Allocation failed for size %d\n", n);
            continue;
        }

        // Initialize x with random data
        for (int i = 0; i < n; i++) {
            x[i] = (float)(rand() % 100) / 10.0f;
        }
        
        // Warmup
        for (int i = 0; i < 5; i++) {
            cblas_scopy(n, x, 1, y, 1);
        }

        // Benchmark
        double start = get_time();
        for (int i = 0; i < ITERATIONS; i++) {
            cblas_scopy(n, x, 1, y, 1);
        }
        double end = get_time();
        
        double elapsed = end - start;
        double ops_per_iter = n;  // 1 load + 1 store per element
        double total_ops = ops_per_iter * ITERATIONS;
        double gflops = (total_ops / elapsed) / 1e9;
        
        // Calculate bandwidth: 1 read + 1 write = 2 * sizeof(float) bytes per element
        double bytes_per_iter = n * 2.0 * sizeof(float);
        double total_bytes = bytes_per_iter * ITERATIONS;
        double bandwidth_gbs = (total_bytes / elapsed) / 1e9;

        printf("%-10d  %12.6f  %9.2f GB/s  %9.2f GOps/s\n",
               n, elapsed, bandwidth_gbs, gflops);

        free(x);
        free(y);
    }

    printf("\n");
    cblas_print_stats();
    
    cblas_shutdown();
    return 0;
}
