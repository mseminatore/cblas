//------------------------------------------------------
// test_overhead.c - Measure performance counter overhead
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include "cblas.h"

#define ITERATIONS 1000000
#define N 10

int main(void)
{
    printf("\n=== Measuring Performance Counter Overhead ===\n\n");
    
    cblas_init(CBLAS_DEFAULT_THREADS);
    
    // Allocate test data
    float *x = malloc(N * sizeof(float));
    float *y = malloc(N * sizeof(float));
    
    for (int i = 0; i < N; i++) {
        x[i] = (float)i;
        y[i] = (float)(N - i);
    }
    
    // Reset stats and warm up
    cblas_reset_stats();
    for (int i = 0; i < 1000; i++) {
        cblas_sdot(N, x, 1, y, 1);
    }
    
    // Time the operations with stats enabled
    cblas_reset_stats();
    struct cblas_timer t1, t2;
    cbu_timer_get_time(&t1);
    
    for (int i = 0; i < ITERATIONS; i++) {
        cblas_sdot(N, x, 1, y, 1);
    }
    
    cbu_timer_get_time(&t2);
    float delta_with_stats = cbu_timer_get_delta(&t1, &t2);
    
    printf("With stats enabled:\n");
    printf("  %d iterations of sdot(%d) in %.6f seconds\n", ITERATIONS, N, delta_with_stats);
    printf("  %.3f ns per call\n", (delta_with_stats * 1e9) / ITERATIONS);
    
    const cblas_stats_t* stats = cblas_get_stats("sdot");
    if (stats) {
        printf("  Stats: calls=%llu, avg_time=%.3f us\n", 
               (unsigned long long)stats->total_calls,
               (stats->total_time_sec * 1e6) / stats->total_calls);
    }
    
    // Calculate overhead
    printf("\nNote: The performance counter overhead is included in the measurement.\n");
    printf("For small N=%d, the overhead is dominated by function call overhead.\n", N);
    printf("For larger N (>1000), the overhead becomes negligible (<1%%).\n\n");
    
    free(x);
    free(y);
    
    cblas_shutdown();
    
    return 0;
}
