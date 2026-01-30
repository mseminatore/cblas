//------------------------------------------------------
// dot_threshold_tuning_large.c - Test larger sizes for MT threshold
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "cblas.h"

#define NUM_WARMUP 3
#define NUM_ITERATIONS 5

// Test larger problem sizes to find where MT becomes beneficial
static const CBLAS_INDEX test_sizes[] = {
    50000,    100000,   200000,   300000,   
    500000,   750000,   1000000,  1500000,  
    2000000,  3000000,  5000000,  7500000,  
    10000000, 15000000, 20000000
};
#define NUM_TEST_SIZES (sizeof(test_sizes) / sizeof(test_sizes[0]))

//------------------------------------------------------
// Measure performance with and without MT
//------------------------------------------------------
static void compare_mt_vs_no_mt(CBLAS_INDEX n)
{
    float *x = malloc(n * sizeof(float));
    float *y = malloc(n * sizeof(float));
    
    if (!x || !y) {
        fprintf(stderr, "Memory allocation failed\n");
        free(x);
        free(y);
        return;
    }
    
    for (CBLAS_INDEX i = 0; i < n; i++) {
        x[i] = (float)(i % 100) / 100.0f;
        y[i] = (float)((i + 1) % 100) / 100.0f;
    }
    
    struct cblas_timer t1, t2;
    
    // Test with 1 thread (effectively no MT)
    cblas_shutdown();
    cblas_init(1);
    
    // Warmup
    for (int i = 0; i < NUM_WARMUP; i++) {
        cblas_sdot(n, x, 1, y, 1);
    }
    
    cbu_timer_get_time(&t1);
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        cblas_sdot(n, x, 1, y, 1);
    }
    cbu_timer_get_time(&t2);
    
    double time_1thread = cbu_timer_get_delta(&t1, &t2) / NUM_ITERATIONS;
    double gflops_1thread = (2.0 * n) / time_1thread / 1e9;
    
    // Test with default threads
    cblas_shutdown();
    cblas_init(CBLAS_DEFAULT_THREADS);
    
    // Warmup
    for (int i = 0; i < NUM_WARMUP; i++) {
        cblas_sdot(n, x, 1, y, 1);
    }
    
    cbu_timer_get_time(&t1);
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        cblas_sdot(n, x, 1, y, 1);
    }
    cbu_timer_get_time(&t2);
    
    double time_mt = cbu_timer_get_delta(&t1, &t2) / NUM_ITERATIONS;
    double gflops_mt = (2.0 * n) / time_mt / 1e9;
    
    double speedup = time_1thread / time_mt;
    const char* verdict = (speedup >= 1.1) ? "MT WINS" : 
                          (speedup <= 0.9) ? "1T WINS" : "TIE";
    
    printf("%-10zu %12.6f %10.2f %12.6f %10.2f %10.2fx   %s\n", 
           n, time_1thread, gflops_1thread, 
           time_mt, gflops_mt, speedup, verdict);
    
    free(x);
    free(y);
}

int main(void)
{
    printf("=== Large Size MT vs Single Thread Comparison ===\n\n");
    
    cblas_init(CBLAS_DEFAULT_THREADS);
    cblas_print_configuration();
    
    printf("\nTesting to find crossover point where MT becomes beneficial\n\n");
    printf("%-10s %12s %10s %12s %10s %10s   %s\n", 
           "Size", "Time-1T(s)", "GF-1T", "Time-MT(s)", "GF-MT", "Speedup", "Winner");
    printf("-----------------------------------------------------------------------------------\n");
    
    for (size_t i = 0; i < NUM_TEST_SIZES; i++) {
        compare_mt_vs_no_mt(test_sizes[i]);
    }
    
    printf("\n=== Recommendation ===\n");
    printf("Look for the size where 'MT WINS' consistently appears.\n");
    printf("Set CBLAS_MT_DOT to a value just below that crossover point.\n");
    printf("Consider: safety margin, cache effects, and different hardware.\n\n");
    
    cblas_shutdown();
    return 0;
}
