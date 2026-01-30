//------------------------------------------------------
// dot_threshold_tuning.c - Tune CBLAS_MT_DOT threshold
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "cblas.h"

#define NUM_WARMUP 3
#define NUM_ITERATIONS 10

// Test multiple problem sizes around potential threshold values
static const CBLAS_INDEX test_sizes[] = {
    1000,    2000,    4000,    5000,    
    8000,    10000,   12000,   15000,   
    20000,   25000,   30000,   40000,   
    50000,   75000,   100000,  150000,  
    200000,  300000,  500000,  1000000,
    2000000, 4000000, 8000000
};
#define NUM_TEST_SIZES (sizeof(test_sizes) / sizeof(test_sizes[0]))

//------------------------------------------------------
// Measure performance for a single problem size
//------------------------------------------------------
static double measure_sdot_performance(CBLAS_INDEX n, float *x, float *y)
{
    struct cblas_timer t1, t2;
    double min_time = 1e9;
    
    // Warmup
    for (int i = 0; i < NUM_WARMUP; i++) {
        cblas_sdot(n, x, 1, y, 1);
    }
    
    // Measure
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        cbu_timer_get_time(&t1);
        cblas_sdot(n, x, 1, y, 1);
        cbu_timer_get_time(&t2);
        
        double dt = cbu_timer_get_delta(&t1, &t2);
        if (dt < min_time) {
            min_time = dt;
        }
    }
    
    return min_time;
}

//------------------------------------------------------
// Main tuning function
//------------------------------------------------------
void tune_threshold(void)
{
    printf("=== CBLAS_MT_DOT Threshold Tuning ===\n\n");
    printf("Testing performance across various problem sizes\n");
    printf("to determine optimal multi-threading threshold.\n\n");
    
    // Allocate test vectors (use maximum size)
    CBLAS_INDEX max_size = test_sizes[NUM_TEST_SIZES - 1];
    float *x = malloc(max_size * sizeof(float));
    float *y = malloc(max_size * sizeof(float));
    
    if (!x || !y) {
        fprintf(stderr, "Memory allocation failed\n");
        free(x);
        free(y);
        return;
    }
    
    // Initialize vectors
    for (CBLAS_INDEX i = 0; i < max_size; i++) {
        x[i] = (float)(i % 100) / 100.0f;
        y[i] = (float)((i + 1) % 100) / 100.0f;
    }
    
    printf("%-10s %12s %12s %12s %12s\n", 
           "Size", "Time(s)", "GFlops", "MT Used", "Performance");
    printf("----------------------------------------------------------------\n");
    
    // Test each size
    for (size_t i = 0; i < NUM_TEST_SIZES; i++) {
        CBLAS_INDEX n = test_sizes[i];
        
        cblas_reset_stats();
        double time_sec = measure_sdot_performance(n, x, y);
        
        // Calculate performance metrics
        double gflops = (2.0 * n) / time_sec / 1e9;
        
        // Check if MT was used
        const cblas_stats_t* stats = cblas_get_stats("sdot");
        int mt_used = (stats && stats->mt_activations > 0) ? 1 : 0;
        
        // Calculate relative performance (normalized to first measurement)
        static double baseline = 0.0;
        if (i == 0) baseline = gflops;
        double relative_perf = gflops / baseline;
        
        printf("%-10zu %12.6f %12.2f %12s %12.2fx\n", 
               n, time_sec, gflops, 
               mt_used ? "Yes" : "No",
               relative_perf);
    }
    
    printf("\n");
    
    // Analysis section
    printf("=== Analysis ===\n\n");
    printf("Current CBLAS_MT_DOT threshold: %d elements\n\n", CBLAS_MT_DOT);
    
    printf("Recommendations:\n");
    printf("1. Look for the size where MT activation starts providing benefit\n");
    printf("2. The optimal threshold is typically where:\n");
    printf("   - Performance with MT >= Performance without MT\n");
    printf("   - MT overhead is amortized by parallelism gains\n");
    printf("3. Consider cache effects and thread overhead\n");
    printf("4. Test on representative hardware (different core counts)\n\n");
    
    printf("Based on this data:\n");
    printf("- If MT consistently improves performance at sizes > X,\n");
    printf("  consider setting CBLAS_MT_DOT = X\n");
    printf("- If MT shows no benefit or degrades performance,\n");
    printf("  consider increasing CBLAS_MT_DOT\n");
    printf("- The crossover point is typically visible in the data\n\n");
    
    free(x);
    free(y);
}

//------------------------------------------------------
// Compare performance with different thread counts
//------------------------------------------------------
void compare_thread_counts(void)
{
    printf("\n=== Thread Count Comparison ===\n\n");
    printf("Testing how different thread counts affect performance\n");
    printf("at sizes near current threshold (%d)\n\n", CBLAS_MT_DOT);
    
    // Test sizes around the threshold
    CBLAS_INDEX test_n[] = {
        CBLAS_MT_DOT / 2,     // Below threshold
        CBLAS_MT_DOT,         // At threshold
        CBLAS_MT_DOT * 2,     // Above threshold
        CBLAS_MT_DOT * 4      // Well above threshold
    };
    
    int thread_counts[] = {1, 2, 4, 8};
    
    for (size_t size_idx = 0; size_idx < 4; size_idx++) {
        CBLAS_INDEX n = test_n[size_idx];
        
        float *x = malloc(n * sizeof(float));
        float *y = malloc(n * sizeof(float));
        
        if (!x || !y) {
            fprintf(stderr, "Memory allocation failed\n");
            free(x);
            free(y);
            continue;
        }
        
        for (CBLAS_INDEX i = 0; i < n; i++) {
            x[i] = (float)(i % 100) / 100.0f;
            y[i] = (float)((i + 1) % 100) / 100.0f;
        }
        
        printf("Problem size: %zu elements\n", n);
        printf("%-10s %12s %12s %12s\n", 
               "Threads", "Time(s)", "GFlops", "Speedup");
        printf("------------------------------------------------\n");
        
        double baseline_gflops = 0.0;
        
        for (size_t t = 0; t < 4; t++) {
            int num_threads = thread_counts[t];
            
            cblas_shutdown();
            cblas_init(num_threads);
            cblas_reset_stats();
            
            double time_sec = measure_sdot_performance(n, x, y);
            double gflops = (2.0 * n) / time_sec / 1e9;
            
            if (t == 0) baseline_gflops = gflops;
            double speedup = gflops / baseline_gflops;
            
            printf("%-10d %12.6f %12.2f %12.2fx\n", 
                   num_threads, time_sec, gflops, speedup);
        }
        
        printf("\n");
        
        free(x);
        free(y);
    }
    
    // Restore default
    cblas_shutdown();
    cblas_init(CBLAS_DEFAULT_THREADS);
}

//------------------------------------------------------
// Main
//------------------------------------------------------
int main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;
    
    cblas_init(CBLAS_DEFAULT_THREADS);
    cblas_print_configuration();
    
    tune_threshold();
    compare_thread_counts();
    
    cblas_shutdown();
    
    return 0;
}
