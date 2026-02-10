//------------------------------------------------------
// test_dot_threshold.c - Validate CBLAS_MT_DOT threshold
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cblas.h"
#include "test.h"

// Counter variables for test framework
int test_number = 0;
int test_suites = 0;
int test_failures = 0;
int test_modules = 0;

//------------------------------------------------------
// Test that MT doesn't activate below threshold
//------------------------------------------------------
static int test_mt_below_threshold(void)
{
    cblas_init(CBLAS_DEFAULT_THREADS);
    
    // Test size below threshold
    const CBLAS_INDEX n = CBLAS_MT_DOT / 2;
    float *x = malloc(n * sizeof(float));
    float *y = malloc(n * sizeof(float));
    
    if (!x || !y) {
        fprintf(stderr, "Memory allocation failed\n");
        free(x);
        free(y);
        return 0;
    }
    
    // Initialize
    for (CBLAS_INDEX i = 0; i < n; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }
    
    // Reset stats
    cblas_reset_stats();
    
    // Call sdot
    float result = cblas_sdot(n, x, 1, y, 1);
    
    // Check result
    float expected = (float)n * 2.0f;
    int result_ok = (fabs(result - expected) < EPSILON);
    
    // Check that MT was NOT used
    const cblas_stats_t* stats = cblas_get_stats("sdot");
    int mt_not_used = (stats && stats->mt_activations == 0);
    
    free(x);
    free(y);
    
    cblas_shutdown();
    
    return result_ok && mt_not_used;
}

//------------------------------------------------------
// Test that MT activates above threshold
//------------------------------------------------------
static int test_mt_above_threshold(void)
{
    cblas_init(CBLAS_DEFAULT_THREADS);
    
    // Test size well above threshold (2x)
    const CBLAS_INDEX n = CBLAS_MT_DOT * 2;
    float *x = malloc(n * sizeof(float));
    float *y = malloc(n * sizeof(float));
    
    if (!x || !y) {
        fprintf(stderr, "Memory allocation failed\n");
        free(x);
        free(y);
        return 0;
    }
    
    // Initialize
    for (CBLAS_INDEX i = 0; i < n; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }
    
    // Reset stats
    cblas_reset_stats();
    
    // Call sdot
    float result = cblas_sdot(n, x, 1, y, 1);
    
    // Check result
    float expected = (float)n * 2.0f;
    int result_ok = (fabs(result - expected) < EPSILON);
    
    // Check that MT WAS used
    const cblas_stats_t* stats = cblas_get_stats("sdot");
    int mt_used = (stats && stats->mt_activations > 0);
    
    free(x);
    free(y);
    
    cblas_shutdown();
    
    return result_ok && mt_used;
}

//------------------------------------------------------
// Test threshold value is reasonable
//------------------------------------------------------
static int test_threshold_value(void)
{
    // Verify threshold is set to a reasonable value
    // Testing shows MT overhead exceeds benefit for Level-1 ops until ~500K elements
    // on modern CPUs with fast memory. The threshold should be high enough to
    // avoid MT overhead hurting small/medium vector operations.
    // Threshold should be >= 100000 (MT overhead is significant for memory-bound ops)
    // and <= 10000000 (still allow MT for very large vectors)
    int threshold_ok = (CBLAS_MT_DOT >= 100000 && CBLAS_MT_DOT <= 10000000);
    
    if (!threshold_ok) {
        fprintf(stderr, "CBLAS_MT_DOT (%lu) is outside reasonable range (100000-10000000)\n", 
                (unsigned long)CBLAS_MT_DOT);
    }
    
    return threshold_ok;
}

//------------------------------------------------------
// Test correctness at boundary
//------------------------------------------------------
static int test_at_boundary(void)
{
    cblas_init(CBLAS_DEFAULT_THREADS);
    
    // Test at exactly threshold and threshold+1
    CBLAS_INDEX sizes[] = {CBLAS_MT_DOT, CBLAS_MT_DOT + 1};
    
    for (int i = 0; i < 2; i++) {
        CBLAS_INDEX n = sizes[i];
        float *x = malloc(n * sizeof(float));
        float *y = malloc(n * sizeof(float));
        
        if (!x || !y) {
            fprintf(stderr, "Memory allocation failed\n");
            free(x);
            free(y);
            cblas_shutdown();
            return 0;
        }
        
        // Initialize
        for (CBLAS_INDEX j = 0; j < n; j++) {
            x[j] = 1.0f;
            y[j] = 2.0f;
        }
        
        // Call sdot
        float result = cblas_sdot(n, x, 1, y, 1);
        
        // Check result
        float expected = (float)n * 2.0f;
        if (fabs(result - expected) >= EPSILON) {
            fprintf(stderr, "Incorrect result at n=%zu: got %f, expected %f\n", 
                    n, result, expected);
            free(x);
            free(y);
            cblas_shutdown();
            return 0;
        }
        
        free(x);
        free(y);
    }
    
    cblas_shutdown();
    return 1;
}

int main(void)
{
    BEGIN_TESTS();
    
    MODULE("Dot Product Threshold Validation");
    
    SUITE("CBLAS_MT_DOT threshold tests");
    TESTEX("Threshold value is reasonable", test_threshold_value());
    TESTEX("MT doesn't activate below threshold", test_mt_below_threshold());
    TESTEX("MT activates above threshold", test_mt_above_threshold());
    TESTEX("Correctness at boundary", test_at_boundary());
    
    END_TESTS();
}
