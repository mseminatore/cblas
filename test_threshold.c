//------------------------------------------------------
// test_threshold.c - Test MT threshold enforcement
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cblas.h"
#include "test.h"

// Counter variables for test framework
int test_number = 0;
int test_suites = 0;
int test_failures = 0;
int test_modules = 0;

// Test helper to verify threshold behavior
static int test_mt_threshold_enforcement(void)
{
    cblas_init(4);  // Initialize with 4 threads
    
    // Test 1: Small vectors (below threshold) should not use MT
    {
        const size_t n_small = CBLAS_MT_COPY / 2;  // Half the threshold
        float *x = malloc(n_small * sizeof(float));
        float *y = malloc(n_small * sizeof(float));
        
        if (!x || !y) {
            fprintf(stderr, "Memory allocation failed\n");
            free(x);
            free(y);
            return 0;
        }
        
        // Initialize
        for (size_t i = 0; i < n_small; i++) {
            x[i] = (float)i;
            y[i] = 0.0f;
        }
        
        // Reset stats
        cblas_reset_stats();
        
        // Test cblas_scopy with small n
        cblas_scopy(n_small, x, 1, y, 1);
        
        // Verify result
        int copy_ok = 1;
        for (size_t i = 0; i < n_small; i++) {
            if (y[i] != x[i]) {
                copy_ok = 0;
                break;
            }
        }
        
        free(x);
        free(y);
        
        if (!copy_ok) {
            fprintf(stderr, "cblas_scopy failed for small n\n");
            return 0;
        }
    }
    
    // Test 2: Large vectors (above threshold) should use MT
    {
        const size_t n_large = CBLAS_MT_COPY * 2;  // Double the threshold
        float *x = malloc(n_large * sizeof(float));
        float *y = malloc(n_large * sizeof(float));
        
        if (!x || !y) {
            fprintf(stderr, "Memory allocation failed\n");
            free(x);
            free(y);
            return 0;
        }
        
        // Initialize
        for (size_t i = 0; i < n_large; i++) {
            x[i] = (float)i;
            y[i] = 0.0f;
        }
        
        // Reset stats
        cblas_reset_stats();
        
        // Test cblas_scopy with large n
        cblas_scopy(n_large, x, 1, y, 1);
        
        // Verify result
        int copy_ok = 1;
        for (size_t i = 0; i < n_large; i++) {
            if (y[i] != x[i]) {
                copy_ok = 0;
                break;
            }
        }
        
        free(x);
        free(y);
        
        if (!copy_ok) {
            fprintf(stderr, "cblas_scopy failed for large n\n");
            return 0;
        }
    }
    
    // Test 3: Test cblas_sdot with small and large n
    {
        const size_t n_small = CBLAS_MT_DOT / 2;
        float *x = malloc(n_small * sizeof(float));
        float *y = malloc(n_small * sizeof(float));
        
        if (!x || !y) {
            fprintf(stderr, "Memory allocation failed\n");
            free(x);
            free(y);
            return 0;
        }
        
        for (size_t i = 0; i < n_small; i++) {
            x[i] = 1.0f;
            y[i] = 2.0f;
        }
        
        float result = cblas_sdot(n_small, x, 1, y, 1);
        float expected = (float)n_small * 2.0f;
        
        free(x);
        free(y);
        
        if (fabs(result - expected) > EPSILON) {
            fprintf(stderr, "cblas_sdot failed: got %f, expected %f\n", result, expected);
            return 0;
        }
    }
    
    // Test 4: Test cblas_sswap with threshold
    {
        const size_t n_small = CBLAS_MT_COPY / 2;
        float *x = malloc(n_small * sizeof(float));
        float *y = malloc(n_small * sizeof(float));
        float *x_orig = malloc(n_small * sizeof(float));
        float *y_orig = malloc(n_small * sizeof(float));
        
        if (!x || !y || !x_orig || !y_orig) {
            fprintf(stderr, "Memory allocation failed\n");
            free(x);
            free(y);
            free(x_orig);
            free(y_orig);
            return 0;
        }
        
        for (size_t i = 0; i < n_small; i++) {
            x[i] = x_orig[i] = (float)i;
            y[i] = y_orig[i] = (float)(n_small - i);
        }
        
        cblas_sswap(n_small, x, 1, y, 1);
        
        int swap_ok = 1;
        for (size_t i = 0; i < n_small; i++) {
            if (x[i] != y_orig[i] || y[i] != x_orig[i]) {
                swap_ok = 0;
                break;
            }
        }
        
        free(x);
        free(y);
        free(x_orig);
        free(y_orig);
        
        if (!swap_ok) {
            fprintf(stderr, "cblas_sswap failed\n");
            return 0;
        }
    }
    
    cblas_shutdown();
    return 1;
}

int main(void)
{
    BEGIN_TESTS();
    
    MODULE("MT Threshold Enforcement");
    
    SUITE("cblas MT threshold tests");
    TESTEX("MT threshold enforcement", test_mt_threshold_enforcement());
    
    END_TESTS();
}
