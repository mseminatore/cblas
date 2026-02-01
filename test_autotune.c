//------------------------------------------------------
// test_autotune.c - Test auto-tuning infrastructure
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cblas.h"
#include "test.h"

/**
 * @brief Test that default thresholds are set correctly
 */
static int test_default_thresholds(void)
{
    cblas_init(4);
    
    // Reset to defaults (this is what happens when CBLAS_AUTO_TUNE is not set)
    cblas_reset_thresholds();
    
    // Verify defaults match expected values
    if (cblas_mt_dot_threshold != CBLAS_MT_DOT_DEFAULT) {
        fprintf(stderr, "DOT threshold mismatch: expected %d, got %lu\n",
                CBLAS_MT_DOT_DEFAULT, (unsigned long)cblas_mt_dot_threshold);
        cblas_shutdown();
        return 0;
    }
    
    if (cblas_mt_axpy_threshold != CBLAS_MT_AXPY_DEFAULT) {
        fprintf(stderr, "AXPY threshold mismatch: expected %d, got %lu\n",
                CBLAS_MT_AXPY_DEFAULT, (unsigned long)cblas_mt_axpy_threshold);
        cblas_shutdown();
        return 0;
    }
    
    if (cblas_mt_copy_threshold != CBLAS_MT_COPY_DEFAULT) {
        fprintf(stderr, "COPY threshold mismatch: expected %d, got %lu\n",
                CBLAS_MT_COPY_DEFAULT, (unsigned long)cblas_mt_copy_threshold);
        cblas_shutdown();
        return 0;
    }
    
    if (cblas_mt_ger_threshold != CBLAS_MT_GER_DEFAULT) {
        fprintf(stderr, "GER threshold mismatch: expected %d, got %lu\n",
                CBLAS_MT_GER_DEFAULT, (unsigned long)cblas_mt_ger_threshold);
        cblas_shutdown();
        return 0;
    }
    
    if (cblas_mt_gemm_threshold != CBLAS_MT_GEMM_DEFAULT) {
        fprintf(stderr, "GEMM threshold mismatch: expected %d, got %lu\n",
                CBLAS_MT_GEMM_DEFAULT, (unsigned long)cblas_mt_gemm_threshold);
        cblas_shutdown();
        return 0;
    }
    
    if (cblas_mt_gemv_threshold != CBLAS_MT_GEMV_DEFAULT) {
        fprintf(stderr, "GEMV threshold mismatch: expected %d, got %lu\n",
                CBLAS_MT_GEMV_DEFAULT, (unsigned long)cblas_mt_gemv_threshold);
        cblas_shutdown();
        return 0;
    }
    
    cblas_shutdown();
    return 1;
}

/**
 * @brief Test that auto-tuning modifies thresholds
 */
static int test_autotune_modifies_thresholds(void)
{
    cblas_init(4);
    
    // Store default values
    CBLAS_INDEX default_dot = CBLAS_MT_DOT_DEFAULT;
    CBLAS_INDEX default_axpy = CBLAS_MT_AXPY_DEFAULT;
    CBLAS_INDEX default_copy = CBLAS_MT_COPY_DEFAULT;
    
    // Run auto-tune
    cblas_autotune_thresholds();
    
    // Check that at least one threshold was modified
    // (On a 2-core system with limited memory bandwidth, thresholds may be higher)
    int modified = 0;
    
    if (cblas_mt_dot_threshold != default_dot) {
        modified = 1;
        printf("  DOT threshold changed: %lu -> %lu\n", 
               (unsigned long)default_dot, (unsigned long)cblas_mt_dot_threshold);
    }
    
    if (cblas_mt_axpy_threshold != default_axpy) {
        modified = 1;
        printf("  AXPY threshold changed: %lu -> %lu\n", 
               (unsigned long)default_axpy, (unsigned long)cblas_mt_axpy_threshold);
    }
    
    if (cblas_mt_copy_threshold != default_copy) {
        modified = 1;
        printf("  COPY threshold changed: %lu -> %lu\n", 
               (unsigned long)default_copy, (unsigned long)cblas_mt_copy_threshold);
    }
    
    // Always modified by design (heuristic calculation)
    if (cblas_mt_ger_threshold != CBLAS_MT_GER_DEFAULT) {
        modified = 1;
        printf("  GER threshold changed: %d -> %lu\n", 
               CBLAS_MT_GER_DEFAULT, (unsigned long)cblas_mt_ger_threshold);
    }
    
    cblas_shutdown();
    
    // On 2-thread system, auto-tune should modify at least GER/GEMV/GEMM
    if (!modified) {
        fprintf(stderr, "Auto-tune did not modify any thresholds\n");
        return 0;
    }
    
    return 1;
}

/**
 * @brief Test that operations still work correctly after auto-tuning
 */
static int test_operations_after_autotune(void)
{
    cblas_init(4);
    cblas_autotune_thresholds();
    
    // Test DOT
    const size_t n = 1000;
    float *x = malloc(n * sizeof(float));
    float *y = malloc(n * sizeof(float));
    
    if (!x || !y) {
        fprintf(stderr, "Memory allocation failed\n");
        free(x);
        free(y);
        cblas_shutdown();
        return 0;
    }
    
    for (size_t i = 0; i < n; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }
    
    float result = cblas_sdot(n, x, 1, y, 1);
    float expected = (float)n * 2.0f;
    
    if (fabs(result - expected) > EPSILON) {
        fprintf(stderr, "DOT failed after autotune: got %f, expected %f\n", 
                result, expected);
        free(x);
        free(y);
        cblas_shutdown();
        return 0;
    }
    
    // Test COPY
    for (size_t i = 0; i < n; i++) {
        y[i] = 0.0f;
    }
    
    cblas_scopy(n, x, 1, y, 1);
    
    for (size_t i = 0; i < n; i++) {
        if (fabs(y[i] - x[i]) > EPSILON) {
            fprintf(stderr, "COPY failed after autotune at index %zu\n", i);
            free(x);
            free(y);
            cblas_shutdown();
            return 0;
        }
    }
    
    // Test AXPY
    for (size_t i = 0; i < n; i++) {
        x[i] = 1.0f;
        y[i] = 1.0f;
    }
    
    cblas_saxpy(n, 2.0f, x, 1, y, 1);
    
    for (size_t i = 0; i < n; i++) {
        float expected_y = 3.0f;  // 1.0 + 2.0 * 1.0
        if (fabs(y[i] - expected_y) > EPSILON) {
            fprintf(stderr, "AXPY failed after autotune at index %zu\n", i);
            free(x);
            free(y);
            cblas_shutdown();
            return 0;
        }
    }
    
    free(x);
    free(y);
    cblas_shutdown();
    return 1;
}

/**
 * @brief Test that reset_thresholds restores defaults
 */
static int test_reset_thresholds(void)
{
    cblas_init(4);
    
    // Auto-tune
    cblas_autotune_thresholds();
    
    // Reset
    cblas_reset_thresholds();
    
    // Verify all are back to defaults
    if (cblas_mt_dot_threshold != CBLAS_MT_DOT_DEFAULT ||
        cblas_mt_axpy_threshold != CBLAS_MT_AXPY_DEFAULT ||
        cblas_mt_copy_threshold != CBLAS_MT_COPY_DEFAULT ||
        cblas_mt_ger_threshold != CBLAS_MT_GER_DEFAULT ||
        cblas_mt_gemm_threshold != CBLAS_MT_GEMM_DEFAULT ||
        cblas_mt_gemv_threshold != CBLAS_MT_GEMV_DEFAULT) {
        fprintf(stderr, "Reset thresholds did not restore all defaults\n");
        cblas_shutdown();
        return 0;
    }
    
    cblas_shutdown();
    return 1;
}

/**
 * @brief Test auto-tuning with single thread (should set high thresholds)
 */
static int test_autotune_single_thread(void)
{
    cblas_init(1);
    
    // Auto-tune with 1 thread should reset to defaults
    cblas_autotune_thresholds();
    
    // Verify defaults are used
    if (cblas_mt_dot_threshold != CBLAS_MT_DOT_DEFAULT ||
        cblas_mt_axpy_threshold != CBLAS_MT_AXPY_DEFAULT ||
        cblas_mt_copy_threshold != CBLAS_MT_COPY_DEFAULT) {
        fprintf(stderr, "Auto-tune with 1 thread should use defaults\n");
        cblas_shutdown();
        return 0;
    }
    
    cblas_shutdown();
    return 1;
}

int test_main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;
    
    BEGIN_TESTS();
    
    MODULE("Auto-tuning Infrastructure");
    
    SUITE("Threshold management");
    TESTEX("Default thresholds", test_default_thresholds());
    TESTEX("Reset thresholds", test_reset_thresholds());
    
    SUITE("Auto-tuning");
    TESTEX("Auto-tune modifies thresholds", test_autotune_modifies_thresholds());
    TESTEX("Operations work after auto-tune", test_operations_after_autotune());
    TESTEX("Auto-tune with single thread", test_autotune_single_thread());
    
    END_TESTS();
}
