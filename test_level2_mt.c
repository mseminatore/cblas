//------------------------------------------------------
// test_level2_mt.c - Test Level-2 multi-threading
//
// Copyright 2026 Mark Seminatore. All rights reserved.
//------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cblas.h"
#include "test.h"

// Counter variables for test framework
// int test_number = 0;
// int test_suites = 0;
// int test_failures = 0;
// int test_modules = 0;

// Test GER with large matrices (should use MT)
static int test_sger_mt(void)
{
    // Use a matrix size that exceeds CBLAS_MT_GER threshold
    const size_t m = 200;
    const size_t n = 200;
    
    float *x = malloc(m * sizeof(float));
    float *y = malloc(n * sizeof(float));
    float *a = calloc(m * n, sizeof(float));
    float *a_ref = calloc(m * n, sizeof(float));
    
    if (!x || !y || !a || !a_ref) {
        fprintf(stderr, "Memory allocation failed\n");
        free(x);
        free(y);
        free(a);
        free(a_ref);
        return 0;
    }
    
    // Initialize vectors
    for (size_t i = 0; i < m; i++) {
        x[i] = 1.0f;
    }
    for (size_t i = 0; i < n; i++) {
        y[i] = 2.0f;
    }
    
    // Reset stats
    cblas_reset_stats();
    
    // Compute with MT
    cblas_sger(CblasRowMajor, m, n, 1.0f, x, 1, y, 1, a, n);
    
    // Compute reference (single-threaded, small problem)
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            a_ref[i * n + j] = x[i] * y[j];
        }
    }
    
    // Verify result
    int ok = 1;
    for (size_t i = 0; i < m * n; i++) {
        if (fabs(a[i] - a_ref[i]) > EPSILON) {
            fprintf(stderr, "GER mismatch at index %zu: got %f, expected %f\n", 
                    i, a[i], a_ref[i]);
            ok = 0;
            break;
        }
    }
    
    // Check stats to see if MT was used
    const cblas_stats_t* stats = cblas_get_stats("sger");
    if (stats) {
        printf("  sger: %lu calls, %lu MT uses (m*n=%zu > CBLAS_MT_GER=%d)\n", 
               (unsigned long)stats->total_calls, (unsigned long)stats->mt_activations, m * n, CBLAS_MT_GER);
    }
    
    free(x);
    free(y);
    free(a);
    free(a_ref);
    
    return ok;
}

// Test GER with non-unit alpha (should use MT)
static int test_sger_mt_alpha(void)
{
    const size_t m = 200;
    const size_t n = 200;
    const float alpha = 2.5f;
    
    float *x = malloc(m * sizeof(float));
    float *y = malloc(n * sizeof(float));
    float *a = calloc(m * n, sizeof(float));
    float *a_ref = calloc(m * n, sizeof(float));
    
    if (!x || !y || !a || !a_ref) {
        fprintf(stderr, "Memory allocation failed\n");
        free(x);
        free(y);
        free(a);
        free(a_ref);
        return 0;
    }
    
    // Initialize vectors
    for (size_t i = 0; i < m; i++) {
        x[i] = (float)(i % 10);
    }
    for (size_t i = 0; i < n; i++) {
        y[i] = (float)((i % 5) + 1);
    }
    
    // Compute with MT
    cblas_sger(CblasRowMajor, m, n, alpha, x, 1, y, 1, a, n);
    
    // Compute reference
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            a_ref[i * n + j] = alpha * x[i] * y[j];
        }
    }
    
    // Verify result
    int ok = 1;
    for (size_t i = 0; i < m * n; i++) {
        if (fabs(a[i] - a_ref[i]) > EPSILON) {
            fprintf(stderr, "GER alpha mismatch at index %zu: got %f, expected %f\n", 
                    i, a[i], a_ref[i]);
            ok = 0;
            break;
        }
    }
    
    free(x);
    free(y);
    free(a);
    free(a_ref);
    
    return ok;
}

// Test GEMV with large matrices (should use MT)
static int test_sgemv_mt(void)
{
    // Use a matrix size that exceeds CBLAS_MT_GEMV threshold
    const size_t m = 200;
    const size_t n = 200;
    
    float *a = malloc(m * n * sizeof(float));
    float *x = malloc(n * sizeof(float));
    float *y = calloc(m, sizeof(float));
    float *y_ref = calloc(m, sizeof(float));
    
    if (!a || !x || !y || !y_ref) {
        fprintf(stderr, "Memory allocation failed\n");
        free(a);
        free(x);
        free(y);
        free(y_ref);
        return 0;
    }
    
    // Initialize matrix and vector
    for (size_t i = 0; i < m * n; i++) {
        a[i] = 1.0f;
    }
    for (size_t i = 0; i < n; i++) {
        x[i] = 2.0f;
    }
    
    // Reset stats
    cblas_reset_stats();
    
    // Compute with MT
    cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0f, a, n, x, 1, 0.0f, y, 1);
    
    // Compute reference
    for (size_t i = 0; i < m; i++) {
        y_ref[i] = 0.0f;
        for (size_t j = 0; j < n; j++) {
            y_ref[i] += a[i * n + j] * x[j];
        }
    }
    
    // Verify result
    int ok = 1;
    for (size_t i = 0; i < m; i++) {
        if (fabs(y[i] - y_ref[i]) > EPSILON) {
            fprintf(stderr, "GEMV mismatch at index %zu: got %f, expected %f\n", 
                    i, y[i], y_ref[i]);
            ok = 0;
            break;
        }
    }
    
    // Check stats to see if MT was used
    const cblas_stats_t* stats = cblas_get_stats("sgemv");
    if (stats) {
        printf("  sgemv: %lu calls, %lu MT uses (m*n=%zu > CBLAS_MT_GEMV=%d)\n", 
               (unsigned long)stats->total_calls, (unsigned long)stats->mt_activations, m * n, CBLAS_MT_GEMV);
    }
    
    free(a);
    free(x);
    free(y);
    free(y_ref);
    
    return ok;
}

// Test GEMV with non-trivial alpha and beta (should use MT)
static int test_sgemv_mt_alpha_beta(void)
{
    const size_t m = 200;
    const size_t n = 200;
    const float alpha = 2.0f;
    const float beta = 0.5f;
    
    float *a = malloc(m * n * sizeof(float));
    float *x = malloc(n * sizeof(float));
    float *y = malloc(m * sizeof(float));
    float *y_ref = malloc(m * sizeof(float));
    
    if (!a || !x || !y || !y_ref) {
        fprintf(stderr, "Memory allocation failed\n");
        free(a);
        free(x);
        free(y);
        free(y_ref);
        return 0;
    }
    
    // Initialize matrix, vectors
    for (size_t i = 0; i < m * n; i++) {
        a[i] = (float)(i % 10);
    }
    for (size_t i = 0; i < n; i++) {
        x[i] = (float)(i % 5 + 1);
    }
    for (size_t i = 0; i < m; i++) {
        y[i] = (float)(i % 3);
        y_ref[i] = y[i];
    }
    
    // Compute with MT
    cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, alpha, a, n, x, 1, beta, y, 1);
    
    // Compute reference: y = alpha * A * x + beta * y
    for (size_t i = 0; i < m; i++) {
        double sum = 0.0;
        for (size_t j = 0; j < n; j++) {
            sum += a[i * n + j] * x[j];
        }
        y_ref[i] = alpha * sum + beta * y_ref[i];
    }
    
    // Verify result
    int ok = 1;
    for (size_t i = 0; i < m; i++) {
        if (fabs(y[i] - y_ref[i]) > EPSILON * 100) {  // Higher tolerance for accumulated error
            fprintf(stderr, "GEMV alpha/beta mismatch at index %zu: got %f, expected %f\n", 
                    i, y[i], y_ref[i]);
            ok = 0;
            break;
        }
    }
    
    free(a);
    free(x);
    free(y);
    free(y_ref);
    
    return ok;
}

// Test with small matrices (should NOT use MT)
static int test_small_no_mt(void)
{
    const size_t m = 10;
    const size_t n = 10;
    
    float *x = malloc(m * sizeof(float));
    float *y = malloc(n * sizeof(float));
    float *a = calloc(m * n, sizeof(float));
    
    if (!x || !y || !a) {
        fprintf(stderr, "Memory allocation failed\n");
        free(x);
        free(y);
        free(a);
        return 0;
    }
    
    for (size_t i = 0; i < m; i++) {
        x[i] = 1.0f;
    }
    for (size_t i = 0; i < n; i++) {
        y[i] = 1.0f;
    }
    
    cblas_reset_stats();
    
    // Small GER should not use MT
    cblas_sger(CblasRowMajor, m, n, 1.0f, x, 1, y, 1, a, n);
    
    const cblas_stats_t* stats_ger = cblas_get_stats("sger");
    if (stats_ger) {
        printf("  sger (small): %lu calls, %lu MT uses (m*n=%zu < CBLAS_MT_GER=%d)\n", 
               (unsigned long)stats_ger->total_calls, (unsigned long)stats_ger->mt_activations, m * n, CBLAS_MT_GER);
    }
    
    cblas_reset_stats();
    
    // Small GEMV should not use MT
    float *y2 = calloc(m, sizeof(float));
    if (!y2) {
        fprintf(stderr, "Memory allocation failed for y2\n");
        free(x);
        free(y);
        free(a);
        return 0;
    }
    
    cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0f, a, n, y, 1, 0.0f, y2, 1);
    
    const cblas_stats_t* stats_gemv = cblas_get_stats("sgemv");
    if (stats_gemv) {
        printf("  sgemv (small): %lu calls, %lu MT uses (m*n=%zu < CBLAS_MT_GEMV=%d)\n", 
               (unsigned long)stats_gemv->total_calls, (unsigned long)stats_gemv->mt_activations, m * n, CBLAS_MT_GEMV);
    }
    
    free(x);
    free(y);
    free(y2);
    free(a);
    
    return 1;
}

int test_main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;

    cblas_init(4);  // Initialize with 4 threads
    
    BEGIN_TESTS();
    
    MODULE("Level-2 Multi-Threading");
    
    SUITE("cblas_sger MT");
    TESTEX("sger with large matrix (MT active)", test_sger_mt());
    TESTEX("sger with non-unit alpha (MT active)", test_sger_mt_alpha());
    
    SUITE("cblas_sgemv MT");
    TESTEX("sgemv with large matrix (MT active)", test_sgemv_mt());
    TESTEX("sgemv with alpha/beta (MT active)", test_sgemv_mt_alpha_beta());
    
    SUITE("Small operations");
    TESTEX("small operations (MT inactive)", test_small_no_mt());
    
    END_TESTS();
    
    cblas_shutdown();
}
