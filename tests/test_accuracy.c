//------------------------------------------------------
//
// Numerical Accuracy Tests
// Tests accumulated rounding errors and edge cases
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//
//------------------------------------------------------
// ERROR BOUND THEORY:
//
// For floating-point operations, accumulated error scales with:
//   DOT product of n elements: relative error ≈ n * ε
//   GEMM (m×k × k×n): error per element ≈ k * ε
//
// Where ε is machine epsilon:
//   Single precision: FLT_EPSILON ≈ 1.19e-7
//   Double precision: DBL_EPSILON ≈ 2.22e-16
//
// We use a safety factor of 2-10x for practical tests to account
// for compiler optimizations, FMA rounding, and memory layout effects.
//------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "test.h"
#include "cblas.h"

//------------------------------------------------------
// Naive reference DOT implementations
// Use Kahan summation for improved accuracy as reference
//------------------------------------------------------
static float naive_sdot_kahan(CBLAS_INDEX n, const float* x, const float* y)
{
    float sum = 0.0f;
    float c = 0.0f;  // Compensation for lost low-order bits
    for (CBLAS_INDEX i = 0; i < n; i++) {
        float prod = x[i] * y[i] - c;
        float t = sum + prod;
        c = (t - sum) - prod;
        sum = t;
    }
    return sum;
}

static double naive_ddot_kahan(CBLAS_INDEX n, const double* x, const double* y)
{
    double sum = 0.0;
    double c = 0.0;
    for (CBLAS_INDEX i = 0; i < n; i++) {
        double prod = x[i] * y[i] - c;
        double t = sum + prod;
        c = (t - sum) - prod;
        sum = t;
    }
    return sum;
}

// Simple naive DOT (without Kahan) for comparison
static float naive_sdot(CBLAS_INDEX n, const float* x, const float* y)
{
    float sum = 0.0f;
    for (CBLAS_INDEX i = 0; i < n; i++) {
        sum += x[i] * y[i];
    }
    return sum;
}

CBLAS_UNUSED static double naive_ddot(CBLAS_INDEX n, const double* x, const double* y)
{
    double sum = 0.0;
    for (CBLAS_INDEX i = 0; i < n; i++) {
        sum += x[i] * y[i];
    }
    return sum;
}

//------------------------------------------------------
// Naive reference GEMM implementation
//------------------------------------------------------
static void naive_sgemm(CBLAS_INDEX m, CBLAS_INDEX n, CBLAS_INDEX k,
                        float alpha, const float* A, CBLAS_INDEX lda,
                        const float* B, CBLAS_INDEX ldb,
                        float beta, float* C, CBLAS_INDEX ldc)
{
    for (CBLAS_INDEX i = 0; i < m; i++) {
        for (CBLAS_INDEX j = 0; j < n; j++) {
            C[i * ldc + j] *= beta;
        }
    }
    for (CBLAS_INDEX i = 0; i < m; i++) {
        for (CBLAS_INDEX j = 0; j < n; j++) {
            float sum = 0.0f;
            for (CBLAS_INDEX p = 0; p < k; p++) {
                sum += A[i * lda + p] * B[p * ldb + j];
            }
            C[i * ldc + j] += alpha * sum;
        }
    }
}

static void naive_dgemm(CBLAS_INDEX m, CBLAS_INDEX n, CBLAS_INDEX k,
                        double alpha, const double* A, CBLAS_INDEX lda,
                        const double* B, CBLAS_INDEX ldb,
                        double beta, double* C, CBLAS_INDEX ldc)
{
    for (CBLAS_INDEX i = 0; i < m; i++) {
        for (CBLAS_INDEX j = 0; j < n; j++) {
            C[i * ldc + j] *= beta;
        }
    }
    for (CBLAS_INDEX i = 0; i < m; i++) {
        for (CBLAS_INDEX j = 0; j < n; j++) {
            double sum = 0.0;
            for (CBLAS_INDEX p = 0; p < k; p++) {
                sum += A[i * lda + p] * B[p * ldb + j];
            }
            C[i * ldc + j] += alpha * sum;
        }
    }
}

//------------------------------------------------------
// Random data generation with seed
//------------------------------------------------------
static void fill_random_svector(float* v, CBLAS_INDEX n, unsigned int seed)
{
    srand(seed);
    for (CBLAS_INDEX i = 0; i < n; i++) {
        v[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }
}

static void fill_random_dvector(double* v, CBLAS_INDEX n, unsigned int seed)
{
    srand(seed);
    for (CBLAS_INDEX i = 0; i < n; i++) {
        v[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }
}

//------------------------------------------------------
// Error calculation utilities
//------------------------------------------------------
static float compute_relative_error_s(float computed, float reference)
{
    if (fabsf(reference) < FLT_MIN) {
        return fabsf(computed - reference);
    }
    return fabsf(computed - reference) / fabsf(reference);
}

static double compute_relative_error_d(double computed, double reference)
{
    if (fabs(reference) < DBL_MIN) {
        return fabs(computed - reference);
    }
    return fabs(computed - reference) / fabs(reference);
}

static float compute_max_error_sarray(float* computed, float* reference, CBLAS_INDEX n)
{
    float max_err = 0.0f;
    for (CBLAS_INDEX i = 0; i < n; i++) {
        float err = fabsf(computed[i] - reference[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

static double compute_max_error_darray(double* computed, double* reference, CBLAS_INDEX n)
{
    double max_err = 0.0;
    for (CBLAS_INDEX i = 0; i < n; i++) {
        double err = fabs(computed[i] - reference[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

//------------------------------------------------------
// LARGE DOT TESTS
// Test accumulated rounding errors in large vector operations
// Expected error: relative_error ≤ safety_factor * n * epsilon
//------------------------------------------------------
static int run_sdot_accuracy_test(CBLAS_INDEX n, unsigned int seed)
{
    float* x = malloc(n * sizeof(float));
    float* y = malloc(n * sizeof(float));
    if (!x || !y) { free(x); free(y); return 0; }

    fill_random_svector(x, n, seed);
    fill_random_svector(y, n, seed + 1);

    float result_cblas = cblas_sdot(n, x, 1, y, 1);
    float result_kahan = naive_sdot_kahan(n, x, y);
    float result_naive = naive_sdot(n, x, y);

    // Error bound: n * epsilon with safety factor
    float expected_error = (float)n * FLT_EPSILON * 10.0f;
    float rel_err_vs_kahan = compute_relative_error_s(result_cblas, result_kahan);
    float rel_err_vs_naive = compute_relative_error_s(result_cblas, result_naive);

    int pass = (rel_err_vs_kahan < expected_error) || (rel_err_vs_naive < expected_error);

    if (!pass) {
        printf("\n    [DEBUG] sdot n=%d: cblas=%.6e, kahan=%.6e, naive=%.6e\n",
               (int)n, result_cblas, result_kahan, result_naive);
        printf("    rel_err_kahan=%.2e, rel_err_naive=%.2e, bound=%.2e\n",
               rel_err_vs_kahan, rel_err_vs_naive, expected_error);
    }

    free(x); free(y);
    return pass;
}

static int run_ddot_accuracy_test(CBLAS_INDEX n, unsigned int seed)
{
    double* x = malloc(n * sizeof(double));
    double* y = malloc(n * sizeof(double));
    if (!x || !y) { free(x); free(y); return 0; }

    fill_random_dvector(x, n, seed);
    fill_random_dvector(y, n, seed + 1);

    double result_cblas = cblas_ddot(n, x, 1, y, 1);
    double result_kahan = naive_ddot_kahan(n, x, y);

    // Error bound: n * epsilon with safety factor
    double expected_error = (double)n * DBL_EPSILON * 10.0;
    double rel_err = compute_relative_error_d(result_cblas, result_kahan);

    int pass = (rel_err < expected_error);

    if (!pass) {
        printf("\n    [DEBUG] ddot n=%d: cblas=%.10e, kahan=%.10e, rel_err=%.2e, bound=%.2e\n",
               (int)n, result_cblas, result_kahan, rel_err, expected_error);
    }

    free(x); free(y);
    return pass;
}

static void test_large_dot(void)
{
    SUITE("Large DOT accuracy (accumulated rounding)");

    COMMENT("sdot n=10,000");
    TEST(run_sdot_accuracy_test(10000, 1000));

    COMMENT("sdot n=100,000");
    TEST(run_sdot_accuracy_test(100000, 2000));

    COMMENT("sdot n=1,000,000");
    TEST(run_sdot_accuracy_test(1000000, 3000));

    COMMENT("ddot n=10,000");
    TEST(run_ddot_accuracy_test(10000, 4000));

    COMMENT("ddot n=100,000");
    TEST(run_ddot_accuracy_test(100000, 5000));

    COMMENT("ddot n=1,000,000");
    TEST(run_ddot_accuracy_test(1000000, 6000));
}

//------------------------------------------------------
// LARGE GEMM TESTS
// Test larger matrix sizes than existing test_gemm_accuracy.c
// Expected error per element: k * epsilon
//------------------------------------------------------
static int run_sgemm_accuracy_test(CBLAS_INDEX m, CBLAS_INDEX n, CBLAS_INDEX k,
                                    float alpha, float beta, unsigned int seed)
{
    float* A = malloc(m * k * sizeof(float));
    float* B = malloc(k * n * sizeof(float));
    float* C_ref = malloc(m * n * sizeof(float));
    float* C_opt = malloc(m * n * sizeof(float));

    if (!A || !B || !C_ref || !C_opt) {
        free(A); free(B); free(C_ref); free(C_opt);
        return 0;
    }

    fill_random_svector(A, m * k, seed);
    fill_random_svector(B, k * n, seed + 1);
    fill_random_svector(C_ref, m * n, seed + 2);
    memcpy(C_opt, C_ref, m * n * sizeof(float));

    naive_sgemm(m, n, k, alpha, A, k, B, n, beta, C_ref, n);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, k, B, n, beta, C_opt, n);

    // Error bound: k * epsilon with safety factor
    float abs_eps = (float)k * FLT_EPSILON * 100.0f;
    float max_err = compute_max_error_sarray(C_opt, C_ref, m * n);
    int result = (max_err < abs_eps);

    if (!result) {
        printf("\n    [DEBUG] sgemm %dx%dx%d: max_err=%.2e, bound=%.2e\n",
               (int)m, (int)n, (int)k, max_err, abs_eps);
    }

    free(A); free(B); free(C_ref); free(C_opt);
    return result;
}

static int run_dgemm_accuracy_test(CBLAS_INDEX m, CBLAS_INDEX n, CBLAS_INDEX k,
                                    double alpha, double beta, unsigned int seed)
{
    double* A = malloc(m * k * sizeof(double));
    double* B = malloc(k * n * sizeof(double));
    double* C_ref = malloc(m * n * sizeof(double));
    double* C_opt = malloc(m * n * sizeof(double));

    if (!A || !B || !C_ref || !C_opt) {
        free(A); free(B); free(C_ref); free(C_opt);
        return 0;
    }

    fill_random_dvector(A, m * k, seed);
    fill_random_dvector(B, k * n, seed + 1);
    fill_random_dvector(C_ref, m * n, seed + 2);
    memcpy(C_opt, C_ref, m * n * sizeof(double));

    naive_dgemm(m, n, k, alpha, A, k, B, n, beta, C_ref, n);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, k, B, n, beta, C_opt, n);

    // Error bound: k * epsilon with safety factor
    double abs_eps = (double)k * DBL_EPSILON * 100.0;
    double max_err = compute_max_error_darray(C_opt, C_ref, m * n);
    int result = (max_err < abs_eps);

    if (!result) {
        printf("\n    [DEBUG] dgemm %dx%dx%d: max_err=%.2e, bound=%.2e\n",
               (int)m, (int)n, (int)k, max_err, abs_eps);
    }

    free(A); free(B); free(C_ref); free(C_opt);
    return result;
}

static void test_large_gemm(void)
{
    SUITE("Large GEMM accuracy (n > 256)");

    COMMENT("sgemm 512x512");
    TEST(run_sgemm_accuracy_test(512, 512, 512, 1.0f, 0.0f, 7000));

    COMMENT("sgemm 1024x1024");
    TEST(run_sgemm_accuracy_test(1024, 1024, 1024, 1.0f, 0.0f, 7001));

    COMMENT("dgemm 512x512");
    TEST(run_dgemm_accuracy_test(512, 512, 512, 1.0, 0.0, 8000));

    COMMENT("dgemm 1024x1024");
    TEST(run_dgemm_accuracy_test(1024, 1024, 1024, 1.0, 0.0, 8001));

    // Non-square large
    COMMENT("sgemm 1000x500 x 500x800");
    TEST(run_sgemm_accuracy_test(1000, 800, 500, 1.0f, 0.0f, 7002));

    COMMENT("dgemm 800x1200 x 1200x600");
    TEST(run_dgemm_accuracy_test(800, 600, 1200, 1.0, 0.0, 8002));
}

//------------------------------------------------------
// EDGE CASE TESTS
// Test behavior with denormals, infinities, and NaN
//------------------------------------------------------
static void test_edge_cases(void)
{
    SUITE("Edge cases (special floating-point values)");

    // Test with zeros
    {
        float x[] = {0.0f, 0.0f, 0.0f, 0.0f};
        float y[] = {1.0f, 2.0f, 3.0f, 4.0f};
        float result = cblas_sdot(4, x, 1, y, 1);
        COMMENT("sdot with all zeros in x");
        TEST(result == 0.0f);
    }

    // Test with very small (near denormal) values - single precision
    {
        float x[] = {1e-38f, 1e-38f, 1e-38f, 1e-38f};
        float y[] = {1.0f, 1.0f, 1.0f, 1.0f};
        float result = cblas_sdot(4, x, 1, y, 1);
        float expected = 4e-38f;
        COMMENT("sdot with near-denormal values");
        TEST(fabsf(result - expected) < 1e-44f);
    }

    // Test with large magnitude values - single precision
    {
        float x[] = {1e30f, 1e30f, 1e30f, 1e30f};
        float y[] = {1e-30f, 1e-30f, 1e-30f, 1e-30f};
        float result = cblas_sdot(4, x, 1, y, 1);
        COMMENT("sdot with large/small magnitude cancellation");
        TEST(fabsf(result - 4.0f) < 1e-5f);
    }

    // Test infinity handling
    {
        float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
        float y[] = {INFINITY, 1.0f, 1.0f, 1.0f};
        float result = cblas_sdot(4, x, 1, y, 1);
        COMMENT("sdot with infinity produces infinity");
        TEST(isinf(result) && result > 0);
    }

    // Test NaN propagation
    {
        float x[] = {1.0f, NAN, 3.0f, 4.0f};
        float y[] = {1.0f, 1.0f, 1.0f, 1.0f};
        float result = cblas_sdot(4, x, 1, y, 1);
        COMMENT("sdot with NaN propagates NaN");
        TEST(isnan(result));
    }

    // Test negative infinity
    {
        float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
        float y[] = {-INFINITY, 1.0f, 1.0f, 1.0f};
        float result = cblas_sdot(4, x, 1, y, 1);
        COMMENT("sdot with -infinity produces -infinity");
        TEST(isinf(result) && result < 0);
    }

    // Double precision edge cases
    {
        double x[] = {1e-300, 1e-300, 1e-300, 1e-300};
        double y[] = {1.0, 1.0, 1.0, 1.0};
        double result = cblas_ddot(4, x, 1, y, 1);
        double expected = 4e-300;
        COMMENT("ddot with very small values");
        TEST(fabs(result - expected) < 1e-310);
    }

    {
        double x[] = {1.0, NAN, 3.0, 4.0};
        double y[] = {1.0, 1.0, 1.0, 1.0};
        double result = cblas_ddot(4, x, 1, y, 1);
        COMMENT("ddot with NaN propagates NaN");
        TEST(isnan(result));
    }
}

//------------------------------------------------------
// STRESS TESTS
// Test consistency across many random inputs
//------------------------------------------------------
static void test_consistency(void)
{
    SUITE("Consistency stress tests");

    // Run many small random DOT tests
    COMMENT("100 random sdot tests (n=1000)");
    int pass_count = 0;
    for (int i = 0; i < 100; i++) {
        if (run_sdot_accuracy_test(1000, 10000 + i)) {
            pass_count++;
        }
    }
    TEST(pass_count == 100);

    COMMENT("100 random ddot tests (n=1000)");
    pass_count = 0;
    for (int i = 0; i < 100; i++) {
        if (run_ddot_accuracy_test(1000, 20000 + i)) {
            pass_count++;
        }
    }
    TEST(pass_count == 100);

    // Run many small random GEMM tests
    COMMENT("20 random sgemm tests (64x64)");
    pass_count = 0;
    for (int i = 0; i < 20; i++) {
        if (run_sgemm_accuracy_test(64, 64, 64, 1.0f, 0.0f, 30000 + i)) {
            pass_count++;
        }
    }
    TEST(pass_count == 20);

    COMMENT("20 random dgemm tests (64x64)");
    pass_count = 0;
    for (int i = 0; i < 20; i++) {
        if (run_dgemm_accuracy_test(64, 64, 64, 1.0, 0.0, 40000 + i)) {
            pass_count++;
        }
    }
    TEST(pass_count == 20);
}

//------------------------------------------------------
// Main test runner
//------------------------------------------------------
int test_main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;

    cblas_init(CBLAS_DEFAULT_THREADS);
    cblas_print_configuration();

    MODULE("Numerical Accuracy Tests");

    test_large_dot();
    test_large_gemm();
    test_edge_cases();
    test_consistency();

    cblas_shutdown();

    return 0;
}
