//------------------------------------------------------
//
// GEMM Accuracy Tests - Extended parameterized tests
// comparing cblas_sgemm/dgemm against naive reference
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//
// NOTE: sgemm tests currently detect known accuracy issues.
// dgemm tests pass. sgemm issues are documented for future fixes.
//------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "test.h"
#include "cblas.h"

// Set to 1 to enable sgemm tests
// Set to 0 to only run dgemm tests
#define TEST_SGEMM_ACCURACY 1

//------------------------------------------------------
// Naive reference GEMM implementations
// C = alpha * A * B + beta * C (row-major, no transpose)
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
// Random matrix generation
//------------------------------------------------------
static void fill_random_smatrix(float* M, CBLAS_INDEX rows, CBLAS_INDEX cols, unsigned int seed)
{
	srand(seed);
	for (CBLAS_INDEX i = 0; i < rows * cols; i++) {
		M[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
	}
}

static void fill_random_dmatrix(double* M, CBLAS_INDEX rows, CBLAS_INDEX cols, unsigned int seed)
{
	srand(seed);
	for (CBLAS_INDEX i = 0; i < rows * cols; i++) {
		M[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
	}
}

//------------------------------------------------------
// Array comparison with relaxed epsilon for large matrices
//------------------------------------------------------
CBLAS_UNUSED static int equal_sarray_relaxed(float* a, float* b, CBLAS_INDEX len, float eps)
{
	for (CBLAS_INDEX i = 0; i < len; i++) {
		if (fabsf(a[i] - b[i]) > eps)
			return 0;
	}
	return 1;
}

static int equal_darray_relaxed(double* a, double* b, CBLAS_INDEX len, double eps)
{
	for (CBLAS_INDEX i = 0; i < len; i++) {
		if (fabs(a[i] - b[i]) > eps)
			return 0;
	}
	return 1;
}

//------------------------------------------------------
// Test helper: run single sgemm accuracy test
//------------------------------------------------------
CBLAS_UNUSED static int run_sgemm_accuracy_test(CBLAS_INDEX m, CBLAS_INDEX n, CBLAS_INDEX k, 
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

	fill_random_smatrix(A, m, k, seed);
	fill_random_smatrix(B, k, n, seed + 1);
	fill_random_smatrix(C_ref, m, n, seed + 2);
	memcpy(C_opt, C_ref, m * n * sizeof(float));

	naive_sgemm(m, n, k, alpha, A, k, B, n, beta, C_ref, n);
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, k, B, n, beta, C_opt, n);

	// Find max absolute difference and relative error
	float max_diff = 0.0f;
	float max_rel_err = 0.0f;
	for (CBLAS_INDEX i = 0; i < m * n; i++) {
		float diff = fabsf(C_ref[i] - C_opt[i]);
		if (diff > max_diff) max_diff = diff;
		if (fabsf(C_ref[i]) > 1e-10f) {
			float rel = diff / fabsf(C_ref[i]);
			if (rel > max_rel_err) max_rel_err = rel;
		}
	}

	// Use absolute error tolerance that scales with k (number of FMA operations)
	// Single precision: ~7 digits, so k * 1e-5 is appropriate
	float abs_eps = k * 1e-5f;
	int result = (max_diff < abs_eps);

	if (!result) {
		printf("\n    [DEBUG] sgemm %dx%dx%d: max_diff=%.2e, max_rel_err=%.2e (eps=%.2e)\n",
			(int)m, (int)n, (int)k, max_diff, max_rel_err, abs_eps);
	}

	free(A); free(B); free(C_ref); free(C_opt);
	return result;
}

//------------------------------------------------------
// Test helper: run single dgemm accuracy test
//------------------------------------------------------
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

	fill_random_dmatrix(A, m, k, seed);
	fill_random_dmatrix(B, k, n, seed + 1);
	fill_random_dmatrix(C_ref, m, n, seed + 2);
	memcpy(C_opt, C_ref, m * n * sizeof(double));

	naive_dgemm(m, n, k, alpha, A, k, B, n, beta, C_ref, n);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, k, B, n, beta, C_opt, n);

	// Relaxed epsilon scales with matrix size
	double eps = k * 1e-10;
	int result = equal_darray_relaxed(C_ref, C_opt, m * n, eps);

	free(A); free(B); free(C_ref); free(C_opt);
	return result;
}

//------------------------------------------------------
// Transpose-aware naive reference GEMM
// C = alpha * op(A) * op(B) + beta * C (row-major)
//------------------------------------------------------
static void naive_sgemm_trans(CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb,
                              CBLAS_INDEX m, CBLAS_INDEX n, CBLAS_INDEX k,
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
				float a_val = (transa == CblasNoTrans) ? A[i * lda + p] : A[p * lda + i];
				float b_val = (transb == CblasNoTrans) ? B[p * ldb + j] : B[j * ldb + p];
				sum += a_val * b_val;
			}
			C[i * ldc + j] += alpha * sum;
		}
	}
}

static void naive_dgemm_trans(CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb,
                              CBLAS_INDEX m, CBLAS_INDEX n, CBLAS_INDEX k,
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
				double a_val = (transa == CblasNoTrans) ? A[i * lda + p] : A[p * lda + i];
				double b_val = (transb == CblasNoTrans) ? B[p * ldb + j] : B[j * ldb + p];
				sum += a_val * b_val;
			}
			C[i * ldc + j] += alpha * sum;
		}
	}
}

//------------------------------------------------------
// Test helper: run sgemm accuracy test with transpose
// A is stored as rows_a × cols_a, B as rows_b × cols_b
//------------------------------------------------------
static int run_sgemm_trans_test(CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb,
                                CBLAS_INDEX m, CBLAS_INDEX n, CBLAS_INDEX k,
                                float alpha, float beta, unsigned int seed)
{
	// Storage dimensions depend on transpose flags
	// NoTrans: A is m×k, Trans: A is k×m (stored)
	CBLAS_INDEX rows_a = (transa == CblasNoTrans) ? m : k;
	CBLAS_INDEX cols_a = (transa == CblasNoTrans) ? k : m;
	CBLAS_INDEX rows_b = (transb == CblasNoTrans) ? k : n;
	CBLAS_INDEX cols_b = (transb == CblasNoTrans) ? n : k;
	CBLAS_INDEX lda = cols_a;
	CBLAS_INDEX ldb = cols_b;

	float* A = malloc(rows_a * cols_a * sizeof(float));
	float* B = malloc(rows_b * cols_b * sizeof(float));
	float* C_ref = malloc(m * n * sizeof(float));
	float* C_opt = malloc(m * n * sizeof(float));

	if (!A || !B || !C_ref || !C_opt) {
		free(A); free(B); free(C_ref); free(C_opt);
		return 0;
	}

	fill_random_smatrix(A, rows_a, cols_a, seed);
	fill_random_smatrix(B, rows_b, cols_b, seed + 1);
	fill_random_smatrix(C_ref, m, n, seed + 2);
	memcpy(C_opt, C_ref, m * n * sizeof(float));

	naive_sgemm_trans(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C_ref, n);
	cblas_sgemm(CblasRowMajor, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C_opt, n);

	float max_diff = 0.0f;
	float max_rel_err = 0.0f;
	for (CBLAS_INDEX i = 0; i < m * n; i++) {
		float diff = fabsf(C_ref[i] - C_opt[i]);
		if (diff > max_diff) max_diff = diff;
		if (fabsf(C_ref[i]) > 1e-10f) {
			float rel = diff / fabsf(C_ref[i]);
			if (rel > max_rel_err) max_rel_err = rel;
		}
	}

	float abs_eps = k * 1e-5f;
	int result = (max_diff < abs_eps);

	if (!result) {
		const char* ta = (transa == CblasTrans) ? "T" : "N";
		const char* tb = (transb == CblasTrans) ? "T" : "N";
		printf("\n    [DEBUG] sgemm %s%s %dx%dx%d: max_diff=%.2e, max_rel_err=%.2e (eps=%.2e)\n",
			ta, tb, (int)m, (int)n, (int)k, max_diff, max_rel_err, abs_eps);
	}

	free(A); free(B); free(C_ref); free(C_opt);
	return result;
}

//------------------------------------------------------
// Test helper: run dgemm accuracy test with transpose
//------------------------------------------------------
static int run_dgemm_trans_test(CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb,
                                CBLAS_INDEX m, CBLAS_INDEX n, CBLAS_INDEX k,
                                double alpha, double beta, unsigned int seed)
{
	CBLAS_INDEX rows_a = (transa == CblasNoTrans) ? m : k;
	CBLAS_INDEX cols_a = (transa == CblasNoTrans) ? k : m;
	CBLAS_INDEX rows_b = (transb == CblasNoTrans) ? k : n;
	CBLAS_INDEX cols_b = (transb == CblasNoTrans) ? n : k;
	CBLAS_INDEX lda = cols_a;
	CBLAS_INDEX ldb = cols_b;

	double* A = malloc(rows_a * cols_a * sizeof(double));
	double* B = malloc(rows_b * cols_b * sizeof(double));
	double* C_ref = malloc(m * n * sizeof(double));
	double* C_opt = malloc(m * n * sizeof(double));

	if (!A || !B || !C_ref || !C_opt) {
		free(A); free(B); free(C_ref); free(C_opt);
		return 0;
	}

	fill_random_dmatrix(A, rows_a, cols_a, seed);
	fill_random_dmatrix(B, rows_b, cols_b, seed + 1);
	fill_random_dmatrix(C_ref, m, n, seed + 2);
	memcpy(C_opt, C_ref, m * n * sizeof(double));

	naive_dgemm_trans(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C_ref, n);
	cblas_dgemm(CblasRowMajor, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C_opt, n);

	double eps = k * 1e-10;
	int result = equal_darray_relaxed(C_ref, C_opt, m * n, eps);

	if (!result) {
		const char* ta = (transa == CblasTrans) ? "T" : "N";
		const char* tb = (transb == CblasTrans) ? "T" : "N";
		printf("\n    [DEBUG] dgemm %s%s %dx%dx%d FAILED\n", ta, tb, (int)m, (int)n, (int)k);
	}

	free(A); free(B); free(C_ref); free(C_opt);
	return result;
}
static void test_small_matrices(void)
{
	SUITE("Small matrix accuracy (below blocking)");

#if TEST_SGEMM_ACCURACY
	COMMENT("sgemm 3x3");
	TEST(run_sgemm_accuracy_test(3, 3, 3, 1.0f, 0.0f, 1000));

	COMMENT("sgemm 7x7");
	TEST(run_sgemm_accuracy_test(7, 7, 7, 1.0f, 0.0f, 1001));

	COMMENT("sgemm 15x15");
	TEST(run_sgemm_accuracy_test(15, 15, 15, 1.0f, 0.0f, 1002));

	COMMENT("sgemm 31x31");
	TEST(run_sgemm_accuracy_test(31, 31, 31, 1.0f, 0.0f, 1003));
#endif

	COMMENT("dgemm 3x3");
	TEST(run_dgemm_accuracy_test(3, 3, 3, 1.0, 0.0, 2000));

	COMMENT("dgemm 7x7");
	TEST(run_dgemm_accuracy_test(7, 7, 7, 1.0, 0.0, 2001));

	COMMENT("dgemm 15x15");
	TEST(run_dgemm_accuracy_test(15, 15, 15, 1.0, 0.0, 2002));

	COMMENT("dgemm 31x31");
	TEST(run_dgemm_accuracy_test(31, 31, 31, 1.0, 0.0, 2003));
}

//------------------------------------------------------
// Tile boundary tests (around mc=64, 128, 256)
//------------------------------------------------------
static void test_tile_boundaries(void)
{
	SUITE("Tile boundary accuracy (mc/kc edges)");

#if TEST_SGEMM_ACCURACY
	COMMENT("sgemm 63x63");
	TEST(run_sgemm_accuracy_test(63, 63, 63, 1.0f, 0.0f, 3000));

	COMMENT("sgemm 64x64");
	TEST(run_sgemm_accuracy_test(64, 64, 64, 1.0f, 0.0f, 3001));

	COMMENT("sgemm 65x65");
	TEST(run_sgemm_accuracy_test(65, 65, 65, 1.0f, 0.0f, 3002));

	COMMENT("sgemm 127x127");
	TEST(run_sgemm_accuracy_test(127, 127, 127, 1.0f, 0.0f, 3003));

	COMMENT("sgemm 128x128");
	TEST(run_sgemm_accuracy_test(128, 128, 128, 1.0f, 0.0f, 3004));

	COMMENT("sgemm 129x129");
	TEST(run_sgemm_accuracy_test(129, 129, 129, 1.0f, 0.0f, 3005));
#endif

	COMMENT("dgemm 63x63");
	TEST(run_dgemm_accuracy_test(63, 63, 63, 1.0, 0.0, 4000));

	COMMENT("dgemm 64x64");
	TEST(run_dgemm_accuracy_test(64, 64, 64, 1.0, 0.0, 4001));

	COMMENT("dgemm 65x65");
	TEST(run_dgemm_accuracy_test(65, 65, 65, 1.0, 0.0, 4002));

	COMMENT("dgemm 127x127");
	TEST(run_dgemm_accuracy_test(127, 127, 127, 1.0, 0.0, 4003));

	COMMENT("dgemm 128x128");
	TEST(run_dgemm_accuracy_test(128, 128, 128, 1.0, 0.0, 4004));

	COMMENT("dgemm 129x129");
	TEST(run_dgemm_accuracy_test(129, 129, 129, 1.0, 0.0, 4005));
}

//------------------------------------------------------
// Large block tests (around kc=256, nb=512)
//------------------------------------------------------
static void test_large_blocks(void)
{
	SUITE("Large block accuracy (kc/nb edges)");

#if TEST_SGEMM_ACCURACY
	COMMENT("sgemm 255x255");
	TEST(run_sgemm_accuracy_test(255, 255, 255, 1.0f, 0.0f, 5000));

	COMMENT("sgemm 256x256");
	TEST(run_sgemm_accuracy_test(256, 256, 256, 1.0f, 0.0f, 5001));

	COMMENT("sgemm 257x257");
	TEST(run_sgemm_accuracy_test(257, 257, 257, 1.0f, 0.0f, 5002));
#endif

	COMMENT("dgemm 255x255");
	TEST(run_dgemm_accuracy_test(255, 255, 255, 1.0, 0.0, 6000));

	COMMENT("dgemm 256x256");
	TEST(run_dgemm_accuracy_test(256, 256, 256, 1.0, 0.0, 6001));

	COMMENT("dgemm 257x257");
	TEST(run_dgemm_accuracy_test(257, 257, 257, 1.0, 0.0, 6002));
}

//------------------------------------------------------
// Non-square matrix tests
//------------------------------------------------------
static void test_nonsquare_matrices(void)
{
	SUITE("Non-square matrix accuracy");

#if TEST_SGEMM_ACCURACY
	// Tall-skinny: M >> N
	COMMENT("sgemm tall-skinny (128x16) * (16x32)");
	TESTEX("tall-skinny", run_sgemm_accuracy_test(128, 32, 16, 1.0f, 0.0f, 7001));

	// Short-wide: M << N
	COMMENT("sgemm short-wide (16x128) * (128x256)");
	TESTEX("short-wide", run_sgemm_accuracy_test(16, 256, 128, 1.0f, 0.0f, 7002));

	// Mixed dimensions around boundaries
	COMMENT("sgemm mixed (65x127) * (127x63)");
	TESTEX("mixed dims", run_sgemm_accuracy_test(65, 63, 127, 1.0f, 0.0f, 7004));
#endif

	// Panel multiply: thin K
	COMMENT("dgemm panel (64x64) * (64x8)");
	TESTEX("panel", run_dgemm_accuracy_test(64, 8, 64, 1.0, 0.0, 7003));

	COMMENT("dgemm mixed (129x255) * (255x131)");
	TESTEX("mixed dims large", run_dgemm_accuracy_test(129, 131, 255, 1.0, 0.0, 7005));
}

//------------------------------------------------------
// Alpha/beta scaling tests
//------------------------------------------------------
static void test_scaling(void)
{
	SUITE("Alpha/beta scaling accuracy");

#if TEST_SGEMM_ACCURACY
	COMMENT("sgemm alpha=2.0, beta=0.0");
	TESTEX("alpha=2", run_sgemm_accuracy_test(64, 64, 64, 2.0f, 0.0f, 8001));

	COMMENT("sgemm alpha=0.5, beta=0.5");
	TESTEX("alpha=0.5 beta=0.5", run_sgemm_accuracy_test(64, 64, 64, 0.5f, 0.5f, 8002));

	COMMENT("sgemm alpha=0.0, beta=2.0");
	TESTEX("alpha=0", run_sgemm_accuracy_test(32, 32, 32, 0.0f, 2.0f, 8005));
#endif

	COMMENT("dgemm alpha=-1.0, beta=1.0");
	TESTEX("negative alpha", run_dgemm_accuracy_test(64, 64, 64, -1.0, 1.0, 8003));

	COMMENT("dgemm alpha=1.0, beta=-0.5");
	TESTEX("negative beta", run_dgemm_accuracy_test(64, 64, 64, 1.0, -0.5, 8004));
}

//------------------------------------------------------
// Transpose tests: TransA only (TN)
//------------------------------------------------------
static void test_transpose_TN(void)
{
	SUITE("Transpose TN accuracy (A^T * B)");

#if TEST_SGEMM_ACCURACY
	COMMENT("sgemm TN 3x3x3");
	TEST(run_sgemm_trans_test(CblasTrans, CblasNoTrans, 3, 3, 3, 1.0f, 0.0f, 9000));

	COMMENT("sgemm TN 7x7x7");
	TEST(run_sgemm_trans_test(CblasTrans, CblasNoTrans, 7, 7, 7, 1.0f, 0.0f, 9001));

	COMMENT("sgemm TN 31x31x31");
	TEST(run_sgemm_trans_test(CblasTrans, CblasNoTrans, 31, 31, 31, 1.0f, 0.0f, 9002));

	COMMENT("sgemm TN 64x64x64");
	TEST(run_sgemm_trans_test(CblasTrans, CblasNoTrans, 64, 64, 64, 1.0f, 0.0f, 9003));

	COMMENT("sgemm TN 127x127x127");
	TEST(run_sgemm_trans_test(CblasTrans, CblasNoTrans, 127, 127, 127, 1.0f, 0.0f, 9004));

	COMMENT("sgemm TN 128x128x128");
	TEST(run_sgemm_trans_test(CblasTrans, CblasNoTrans, 128, 128, 128, 1.0f, 0.0f, 9005));

	COMMENT("sgemm TN 257x257x257");
	TEST(run_sgemm_trans_test(CblasTrans, CblasNoTrans, 257, 257, 257, 1.0f, 0.0f, 9006));

	COMMENT("sgemm TN non-square 128x32x64");
	TESTEX("TN non-square", run_sgemm_trans_test(CblasTrans, CblasNoTrans, 128, 32, 64, 1.0f, 0.0f, 9007));

	COMMENT("sgemm TN alpha=2.0 beta=0.5");
	TESTEX("TN scaling", run_sgemm_trans_test(CblasTrans, CblasNoTrans, 64, 64, 64, 2.0f, 0.5f, 9008));
#endif

	COMMENT("dgemm TN 64x64x64");
	TEST(run_dgemm_trans_test(CblasTrans, CblasNoTrans, 64, 64, 64, 1.0, 0.0, 9100));

	COMMENT("dgemm TN 127x127x127");
	TEST(run_dgemm_trans_test(CblasTrans, CblasNoTrans, 127, 127, 127, 1.0, 0.0, 9101));

	COMMENT("dgemm TN 257x257x257");
	TEST(run_dgemm_trans_test(CblasTrans, CblasNoTrans, 257, 257, 257, 1.0, 0.0, 9102));
}

//------------------------------------------------------
// Transpose tests: TransB only (NT)
//------------------------------------------------------
static void test_transpose_NT(void)
{
	SUITE("Transpose NT accuracy (A * B^T)");

#if TEST_SGEMM_ACCURACY
	COMMENT("sgemm NT 3x3x3");
	TEST(run_sgemm_trans_test(CblasNoTrans, CblasTrans, 3, 3, 3, 1.0f, 0.0f, 9200));

	COMMENT("sgemm NT 7x7x7");
	TEST(run_sgemm_trans_test(CblasNoTrans, CblasTrans, 7, 7, 7, 1.0f, 0.0f, 9201));

	COMMENT("sgemm NT 31x31x31");
	TEST(run_sgemm_trans_test(CblasNoTrans, CblasTrans, 31, 31, 31, 1.0f, 0.0f, 9202));

	COMMENT("sgemm NT 64x64x64");
	TEST(run_sgemm_trans_test(CblasNoTrans, CblasTrans, 64, 64, 64, 1.0f, 0.0f, 9203));

	COMMENT("sgemm NT 127x127x127");
	TEST(run_sgemm_trans_test(CblasNoTrans, CblasTrans, 127, 127, 127, 1.0f, 0.0f, 9204));

	COMMENT("sgemm NT 128x128x128");
	TEST(run_sgemm_trans_test(CblasNoTrans, CblasTrans, 128, 128, 128, 1.0f, 0.0f, 9205));

	COMMENT("sgemm NT 257x257x257");
	TEST(run_sgemm_trans_test(CblasNoTrans, CblasTrans, 257, 257, 257, 1.0f, 0.0f, 9206));

	COMMENT("sgemm NT non-square 16x256x128");
	TESTEX("NT non-square", run_sgemm_trans_test(CblasNoTrans, CblasTrans, 16, 256, 128, 1.0f, 0.0f, 9207));

	COMMENT("sgemm NT alpha=0.5 beta=1.0");
	TESTEX("NT scaling", run_sgemm_trans_test(CblasNoTrans, CblasTrans, 64, 64, 64, 0.5f, 1.0f, 9208));
#endif

	COMMENT("dgemm NT 64x64x64");
	TEST(run_dgemm_trans_test(CblasNoTrans, CblasTrans, 64, 64, 64, 1.0, 0.0, 9300));

	COMMENT("dgemm NT 127x127x127");
	TEST(run_dgemm_trans_test(CblasNoTrans, CblasTrans, 127, 127, 127, 1.0, 0.0, 9301));

	COMMENT("dgemm NT 257x257x257");
	TEST(run_dgemm_trans_test(CblasNoTrans, CblasTrans, 257, 257, 257, 1.0, 0.0, 9302));
}

//------------------------------------------------------
// Transpose tests: Both transposed (TT)
//------------------------------------------------------
static void test_transpose_TT(void)
{
	SUITE("Transpose TT accuracy (A^T * B^T)");

#if TEST_SGEMM_ACCURACY
	COMMENT("sgemm TT 3x3x3");
	TEST(run_sgemm_trans_test(CblasTrans, CblasTrans, 3, 3, 3, 1.0f, 0.0f, 9400));

	COMMENT("sgemm TT 7x7x7");
	TEST(run_sgemm_trans_test(CblasTrans, CblasTrans, 7, 7, 7, 1.0f, 0.0f, 9401));

	COMMENT("sgemm TT 31x31x31");
	TEST(run_sgemm_trans_test(CblasTrans, CblasTrans, 31, 31, 31, 1.0f, 0.0f, 9402));

	COMMENT("sgemm TT 64x64x64");
	TEST(run_sgemm_trans_test(CblasTrans, CblasTrans, 64, 64, 64, 1.0f, 0.0f, 9403));

	COMMENT("sgemm TT 127x127x127");
	TEST(run_sgemm_trans_test(CblasTrans, CblasTrans, 127, 127, 127, 1.0f, 0.0f, 9404));

	COMMENT("sgemm TT 257x257x257");
	TEST(run_sgemm_trans_test(CblasTrans, CblasTrans, 257, 257, 257, 1.0f, 0.0f, 9405));

	COMMENT("sgemm TT non-square 65x63x127");
	TESTEX("TT non-square", run_sgemm_trans_test(CblasTrans, CblasTrans, 65, 63, 127, 1.0f, 0.0f, 9406));

	COMMENT("sgemm TT alpha=-1.0 beta=1.0");
	TESTEX("TT scaling", run_sgemm_trans_test(CblasTrans, CblasTrans, 64, 64, 64, -1.0f, 1.0f, 9407));
#endif

	COMMENT("dgemm TT 64x64x64");
	TEST(run_dgemm_trans_test(CblasTrans, CblasTrans, 64, 64, 64, 1.0, 0.0, 9500));

	COMMENT("dgemm TT 127x127x127");
	TEST(run_dgemm_trans_test(CblasTrans, CblasTrans, 127, 127, 127, 1.0, 0.0, 9501));

	COMMENT("dgemm TT 257x257x257");
	TEST(run_dgemm_trans_test(CblasTrans, CblasTrans, 257, 257, 257, 1.0, 0.0, 9502));
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

	MODULE("GEMM Accuracy Tests");

	test_small_matrices();
	test_tile_boundaries();
	test_large_blocks();
	test_nonsquare_matrices();
	test_scaling();
	test_transpose_TN();
	test_transpose_NT();
	test_transpose_TT();

	cblas_shutdown();

	return 0;
}
