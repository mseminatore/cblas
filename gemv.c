//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"
#include "cblas_simd.h"

#ifdef _WIN32
#	include <malloc.h>
#else
#	include <alloca.h>
#endif

// minimum stored-matrix width before the transposed (axpy form) GEMV is
// split across threads by columns
#define CBLAS_GEMV_T_MT_MIN_COLS 512

//------------------------------------------------------
// Level-2 single-precision matrix-vector multiply
// y = alpha * A * x + beta * y
//------------------------------------------------------
void cblas_sgemv(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, CBLAS_INDEX m, CBLAS_INDEX n, float alpha, float* a, CBLAS_INDEX lda, float* x, CBLAS_INDEX incx, float beta, float* y, CBLAS_INDEX incy)
{
#ifdef CBLAS_CHECK_INPUTS
	// For row-major: lda >= n (number of columns)
	// For column-major: lda >= m (number of rows)
	CBLAS_INDEX lda_min = (layout == CblasRowMajor) ? MAX(1, n) : MAX(1, m);

#ifdef CBLAS_XERBLA_INPUTS
	int info = 0;
	if (!a)
		info = 6;
	else if (lda < lda_min)
		info = 7;
	else if (!x)
		info = 8;
	else if (incx == 0)
		info = 9;
	else if (!y)
		info = 11;
	else if (incy == 0)
		info = 12;

	if (info) {
		XERBLA(info);
		return;
	}
#else
	if (!a || lda < lda_min || !x || incx == 0 || !y || incy == 0)
	{
		assert(m >= 0 && n >= 0 && a && lda >= lda_min && x && y && incx != 0 && incy != 0);
		return;
	}
#endif
#endif
	// early returns
	if (m == 0 || n == 0 || (alpha == 0.0f && beta == 1.0f))
		return;

    CBLAS_STATS_START();

	// A row-major A^T*x is a column-major A*x and vice versa, so only two
	// kernels are needed: dot form (walks rows of the stored matrix) and
	// axpy form (accumulates rows of the stored matrix into y)
	int dot_form = (trans == CblasNoTrans && layout == CblasRowMajor) || (trans == CblasTrans && layout == CblasColMajor);
	kernel_function kernel = dot_form ? blas_kernels.sgemv_k : blas_kernels.sgemv_t_k;

	// rows/cols of the stored (row-major view of the) matrix
	CBLAS_INDEX rows = (layout == CblasRowMajor) ? m : n;
	CBLAS_INDEX cols = (layout == CblasRowMajor) ? n : m;

#if defined(MT_ENABLED)
    // axpy form splits columns, so it needs enough of them that each thread
    // still streams contiguous runs of A
    int mt_used = (m * n > CBLAS_MT_GEMV && cblas_get_num_threads() > 1 &&
                   (dot_form || cols >= CBLAS_GEMV_T_MT_MIN_COLS)) ? 1 : 0;

    if (mt_used)
    {
        // dot form splits rows (and y); axpy form splits columns (and y)
        cblas_level2_exec(sizeof(float), kernel, dot_form ? CBLAS_PART_Y : CBLAS_PART_COLS,
                          rows, cols, a, lda, x, incx, y, incy, &alpha, &beta, "SGEMV");
    }
    else
#else
    int mt_used = 0;
#endif
    {
		cblas_args_t args = {
			.x = x, .y = y, .a = a,
			.m = rows, .n = cols,
			.incx = incx, .incy = incy,
			.lda = lda,
			.alpha = &alpha, .beta = &beta
		};
		kernel(&args);
    }

    CBLAS_STATS_END("sgemv", m * n, mt_used);
}

//------------------------------------------------------
// Level-2 double-precision matrix-vector multiply
// y = alpha * A * x + beta * y
//------------------------------------------------------
void cblas_dgemv(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, CBLAS_INDEX m, CBLAS_INDEX n, double alpha, double* a, CBLAS_INDEX lda, double* x, CBLAS_INDEX incx, double beta, double* y, CBLAS_INDEX incy)
{
#ifdef CBLAS_CHECK_INPUTS
	// For row-major: lda >= n (number of columns)
	// For column-major: lda >= m (number of rows)
	CBLAS_INDEX lda_min = (layout == CblasRowMajor) ? MAX(1, n) : MAX(1, m);

#ifdef CBLAS_XERBLA_INPUTS
	int info = 0;
	if (!a)
		info = 6;
	else if (lda < lda_min)
		info = 7;
	else if (!x)
		info = 8;
	else if (incx == 0)
		info = 9;
	else if (!y)
		info = 11;
	else if (incy == 0)
		info = 12;

	if (info) {
		XERBLA(info);
		return;
	}
#else
	if (!a || lda < lda_min || !x || incx == 0 || !y || incy == 0)
	{
		assert(m >= 0 && n >= 0 && a && lda >= lda_min && x && y && incx != 0 && incy != 0);
		return;
	}
#endif
#endif
	// early returns
	if (m == 0 || n == 0 || (alpha == 0.0 && beta == 1.0))
		return;

    CBLAS_STATS_START();

	// A row-major A^T*x is a column-major A*x and vice versa, so only two
	// kernels are needed: dot form (walks rows of the stored matrix) and
	// axpy form (accumulates rows of the stored matrix into y)
	int dot_form = (trans == CblasNoTrans && layout == CblasRowMajor) || (trans == CblasTrans && layout == CblasColMajor);
	kernel_function kernel = dot_form ? blas_kernels.dgemv_k : blas_kernels.dgemv_t_k;

	// rows/cols of the stored (row-major view of the) matrix
	CBLAS_INDEX rows = (layout == CblasRowMajor) ? m : n;
	CBLAS_INDEX cols = (layout == CblasRowMajor) ? n : m;

#if defined(MT_ENABLED)
    // axpy form splits columns, so it needs enough of them that each thread
    // still streams contiguous runs of A
    int mt_used = (m * n > CBLAS_MT_GEMV && cblas_get_num_threads() > 1 &&
                   (dot_form || cols >= CBLAS_GEMV_T_MT_MIN_COLS)) ? 1 : 0;

    if (mt_used)
    {
        // dot form splits rows (and y); axpy form splits columns (and y)
        cblas_level2_exec(sizeof(double), kernel, dot_form ? CBLAS_PART_Y : CBLAS_PART_COLS,
                          rows, cols, a, lda, x, incx, y, incy, &alpha, &beta, "DGEMV");
    }
    else
#else
    int mt_used = 0;
#endif
    {
		cblas_args_t args = {
			.x = x, .y = y, .a = a,
			.m = rows, .n = cols,
			.incx = incx, .incy = incy,
			.lda = lda,
			.alpha = &alpha, .beta = &beta
		};
		kernel(&args);
    }

    CBLAS_STATS_END("dgemv", m * n, mt_used);
}
