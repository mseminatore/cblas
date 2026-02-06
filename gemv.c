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

//------------------------------------------------------
// Level-2 single-precision matrix-vector multiply
// y = alpha * A * x + beta * y
//------------------------------------------------------
void cblas_sgemv(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, CBLAS_INDEX m, CBLAS_INDEX n, float alpha, float* a, CBLAS_INDEX lda, float* x, CBLAS_INDEX incx, float beta, float* y, CBLAS_INDEX incy)
{
#ifdef CBLAS_CHECK_INPUTS

#ifdef CBLAS_XERBLA_INPUTS
	int info = 0;
	if (!a)
		info = 6;
	else if (lda < MAX(1, m))
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
	if (!a || lda < MAX(1, m) || !x || incx == 0 || !y || incy == 0)
	{
		assert(m >= 0 && n >= 0 && a && lda >= MAX(1, m) && x && y && incx != 0 && incy != 0);
		return;
	}
#endif
#endif
	// early returns
	if (m == 0 || n == 0 || (alpha == 0.0f && beta == 1.0f))
		return;

    CBLAS_STATS_START();

#if defined(MT_ENABLED)
    int mt_used = (m * n > CBLAS_MT_GEMV) ? 1 : 0;
    
    // Multi-threaded path for NoTrans case with RowMajor or Trans case with ColMajor
    if (mt_used && ((trans == CblasNoTrans && layout == CblasRowMajor) || (trans == CblasTrans && layout == CblasColMajor)) && blas_kernels.sgemv_k != NULL)
    {
        cblas_level2_exec(sizeof(float), blas_kernels.sgemv_k, CBLAS_PART_Y,
                          m, n, a, lda, x, incx, y, incy, &alpha, &beta, "SGEMV");
    }
    else
#else
    int mt_used = 0;
#endif
    {
        // Single-threaded fallback path
	if ((trans == CblasNoTrans && layout == CblasRowMajor) || (trans == CblasTrans && layout == CblasColMajor))
	{
		// Use dispatched kernel (ISA-specific, selected at init time)
		cblas_args_t args = {
			.x = x, .y = y, .a = a,
			.m = m, .n = n,
			.incx = incx, .incy = incy,
			.lda = lda,
			.alpha = &alpha, .beta = &beta
		};
		blas_kernels.sgemv_k(&args);
	}
	else
	{
		// Transpose case: column-wise access
		float sum;
		if (incx == 1 && incy == 1)
		{
			for (CBLAS_INDEX col = 0; col < n; col++)
			{
				sum = 0.0f;
				for (CBLAS_INDEX row = 0; row < m; row++)
				{
					sum += a[row * lda + col] * x[row];
				}
				y[col] = beta * y[col] + alpha * sum;
			}
		}
		else
		{
			for (CBLAS_INDEX col = 0; col < n; col++)
			{
				sum = 0.0f;
				for (CBLAS_INDEX row = 0; row < m; row++)
				{
					sum += a[row * lda + col] * x[row * incx];
				}
				y[col * incy] = beta * y[col * incy] + alpha * sum;
			}
		}
	}
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

#ifdef CBLAS_XERBLA_INPUTS
	int info = 0;
	if (!a)
		info = 6;
	else if (lda < MAX(1, m))
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
	if (!a || lda < MAX(1, m) || !x || incx == 0 || !y || incy == 0)
	{
		assert(m >= 0 && n >= 0 && a && lda >= MAX(1, m) && x && y && incx != 0 && incy != 0);
		return;
	}
#endif
#endif
	// early returns
	if (m == 0 || n == 0 || (alpha == 0.0 && beta == 1.0))
		return;

    CBLAS_STATS_START();

#if defined(MT_ENABLED)
    int mt_used = (m * n > CBLAS_MT_GEMV) ? 1 : 0;
    
    // Multi-threaded path for NoTrans case with RowMajor or Trans case with ColMajor
    if (mt_used && ((trans == CblasNoTrans && layout == CblasRowMajor) || (trans == CblasTrans && layout == CblasColMajor)) && blas_kernels.dgemv_k != NULL)
    {
        cblas_level2_exec(sizeof(double), blas_kernels.dgemv_k, CBLAS_PART_Y,
                          m, n, a, lda, x, incx, y, incy, &alpha, &beta, "DGEMV");
    }
    else
#else
    int mt_used = 0;
#endif
    {
        // Single-threaded fallback path
	if ((trans == CblasNoTrans && layout == CblasRowMajor) || (trans == CblasTrans && layout == CblasColMajor))
	{
		// Use dispatched kernel (ISA-specific, selected at init time)
		cblas_args_t args = {
			.x = x, .y = y, .a = a,
			.m = m, .n = n,
			.incx = incx, .incy = incy,
			.lda = lda,
			.alpha = &alpha, .beta = &beta
		};
		blas_kernels.dgemv_k(&args);
	}
	else
	{
		// Transpose case: column-wise access
		double sum;
		if (incx == 1 && incy == 1)
		{
			for (CBLAS_INDEX col = 0; col < n; col++)
			{
				sum = 0.0;
				for (CBLAS_INDEX row = 0; row < m; row++)
				{
					sum += a[row * lda + col] * x[row];
				}
				y[col] = beta * y[col] + alpha * sum;
			}
		}
		else
		{
			for (CBLAS_INDEX col = 0; col < n; col++)
			{
				sum = 0.0;
				for (CBLAS_INDEX row = 0; row < m; row++)
				{
					sum += a[row * lda + col] * x[row * incx];
				}
				y[col * incy] = beta * y[col * incy] + alpha * sum;
			}
		}
	}
    }

    CBLAS_STATS_END("dgemv", m * n, mt_used);
}
