//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"

//------------------------------------------------------
// Level-2 single-precision matrix-vector multiply
// y = alpha * A * x + beta * y
//------------------------------------------------------
void cblas_sgemv(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, CBLAS_INDEX m, CBLAS_INDEX n, float alpha, float* a, CBLAS_INDEX lda, float* x, CBLAS_INDEX incx, float beta, float* y, CBLAS_INDEX incy)
{
	float sum;

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

    int mt_used = (m * n > CBLAS_MT_GEMV) ? 1 : 0;

	if ((trans == CblasNoTrans && layout == CblasRowMajor) || (trans == CblasTrans && layout == CblasColMajor))
	{
		if (alpha == 1.0f && beta == 1.0f)
		{
			// for each row of the matrix
			for (CBLAS_INDEX row = 0; row < m; row++)
			{
				sum = 0.0f;

				for (CBLAS_INDEX col = 0; col < n; col++)
				{
					sum += a[row * n + col] * x[col];
				}

				y[row] = y[row] + sum;
			}
		}
		else
		{
			// for each row of the matrix
			for (CBLAS_INDEX row = 0; row < m; row++)
			{
				sum = 0.0f;

				for (CBLAS_INDEX col = 0; col < n; col++)
				{
					sum += alpha * a[row * n + col] * x[col];
				}

				y[row] = beta * y[row] + sum;
			}
		}
	}
	else
	{
		if (alpha == 1.0f && beta == 1.0f)
		{
			for (CBLAS_INDEX col = 0; col < n; col++)
			{
				sum = 0.0f;

				for (CBLAS_INDEX row = 0; row < m; row++)
				{
					sum += a[row * n + col] * x[row];
				}

				y[col] = y[col] + sum;
			}
		}
		else
		{
			for (CBLAS_INDEX col = 0; col < n; col++)
			{
				sum = 0.0f;

				for (CBLAS_INDEX row = 0; row < m; row++)
				{
					sum += alpha * a[row * n + col] * x[row];
				}

				y[col] = beta * y[col] + sum;
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
	double sum;
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

    int mt_used = (m * n > CBLAS_MT_GEMV) ? 1 : 0;

	if ((trans == CblasNoTrans && layout == CblasRowMajor) || (trans == CblasTrans && layout == CblasColMajor))
	{
		if (alpha == 1.0f && beta == 1.0f)
		{
			// for each row of the matrix
			for (CBLAS_INDEX row = 0; row < m; row++)
			{
				sum = 0.0;

				for (CBLAS_INDEX col = 0; col < n; col++)
				{
					sum += a[row * n + col] * x[col];
				}

				y[row] = y[row] + sum;
			}
		}
		else
		{
			// for each row of the matrix
			for (CBLAS_INDEX row = 0; row < m; row++)
			{
				sum = 0.0;

				for (CBLAS_INDEX col = 0; col < n; col++)
				{
					sum += alpha * a[row * n + col] * x[col];
				}

				y[row] = beta * y[row] + sum;
			}
		}
	}
	else
	{
		if (alpha == 1.0 && beta == 1.0)
		{
			for (CBLAS_INDEX col = 0; col < n; col++)
			{
				sum = 0.0;

				for (CBLAS_INDEX row = 0; row < m; row++)
				{
					sum += a[row * n + col] * x[row];
				}

				y[col] = y[col] + sum;
			}
		}
		else
		{
			for (CBLAS_INDEX col = 0; col < n; col++)
			{
				sum = 0.0;

				for (CBLAS_INDEX row = 0; row < m; row++)
				{
					sum += alpha * a[row * n + col] * x[row];
				}

				y[col] = beta * y[col] + sum;
			}
		}
	}

    CBLAS_STATS_END("dgemv", m * n, mt_used);
}
