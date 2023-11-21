//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"

//------------------------------------------------------
// y = alpha * A * x + beta * y
//------------------------------------------------------
void cblas_sgemv(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, CBLAS_INDEX m, CBLAS_INDEX n, float alpha, float *a, CBLAS_INDEX lda, float *x, CBLAS_INDEX incx, float beta, float *y, CBLAS_INDEX incy)
{
	float sum;

    assert(m > 0 && n > 0);

	if ((trans == CblasNoTrans && layout == CblasRowMajor) || (trans == CblasTrans && layout == CblasColMajor))
	{
		// for each row of the matrix
		for (int row = 0; row < m; row++)
		{
			sum = 0.0f;

			for (int col = 0; col < n; col++)
			{
				sum += alpha * a[row * n + col] * x[col];
			}

			y[row] = beta * y[row] + sum;
		}
	}
	else
	{
		for (int col = 0; col < n; col++)
		{
			sum = 0.0f;

			for (int row = 0; row < m; row++)
			{
				sum += alpha * a[row * n + col] * x[row];
			}

			y[col] = beta * y[col] + sum;
		}
	}
}

//------------------------------------------------------
//
//------------------------------------------------------
void cblas_dgemv(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, CBLAS_INDEX m, CBLAS_INDEX n, double alpha, double *a, CBLAS_INDEX lda, double *x, CBLAS_INDEX incx, double beta, double *y, CBLAS_INDEX incy)
{
	double sum;

    assert(m > 0 && n > 0);

	if ((trans == CblasNoTrans && layout == CblasRowMajor) || (trans == CblasTrans && layout == CblasColMajor))
	{
		// for each row of the matrix
		for (int row = 0; row < m; row++)
		{
			sum = 0.0;

			for (int col = 0; col < n; col++)
			{
				sum += alpha * a[row * n + col] * x[col];
			}

			y[row] = beta * y[row] + sum;
		}
	}
	else
	{
		for (int col = 0; col < n; col++)
		{
			sum = 0.0;

			for (int row = 0; row < m; row++)
			{
				sum += alpha * a[row * n + col] * x[row];
			}

			y[col] = beta * y[col] + sum;
		}
	}
}
