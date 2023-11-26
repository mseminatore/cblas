//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"

// macro to simpify matrix element access
#define A(col, row) a[(row) * lda + (col)]

//
void AddDot1x4(CBLAS_INDEX k, float *x, CBLAS_INDEX incx, float *y, float *a, CBLAS_INDEX lda)
{
	for (int p = 0; p < k; p++)
	{
		A(0, 0) += *x * y[p];
		A(1, 0) += *x * y[p + 1];
		A(2, 0) += *x * y[p + 2];
		A(3, 0) += *x * y[p + 3];
	}
}

//
void AddDot(CBLAS_INDEX n, float *x, CBLAS_INDEX incx, float *y, float *a, CBLAS_INDEX lda)
{
	for (int p = 0; p < n; p++)
	{
		*a += *x * y[p];
	}
}

//------------------------------------------------------
// single-precision rank-1 update
//------------------------------------------------------
void cblas_sger(CBLAS_LAYOUT layout, CBLAS_INDEX m, CBLAS_INDEX n, float alpha, float *x, CBLAS_INDEX incx, float *y, CBLAS_INDEX incy, float *a, CBLAS_INDEX lda)
{
	assert(m > 0 && n > 0 && incx != 0 && incy != 0 && alpha != 0.0f);
	assert(x && y && a);
	assert(layout == CblasRowMajor || layout == CblasColMajor);
	assert(lda >= MAX(1, m));

	if (m <= 0 || n <= 0 || alpha == 0.0f)
		return;

	if (layout != CblasRowMajor && layout != CblasColMajor)
		return;

	// TODO - handle incx == 1 special case
    if (layout == CblasRowMajor)
    {
		if (alpha == 1.0f)
		{
			for (int row = 0; row < m; row ++)
			{
				for (int col = 0; col < n; col++)
				{
					a[row * n + col] += x[row] * y[col];
				}
			}
		}
		else
		{
			for (int row = 0; row < m; row++)
			{
				for (int col = 0; col < n; col++)
				{
					a[row * n + col] += alpha * x[row] * y[col];
				}
			}
		}
    } else
    {
		if (alpha == 1.0f)
		{
			for (int col = 0; col < n; col++)
			{
				for (int row = 0; row < m; row++)
				{
					a[col * m + row] += x[row] * y[col];
				}
			}
		}
		else
		{
			for (int col = 0; col < n; col++)
			{
				for (int row = 0; row < m; row++)
				{
					a[col * m + row] += alpha * x[row] * y[col];
				}
			}
		}
	}
}

//------------------------------------------------------
// double-precision rank-1 update
//------------------------------------------------------
void cblas_dger(CBLAS_LAYOUT layout, CBLAS_INDEX m, CBLAS_INDEX n, double alpha, double *x, CBLAS_INDEX incx, double *y, CBLAS_INDEX incy, double *a, CBLAS_INDEX lda)
{
	assert(m > 0 && n > 0 && incx != 0 && incy != 0 && alpha != 0.0f);
	assert(x && y && a);
	assert(layout == CblasRowMajor || layout == CblasColMajor);
	assert(lda >= MAX(1, m));

	if (m <= 0 || n <= 0 || alpha == 0.0f)
		return;

	if (layout != CblasRowMajor && layout != CblasColMajor)
		return;

    if (layout == CblasRowMajor)
    {
		if (alpha == 1.0)
		{
			for (int row = 0; row < m; row++)
			{
				for (int col = 0; col < n; col++)
				{
					a[row * n + col] += x[row] * y[col];
				}
			}
		}
		else
		{
			for (int row = 0; row < m; row++)
			{
				for (int col = 0; col < n; col++)
				{
					a[row * n + col] += alpha * x[row] * y[col];
				}
			}
		}
    } else
    {
		if (alpha == 1.0)
		{
			for (int col = 0; col < n; col++)
			{
				for (int row = 0; row < m; row++)
				{
					a[col * m + row] += x[row] * y[col];
				}
			}
		}
		else
		{
			for (int col = 0; col < n; col++)
			{
				for (int row = 0; row < m; row++)
				{
					a[col * m + row] += alpha * x[row] * y[col];
				}
			}
		}
	}

}
