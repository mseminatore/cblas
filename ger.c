//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"

#define X(i) x[(i) * incx]
#define Y(i) y[(i) * incy]
#define A(col, row) a[(row) * lda + (col)]

//
static void AddProd(float x, float y, float *a)
{
	*a += x * y;
}

//
static void AddProd4x1(float *x, float *y, float *a)
{

}

//
static void sger_row_noalpha(CBLAS_INDEX m, CBLAS_INDEX n, float *x, CBLAS_INDEX incx, float *y, CBLAS_INDEX incy, float *a, CBLAS_INDEX lda)
{
	register float xr;
	float *yc, *ap;

	for (int row = 0; row < m; row++)
	{
		xr = X(row);
		for (int col = 0; col < n; col += 4)
		{
			yc = &Y(col);
			ap = &A(col, row);

			AddProd(xr, *yc, ap);
			AddProd(xr, *(yc + 1), ap + 1);
			AddProd(xr, *(yc + 2), ap + 2);
			AddProd(xr, *(yc + 3), ap + 3);
		}
	}
}

//
static void sger_row_noalpha_plain(CBLAS_INDEX m, CBLAS_INDEX n, float *x, CBLAS_INDEX incx, float *y, CBLAS_INDEX incy, float *a, CBLAS_INDEX lda)
{
	for (int row = 0; row < m; row++)
	{
		for (int col = 0; col < n; col++)
		{
			a[row * n + col] += x[row] * y[col];
		}
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

	int info = 0;
	if (layout != CblasRowMajor && layout != CblasColMajor)
		info = 1;
	else if (m < 0)
		info = 2;
	else if (n < 0)
		info = 3;
	else if (!x)
		info = 5;
	else if (incx == 0)
		info = 6;
	else if (!y)
		info = 7;
	else if (incy == 0)
		info = 8;
	else if (!a)
		info = 9;
	else if (lda < MAX(1, m))
		info = 10;
	
	if (info)  {
		XERBLA(info);
		return;
	}

	if (m == 0 ||  n == 0 || alpha == 0.0f)
		return;

	// TODO - handle incx == 1 special case
    if (layout == CblasRowMajor)
    {
		if (alpha == 1.0f)
		{
			sger_row_noalpha_plain(m, n,  x, incx, y, incy, a, lda);
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

	int info = 0;
	if (layout != CblasRowMajor && layout != CblasColMajor)
		info = 1;
	else if (m < 0)
		info = 2;
	else if (n < 0)
		info = 3;
	else if (!x)
		info = 5;
	else if (incx == 0)
		info = 6;
	else if (!y)
		info = 7;
	else if (incy == 0)
		info = 8;
	else if (!a)
		info = 9;
	else if (lda < MAX(1, m))
		info = 10;
	
	if (info)  {
		XERBLA(info);
		return;
	}

	if (m == 0 ||  n == 0 || alpha == 0.0f)
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
