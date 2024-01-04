//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)
#   include <immintrin.h>
#endif

#if defined(__aarch64__) && defined(__ARM_NEON)
#   include <arm_neon.h>
#endif

// helpful macros
#define X(i) x[(i) * incx]
#define Y(i) y[(i) * incy]
#define A(col, row) a[(row) * lda + (col)]

//------------------------------------------------------
// compute single element product
//------------------------------------------------------
static void AddProd(float x, float y, float *a)
{
	*a += x * y;
}

//------------------------------------------------------
// compute 4 cols by 1 row product
//------------------------------------------------------
static void AddProd4x1(float x, float *y, float *a)
{
#if 0

#if defined(__aarch64__)
	float32x4_t xr, yr, ar;

	// load and dup x
	xr = vld1q_dup_f32(&x);

	// load 4 floats of Y and A
	yr = vld1q_f32(y);
	ar = vld1q_f32(a);
	
	// A += x * y
	ar = vfmaq_f32(ar, xr, yr);
	
	// store 4 floats
	vst1q_f32(a, ar);
#else
	__m128 xr, yr, ar;

	// load and dup x
	xr = _mm_load_ps1(&x);

	// load 4 floats of Y and A
	yr = _mm_load_ps(y);
	ar = _mm_load_ps(a);

	// A += x * y
#ifdef USE_INTEL_FMA
	ar = _mm_fmadd_ps(ar, xr, yr);
#else
	ar = _mm_add_ps(ar, _mm_mul_ps(xr, yr));
#endif

	// store 4 floats
	_mm_store_ps(a, ar);
#endif

#else
	*a += x * *y;
	*(a + 1) += x * *(y + 1);
	*(a + 2) += x * *(y + 2);
	*(a + 3) += x * *(y + 3);
#endif
}

//------------------------------------------------------
// compute 4 cols x 4 rows product
//------------------------------------------------------
static void AddProd4x4_SSE(float* x, float* y, float* a, CBLAS_INDEX lda)
{
	__m128 x0, x1, x2, x3, y0, a0, a1, a2, a3;
	float* ap1, *ap2, *ap3;

	ap1 = a + lda;
	ap2 = ap1 + lda;
	ap3 = ap2 + lda;

	x0 = _mm_load_ps1(x);
	x1 = _mm_load_ps1(x + 1);
	x2 = _mm_load_ps1(x + 2);
	x3 = _mm_load_ps1(x + 3);
	
	y0 = _mm_load_ps(y);
	
	a0 = _mm_load_ps(a);
	a1 = _mm_load_ps(ap1);
	a2 = _mm_load_ps(ap2);
	a3 = _mm_load_ps(ap3);

	// compute 4x4 product
	a0 = _mm_add_ps(a0, _mm_mul_ps(x0, y0));
	a1 = _mm_add_ps(a1, _mm_mul_ps(x1, y0));
	a2 = _mm_add_ps(a2, _mm_mul_ps(x2, y0));
	a3 = _mm_add_ps(a3, _mm_mul_ps(x3, y0));

	// store results
	_mm_store_ps(a, a0);
	_mm_store_ps(a, a1);
	_mm_store_ps(a, a2);
	_mm_store_ps(a, a3);
}

//------------------------------------------------------
// compute 4 cols x 4 rows product
//------------------------------------------------------
static void AddProd4x4(float* x, float* y, float* a, CBLAS_INDEX lda)
{
	// first row
	*a = *x * *y;
	*(a + 1) = *x * *(y + 1);
	*(a + 2) = *x * *(y + 2);
	*(a + 3) = *x * *(y + 3);

	a += lda;
	x++;

	// second row
	*a = *x * *y;
	*(a + 1) = *x * *(y + 1);
	*(a + 2) = *x * *(y + 2);
	*(a + 3) = *x * *(y + 3);

	a += lda;
	x++;

	// third row
	*a = *x * *y;
	*(a + 1) = *x * *(y + 1);
	*(a + 2) = *x * *(y + 2);
	*(a + 3) = *x * *(y + 3);

	a += lda;
	x++;

	// fourth row
	*a = *x * *y;
	*(a + 1) = *x * *(y + 1);
	*(a + 2) = *x * *(y + 2);
	*(a + 3) = *x * *(y + 3);
}

//------------------------------------------------------
//
//------------------------------------------------------
static void sger_row_noalpha4x4(CBLAS_INDEX m, CBLAS_INDEX n, float* x, CBLAS_INDEX incx, float* y, CBLAS_INDEX incy, float* a, CBLAS_INDEX lda)
{
	float*xr, * yc, * ap;
	int col, row;

	for (row = 0; row + 4 <= m; row += 4)
	{
		xr = &X(row);
		yc = y;
		ap = &A(0, row);

		for (col = 0; col + 4 <= n; col += 4)
		{
			AddProd4x4(xr, yc, ap, lda);
			yc += 4;
			ap += 4;
		}

		// handle leftover cols
		switch (n - col)
		{
		case 3: AddProd(*xr, Y(col + 2), &A(col + 2, row));
		case 2: AddProd(*xr, Y(col + 1), &A(col + 1, row));
		case 1: AddProd(*xr, Y(col), &A(col, row));
		case 0:;	// do nothing!
		}
	}

	// handle leftover rows
	switch (m - row)
	{
	case 3: for (col = 0; col < n; col++) AddProd(X(row + 2), Y(col), &A(col, row + 2));
	case 2: for (col = 0; col < n; col++) AddProd(X(row + 1), Y(col), &A(col, row + 1));
	case 1: for (col = 0; col < n; col++) AddProd(X(row), Y(col), &A(col, row));
	case 0:;	// do nothing!
	}
}

//------------------------------------------------------
//
//------------------------------------------------------
static void sger_row_noalpha(CBLAS_INDEX m, CBLAS_INDEX n, float *x, CBLAS_INDEX incx, float *y, CBLAS_INDEX incy, float *a, CBLAS_INDEX lda)
{
	register float xr;
	float *yc, *ap;
	int col;

	for (int row = 0; row < m; row++)
	{
		xr = X(row);
		yc = y;
		ap = &A(0, row);

		for (col = 0; col + 4 <= n; col += 4)
		{
			AddProd4x1(xr, yc, ap);
			yc += 4;
			ap += 4;
		}

		// handle leftover cols
		switch (n - col)
		{
		case 3: AddProd(xr, Y(col + 2), &A(col + 2, row));
		case 2: AddProd(xr, Y(col + 1), &A(col + 1, row));
		case 1: AddProd(xr, Y(col), &A(col, row));
		case 0: ;	// do nothing!
		}
	}
}

//------------------------------------------------------
// 
//------------------------------------------------------
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

	// fast reject case
	if (m == 0 ||  n == 0 || alpha == 0.0f)
		return;

	// TODO - handle incx == 1 special case
    if (layout == CblasRowMajor)
    {
		if (alpha == 1.0f)
		{
			sger_row_noalpha(m, n,  x, incx, y, incy, a, lda);
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
