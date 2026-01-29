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
// compute 8 cols x 4 rows product (non-FMA version)
//------------------------------------------------------
static void AddProd8x4_AVX(float* x, float* y, float* a, CBLAS_INDEX lda)
{
#if defined(__aarch64__)
#else
	__m256 x0, x1, x2, x3, y0, a0, a1, a2, a3;

	// copy single FP to all 8 elements of vector
	x0 = _mm256_broadcast_ss(x);
	x1 = _mm256_broadcast_ss(x + 1);
	x2 = _mm256_broadcast_ss(x + 2);
	x3 = _mm256_broadcast_ss(x + 3);

	y0 = _mm256_load_ps(y);

	// load 4 rows of destination
	a0 = _mm256_load_ps(a);
	a1 = _mm256_load_ps(a + lda);
	a2 = _mm256_load_ps(a + 2 * lda);
	a3 = _mm256_load_ps(a + 3 * lda);

	// compute 8x4 products (non-FMA)
	a0 = _mm256_add_ps(a0, _mm256_mul_ps(x0, y0));
	a1 = _mm256_add_ps(a1, _mm256_mul_ps(x1, y0));
	a2 = _mm256_add_ps(a2, _mm256_mul_ps(x2, y0));
	a3 = _mm256_add_ps(a3, _mm256_mul_ps(x3, y0));

	// store results
	_mm256_store_ps(a, a0);
	_mm256_store_ps(a + lda, a1);
	_mm256_store_ps(a + 2 * lda, a2);
	_mm256_store_ps(a + 3 * lda, a3);
#endif
}

#if defined(USE_SSE) && defined(USE_SIMD) && (defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86))

//------------------------------------------------------
// compute 8 cols x 4 rows product (FMA version)
//------------------------------------------------------
static void AddProd8x4_AVX_fma(float* x, float* y, float* a, CBLAS_INDEX lda)
{
	__m256 x0, x1, x2, x3, y0, a0, a1, a2, a3;

	// copy single FP to all 8 elements of vector
	x0 = _mm256_broadcast_ss(x);
	x1 = _mm256_broadcast_ss(x + 1);
	x2 = _mm256_broadcast_ss(x + 2);
	x3 = _mm256_broadcast_ss(x + 3);

	y0 = _mm256_load_ps(y);

	// load 4 rows of destination
	a0 = _mm256_load_ps(a);
	a1 = _mm256_load_ps(a + lda);
	a2 = _mm256_load_ps(a + 2 * lda);
	a3 = _mm256_load_ps(a + 3 * lda);

	// compute 8x4 products using FMA
	a0 = _mm256_fmadd_ps(x0, y0, a0);
	a1 = _mm256_fmadd_ps(x1, y0, a1);
	a2 = _mm256_fmadd_ps(x2, y0, a2);
	a3 = _mm256_fmadd_ps(x3, y0, a3);

	// store results
	_mm256_store_ps(a, a0);
	_mm256_store_ps(a + lda, a1);
	_mm256_store_ps(a + 2 * lda, a2);
	_mm256_store_ps(a + 3 * lda, a3);
}

#endif

//------------------------------------------------------
// compute 4 cols x 4 rows product
//------------------------------------------------------
static void AddProd4x4_SIMD(float* x, float* y, float* a, CBLAS_INDEX lda)
{
#if defined(__aarch64__)
	float32x4_t x0, x1, x2, x3, y0, a0, a1, a2, a3;

	x0 = vld1q_dup_f32(x);
	x1 = vld1q_dup_f32(x + 1);
	x2 = vld1q_dup_f32(x + 2);
	x3 = vld1q_dup_f32(x + 3);

	y0 = vld1q_f32(y);

	a0 = vld1q_f32(a);
	a1 = vld1q_f32(a + lda);
	a2 = vld1q_f32(a + 2 * lda);
	a3 = vld1q_f32(a + 3 * lda);

#ifdef __ARM_FEATURE_FMA

	// compute 4x4 product using FMA
	a0 = vfmaq_f32(a0, x0, y0);
	a1 = vfmaq_f32(a1, x1, y0);
	a2 = vfmaq_f32(a2, x2, y0);
	a3 = vfmaq_f32(a3, x3, y0);

#else

	// rows 1 - 4 using NEON MUL and ADD A += X * Y
	a0 = vaddq_f32(a0, vmulq_f32(x0, y0));
	a1 = vaddq_f32(a1, vmulq_f32(x1, y0));
	a2 = vaddq_f32(a2, vmulq_f32(x2, y0));
	a3 = vaddq_f32(a3, vmulq_f32(x3, y0));

#endif

    // store 4x4 floats
    vst1q_f32(a, a0);
    vst1q_f32(a + lda, a1);
    vst1q_f32(a + 2 * lda, a2);
    vst1q_f32(a + 3 * lda, a3);

#else
	__m128 x0, x1, x2, x3, y0, a0, a1, a2, a3;

	x0 = _mm_load_ps1(x);
	x1 = _mm_load_ps1(x + 1);
	x2 = _mm_load_ps1(x + 2);
	x3 = _mm_load_ps1(x + 3);
	
	y0 = _mm_load_ps(y);
	
	a0 = _mm_loadu_ps(a);
	a1 = _mm_loadu_ps(a + lda);
	a2 = _mm_loadu_ps(a + 2 * lda);
	a3 = _mm_loadu_ps(a + 3 * lda);

	// compute 4x4 product (non-FMA)
	a0 = _mm_add_ps(a0, _mm_mul_ps(x0, y0));
	a1 = _mm_add_ps(a1, _mm_mul_ps(x1, y0));
	a2 = _mm_add_ps(a2, _mm_mul_ps(x2, y0));
	a3 = _mm_add_ps(a3, _mm_mul_ps(x3, y0));

	// store results
	_mm_storeu_ps(a, a0);
	_mm_storeu_ps(a + lda, a1);
	_mm_storeu_ps(a + 2 * lda, a2);
	_mm_storeu_ps(a + 3 * lda, a3);
#endif
}

#if defined(USE_SSE) && defined(USE_SIMD) && (defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86))

//------------------------------------------------------
// compute 4 cols x 4 rows product (FMA version)
//------------------------------------------------------
static void AddProd4x4_SIMD_fma(float* x, float* y, float* a, CBLAS_INDEX lda)
{
	__m128 x0, x1, x2, x3, y0, a0, a1, a2, a3;

	x0 = _mm_load_ps1(x);
	x1 = _mm_load_ps1(x + 1);
	x2 = _mm_load_ps1(x + 2);
	x3 = _mm_load_ps1(x + 3);
	
	y0 = _mm_load_ps(y);
	
	a0 = _mm_loadu_ps(a);
	a1 = _mm_loadu_ps(a + lda);
	a2 = _mm_loadu_ps(a + 2 * lda);
	a3 = _mm_loadu_ps(a + 3 * lda);

	// compute 4x4 product using FMA
	a0 = _mm_fmadd_ps(x0, y0, a0);
	a1 = _mm_fmadd_ps(x1, y0, a1);
	a2 = _mm_fmadd_ps(x2, y0, a2);
	a3 = _mm_fmadd_ps(x3, y0, a3);

	// store results
	_mm_storeu_ps(a, a0);
	_mm_storeu_ps(a + lda, a1);
	_mm_storeu_ps(a + 2 * lda, a2);
	_mm_storeu_ps(a + 3 * lda, a3);
}

#endif

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
static void sger_row_noalpha8x4(CBLAS_INDEX m, CBLAS_INDEX n, float* x, CBLAS_INDEX incx, float* y, CBLAS_INDEX incy, float* a, CBLAS_INDEX lda)
{
	float* xr, * yc, * ap;
	CBLAS_INDEX col, row;

	for (row = 0; row + 4 <= m; row += 4)
	{
		xr = &X(row);
		yc = y;
		ap = &A(0, row);

		for (col = 0; col + 8 <= n; col += 8)
		{
			AddProd8x4_AVX(xr, yc, ap, lda);
			yc += 8;
			ap += 8;
		}

		// handle leftover cols handling each of the 4 rows in this block
		for (CBLAS_INDEX i = 0; i < 4; i++)
		{
			switch (n - col)
			{
			case 7: AddProd(*xr, Y(col + 6), &A(col + 6, row + i));
			case 6: AddProd(*xr, Y(col + 5), &A(col + 5, row + i));
			case 5: AddProd(*xr, Y(col + 4), &A(col + 4, row + i));
			case 4: AddProd(*xr, Y(col + 3), &A(col + 3, row + i));
			case 3: AddProd(*xr, Y(col + 2), &A(col + 2, row + i));
			case 2: AddProd(*xr, Y(col + 1), &A(col + 1, row + i));
			case 1: AddProd(*xr, Y(col), &A(col, row + i));
			case 0:;	// do nothing!
			}

			xr = &X(row + i);
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
static void sger_row_noalpha4x4(CBLAS_INDEX m, CBLAS_INDEX n, float* x, CBLAS_INDEX incx, float* y, CBLAS_INDEX incy, float* a, CBLAS_INDEX lda)
{
	float *xr, *yc, *ap;
	CBLAS_INDEX col, row;

	for (row = 0; row + 4 <= m; row += 4)
	{
		xr = &X(row);
		yc = y;
		ap = &A(0, row);

		for (col = 0; col + 4 <= n; col += 4)
		{
			AddProd4x4_SIMD(xr, yc, ap, lda);
			yc += 4;
			ap += 4;
		}

		// handle leftover cols handling each of the 4 rows in this block
		for (CBLAS_INDEX i = 0; i < 4; i++)
		{
			switch (n - col)
			{
			case 3: AddProd(*xr, Y(col + 2), &A(col + 2, row + i));
			case 2: AddProd(*xr, Y(col + 1), &A(col + 1, row + i));
			case 1: AddProd(*xr, Y(col), &A(col, row + i));
			case 0:;	// do nothing!
			}

			xr = &X(row + i);
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

#if defined(USE_SSE) && defined(USE_SIMD) && (defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86))

//------------------------------------------------------
// FMA version of sger_row_noalpha4x4
//------------------------------------------------------
static void sger_row_noalpha4x4_fma(CBLAS_INDEX m, CBLAS_INDEX n, float* x, CBLAS_INDEX incx, float* y, CBLAS_INDEX incy, float* a, CBLAS_INDEX lda)
{
	float *xr, *yc, *ap;
	CBLAS_INDEX col, row;

	for (row = 0; row + 4 <= m; row += 4)
	{
		xr = &X(row);
		yc = y;
		ap = &A(0, row);

		for (col = 0; col + 4 <= n; col += 4)
		{
			AddProd4x4_SIMD_fma(xr, yc, ap, lda);
			yc += 4;
			ap += 4;
		}

		// handle leftover cols handling each of the 4 rows in this block
		for (CBLAS_INDEX i = 0; i < 4; i++)
		{
			switch (n - col)
			{
			case 3: AddProd(*xr, Y(col + 2), &A(col + 2, row + i));
			case 2: AddProd(*xr, Y(col + 1), &A(col + 1, row + i));
			case 1: AddProd(*xr, Y(col), &A(col, row + i));
			case 0:;	// do nothing!
			}

			xr = &X(row + i);
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

#endif

//------------------------------------------------------
//
//------------------------------------------------------
static void sger_row_noalpha(CBLAS_INDEX m, CBLAS_INDEX n, float *x, CBLAS_INDEX incx, float *y, CBLAS_INDEX incy, float *a, CBLAS_INDEX lda)
{
	register float xr;
	float *yc, *ap;
	CBLAS_INDEX col;

	for (CBLAS_INDEX row = 0; row < m; row++)
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
	for (CBLAS_INDEX row = 0; row < m; row++)
	{
		for (CBLAS_INDEX col = 0; col < n; col++)
		{
			a[row * n + col] += x[row] * y[col];
		}
	}
}

//------------------------------------------------------
// single-precision rank-1 update
//------------------------------------------------------
void cblas_sger(CBLAS_LAYOUT layout, CBLAS_INDEX m, CBLAS_INDEX n, float alpha, float* x, CBLAS_INDEX incx, float* y, CBLAS_INDEX incy, float* a, CBLAS_INDEX lda)
{
    CBLAS_VALIDATE_GER(layout, m, n, x, incx, y, incy, a, lda, );

	// fast reject case
	if (m == 0 ||  n == 0 || alpha == 0.0f)
		return;

	// TODO - handle incx == 1 special case
    if (layout == CblasRowMajor)
    {
		if (alpha == 1.0f)
		{
#if defined(USE_SSE) && defined(USE_SIMD) && (defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86))
			// Use FMA version if CPU supports it
			if (cpu_get_features() & CPU_x64_FMA3)
			{
				sger_row_noalpha4x4_fma(m, n, x, incx, y, incy, a, lda);
			}
			else
#endif
			{
				sger_row_noalpha4x4(m, n,  x, incx, y, incy, a, lda);
			}
			// sger_row_noalpha8x4(m, n,  x, incx, y, incy, a, lda);
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
    CBLAS_VALIDATE_GER(layout, m, n, x, incx, y, incy, a, lda, );
	// fast reject case
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
