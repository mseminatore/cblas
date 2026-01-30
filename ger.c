//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"

#ifdef _WIN32
#	include <malloc.h>
#else
#	include <alloca.h>
#endif

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
CBLAS_UNUSED static void AddProd8x4_AVX(float* x, float* y, float* a, CBLAS_INDEX lda)
{
#if defined(__aarch64__)
	(void)x;
	(void)y;
	(void)a;
	(void)lda;
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
CBLAS_UNUSED static void AddProd8x4_AVX_fma(float* x, float* y, float* a, CBLAS_INDEX lda)
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
CBLAS_UNUSED static void AddProd4x4(float* x, float* y, float* a, CBLAS_INDEX lda)
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
CBLAS_UNUSED static void sger_row_noalpha8x4(CBLAS_INDEX m, CBLAS_INDEX n, float* x, CBLAS_INDEX incx, float* y, CBLAS_INDEX incy, float* a, CBLAS_INDEX lda)
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
				CBLAS_FALLTHROUGH;
			case 6: AddProd(*xr, Y(col + 5), &A(col + 5, row + i));
				CBLAS_FALLTHROUGH;
			case 5: AddProd(*xr, Y(col + 4), &A(col + 4, row + i));
				CBLAS_FALLTHROUGH;
			case 4: AddProd(*xr, Y(col + 3), &A(col + 3, row + i));
				CBLAS_FALLTHROUGH;
			case 3: AddProd(*xr, Y(col + 2), &A(col + 2, row + i));
				CBLAS_FALLTHROUGH;
			case 2: AddProd(*xr, Y(col + 1), &A(col + 1, row + i));
				CBLAS_FALLTHROUGH;
			case 1: AddProd(*xr, Y(col), &A(col, row + i));
				CBLAS_FALLTHROUGH;
			case 0:;	// do nothing!
			}

			xr = &X(row + i);
		}
	}

	// handle leftover rows
	switch (m - row)
	{
	case 3: for (col = 0; col < n; col++) AddProd(X(row + 2), Y(col), &A(col, row + 2));
		CBLAS_FALLTHROUGH;
	case 2: for (col = 0; col < n; col++) AddProd(X(row + 1), Y(col), &A(col, row + 1));
		CBLAS_FALLTHROUGH;
	case 1: for (col = 0; col < n; col++) AddProd(X(row), Y(col), &A(col, row));
		CBLAS_FALLTHROUGH;
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
				CBLAS_FALLTHROUGH;
			case 2: AddProd(*xr, Y(col + 1), &A(col + 1, row + i));
				CBLAS_FALLTHROUGH;
			case 1: AddProd(*xr, Y(col), &A(col, row + i));
				CBLAS_FALLTHROUGH;
			case 0:;	// do nothing!
			}

			xr = &X(row + i);
		}
	}

	// handle leftover rows
	switch (m - row)
	{
	case 3: for (col = 0; col < n; col++) AddProd(X(row + 2), Y(col), &A(col, row + 2));
		CBLAS_FALLTHROUGH;
	case 2: for (col = 0; col < n; col++) AddProd(X(row + 1), Y(col), &A(col, row + 1));
		CBLAS_FALLTHROUGH;
	case 1: for (col = 0; col < n; col++) AddProd(X(row), Y(col), &A(col, row));
		CBLAS_FALLTHROUGH;
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
				CBLAS_FALLTHROUGH;
			case 2: AddProd(*xr, Y(col + 1), &A(col + 1, row + i));
				CBLAS_FALLTHROUGH;
			case 1: AddProd(*xr, Y(col), &A(col, row + i));
				CBLAS_FALLTHROUGH;
			case 0:;	// do nothing!
			}

			xr = &X(row + i);
		}
	}

	// handle leftover rows
	switch (m - row)
	{
	case 3: for (col = 0; col < n; col++) AddProd(X(row + 2), Y(col), &A(col, row + 2));
		CBLAS_FALLTHROUGH;
	case 2: for (col = 0; col < n; col++) AddProd(X(row + 1), Y(col), &A(col, row + 1));
		CBLAS_FALLTHROUGH;
	case 1: for (col = 0; col < n; col++) AddProd(X(row), Y(col), &A(col, row));
		CBLAS_FALLTHROUGH;
	case 0:;	// do nothing!
	}
}

#endif

//------------------------------------------------------
//
//------------------------------------------------------
CBLAS_UNUSED static void sger_row_noalpha(CBLAS_INDEX m, CBLAS_INDEX n, float *x, CBLAS_INDEX incx, float *y, CBLAS_INDEX incy, float *a, CBLAS_INDEX lda)
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
CBLAS_UNUSED static void sger_row_noalpha_plain(CBLAS_INDEX m, CBLAS_INDEX n, float *x, CBLAS_INDEX incx, float *y, CBLAS_INDEX incy, float *a, CBLAS_INDEX lda)
{
	(void)incx;
	(void)incy;
	(void)lda;
	for (CBLAS_INDEX row = 0; row < m; row++)
	{
		for (CBLAS_INDEX col = 0; col < n; col++)
		{
			a[row * n + col] += x[row] * y[col];
		}
	}
}

//------------------------------------------------------
// GER kernel for multi-threading - single precision
//------------------------------------------------------
void sger_k(cblas_args_t* args)
{
	float* x = (float*)args->x;
	float* y = (float*)args->y;
	float* a = (float*)args->a;
	CBLAS_INDEX m = args->m;
	CBLAS_INDEX n = args->n;
	CBLAS_INDEX incx = args->incx;
	CBLAS_INDEX incy = args->incy;
	CBLAS_INDEX lda = args->lda;
	float alpha = *(float*)args->alpha;

	// Use optimized path when alpha == 1.0
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
			sger_row_noalpha4x4(m, n, x, incx, y, incy, a, lda);
		}
	}
	else
	{
		// Use cache blocking only for large matrices
		if (m > 2 * GER_BLOCK_SIZE)
		{
			// Generic path with cache blocking for non-unit alpha
			for (CBLAS_INDEX i = 0; i < m; i += GER_BLOCK_SIZE)
			{
				CBLAS_INDEX ib = (i + GER_BLOCK_SIZE < m) ? GER_BLOCK_SIZE : (m - i);
				
				for (CBLAS_INDEX row = i; row < i + ib; row++)
				{
					for (CBLAS_INDEX col = 0; col < n; col++)
					{
						A(col, row) += alpha * X(row) * Y(col);
					}
				}
			}
		}
		else
		{
			// Direct processing for small matrices
			for (CBLAS_INDEX row = 0; row < m; row++)
			{
				for (CBLAS_INDEX col = 0; col < n; col++)
				{
					A(col, row) += alpha * X(row) * Y(col);
				}
			}
		}
	}
}

//------------------------------------------------------
// GER kernel for multi-threading - double precision
//------------------------------------------------------
void dger_k(cblas_args_t* args)
{
	double* x = (double*)args->x;
	double* y = (double*)args->y;
	double* a = (double*)args->a;
	CBLAS_INDEX m = args->m;
	CBLAS_INDEX n = args->n;
	CBLAS_INDEX incx = args->incx;
	CBLAS_INDEX incy = args->incy;
	CBLAS_INDEX lda = args->lda;
	double alpha = *(double*)args->alpha;

	// Use optimized path when alpha == 1.0
	if (alpha == 1.0)
	{
		// Use cache blocking only for large matrices
		if (m > 2 * GER_BLOCK_SIZE)
		{
			// Process rows in cache-friendly blocks
			for (CBLAS_INDEX i = 0; i < m; i += GER_BLOCK_SIZE)
			{
				CBLAS_INDEX ib = (i + GER_BLOCK_SIZE < m) ? GER_BLOCK_SIZE : (m - i);
				
				// Optimized path for unit alpha
				for (CBLAS_INDEX row = i; row < i + ib; row++)
				{
					for (CBLAS_INDEX col = 0; col < n; col++)
					{
						a[row * lda + col] += x[row * incx] * y[col * incy];
					}
				}
			}
		}
		else
		{
			// Direct processing for small matrices
			for (CBLAS_INDEX row = 0; row < m; row++)
			{
				for (CBLAS_INDEX col = 0; col < n; col++)
				{
					a[row * lda + col] += x[row * incx] * y[col * incy];
				}
			}
		}
	}
	else
	{
		// Use cache blocking only for large matrices
		if (m > 2 * GER_BLOCK_SIZE)
		{
			// Process rows in cache-friendly blocks
			for (CBLAS_INDEX i = 0; i < m; i += GER_BLOCK_SIZE)
			{
				CBLAS_INDEX ib = (i + GER_BLOCK_SIZE < m) ? GER_BLOCK_SIZE : (m - i);
				
				// Generic path for non-unit alpha
				for (CBLAS_INDEX row = i; row < i + ib; row++)
				{
					for (CBLAS_INDEX col = 0; col < n; col++)
					{
						a[row * lda + col] += alpha * x[row * incx] * y[col * incy];
					}
				}
			}
		}
		else
		{
			// Direct processing for small matrices
			for (CBLAS_INDEX row = 0; row < m; row++)
			{
				for (CBLAS_INDEX col = 0; col < n; col++)
				{
					a[row * lda + col] += alpha * x[row * incx] * y[col * incy];
				}
			}
		}
	}
}

//------------------------------------------------------
// single-precision rank-1 update
//------------------------------------------------------
void cblas_sger(CBLAS_LAYOUT layout, CBLAS_INDEX m, CBLAS_INDEX n, float alpha, float* x, CBLAS_INDEX incx, float* y, CBLAS_INDEX incy, float* a, CBLAS_INDEX lda)
{
    CBLAS_VALIDATE_GER(layout, m, n, x, incx, y, incy, a, lda, );

    CBLAS_STATS_START();

	// fast reject case
	if (m == 0 ||  n == 0 || alpha == 0.0f)
		return;

#if defined(MT_ENABLED)
    int mt_used = (m * n > CBLAS_MT_GER) ? 1 : 0;
    
    if (mt_used && layout == CblasRowMajor && blas_kernels.sger_k != NULL)
    {
        // Multi-threaded path: partition rows across threads
        CBLAS_INDEX thread_count = CLAMP(cblas_get_num_threads(), 1, MAX_THREADS);
        
        #ifdef _WIN32
            work_queue_t *queue = _malloca(thread_count * sizeof(work_queue_t));
            cblas_args_t *args = _malloca(thread_count * sizeof(cblas_args_t));
        #else
            work_queue_t *queue = alloca(thread_count * sizeof(work_queue_t));
            cblas_args_t *args = alloca(thread_count * sizeof(cblas_args_t));
        #endif
        
        CBLAS_INDEX rows_remaining = m;
        CBLAS_INDEX row_offset = 0;
        
        for (CBLAS_INDEX i = 0; i < thread_count; i++)
        {
            // Compute partition size: distribute rows evenly
            CBLAS_INDEX rows_per_thread = (rows_remaining + thread_count - i - 1) / (thread_count - i);
            
            args[i].m = rows_per_thread;
            args[i].n = n;
            args[i].incx = incx;
            args[i].incy = incy;
            args[i].lda = lda;
            args[i].x = &X(row_offset);
            args[i].y = y;
            args[i].a = &A(0, row_offset);
            args[i].alpha = &alpha;
            
            queue[i].finished = 0;
            queue[i].args = &args[i];
            queue[i].kernel = blas_kernels.sger_k;
            queue[i].next = &queue[i + 1];
            
            row_offset += rows_per_thread;
            rows_remaining -= rows_per_thread;
        }
        
        // mark end of task queue
        queue[thread_count - 1].next = NULL;
        
        // synchronously execute task queue
        cblas_execute(thread_count, queue);
    }
    else
#else
    int mt_used = 0;
#endif
    {
        // Single-threaded fallback path
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
            }
            else
            {
                // Use cache blocking only for large matrices
                if (m > 2 * GER_BLOCK_SIZE)
                {
                    // Process rows in cache-friendly blocks
                    for (CBLAS_INDEX i = 0; i < m; i += GER_BLOCK_SIZE)
                    {
                        CBLAS_INDEX ib = (i + GER_BLOCK_SIZE < m) ? GER_BLOCK_SIZE : (m - i);
                        
                        for (CBLAS_INDEX row = i; row < i + ib; row++)
                        {
                            for (CBLAS_INDEX col = 0; col < n; col++)
                            {
                                a[row * n + col] += alpha * x[row] * y[col];
                            }
                        }
                    }
                }
                else
                {
                    // Direct processing for small matrices
                    for (CBLAS_INDEX row = 0; row < m; row++)
                    {
                        for (CBLAS_INDEX col = 0; col < n; col++)
                        {
                            a[row * n + col] += alpha * x[row] * y[col];
                        }
                    }
                }
            }
        } else
        {
            if (alpha == 1.0f)
            {
                // Use cache blocking only for large matrices
                if (n > 2 * GER_BLOCK_SIZE)
                {
                    // Process columns in cache-friendly blocks
                    for (CBLAS_INDEX j = 0; j < n; j += GER_BLOCK_SIZE)
                    {
                        CBLAS_INDEX jb = (j + GER_BLOCK_SIZE < n) ? GER_BLOCK_SIZE : (n - j);
                        
                        for (CBLAS_INDEX col = j; col < j + jb; col++)
                        {
                            for (CBLAS_INDEX row = 0; row < m; row++)
                            {
                                a[col * m + row] += x[row] * y[col];
                            }
                        }
                    }
                }
                else
                {
                    // Direct processing for small matrices
                    for (CBLAS_INDEX col = 0; col < n; col++)
                    {
                        for (CBLAS_INDEX row = 0; row < m; row++)
                        {
                            a[col * m + row] += x[row] * y[col];
                        }
                    }
                }
            }
            else
            {
                // Use cache blocking only for large matrices
                if (n > 2 * GER_BLOCK_SIZE)
                {
                    // Process columns in cache-friendly blocks
                    for (CBLAS_INDEX j = 0; j < n; j += GER_BLOCK_SIZE)
                    {
                        CBLAS_INDEX jb = (j + GER_BLOCK_SIZE < n) ? GER_BLOCK_SIZE : (n - j);
                        
                        for (CBLAS_INDEX col = j; col < j + jb; col++)
                        {
                            for (CBLAS_INDEX row = 0; row < m; row++)
                            {
                                a[col * m + row] += alpha * x[row] * y[col];
                            }
                        }
                    }
                }
                else
                {
                    // Direct processing for small matrices
                    for (CBLAS_INDEX col = 0; col < n; col++)
                    {
                        for (CBLAS_INDEX row = 0; row < m; row++)
                        {
                            a[col * m + row] += alpha * x[row] * y[col];
                        }
                    }
                }
            }
        }
    }

    CBLAS_STATS_END("sger", m * n, mt_used);
}

//------------------------------------------------------
// double-precision rank-1 update
//------------------------------------------------------
void cblas_dger(CBLAS_LAYOUT layout, CBLAS_INDEX m, CBLAS_INDEX n, double alpha, double *x, CBLAS_INDEX incx, double *y, CBLAS_INDEX incy, double *a, CBLAS_INDEX lda)
{
    CBLAS_VALIDATE_GER(layout, m, n, x, incx, y, incy, a, lda, );

    CBLAS_STATS_START();

	// fast reject case
	if (m == 0 ||  n == 0 || alpha == 0.0)
		return;

#if defined(MT_ENABLED)
    int mt_used = (m * n > CBLAS_MT_GER) ? 1 : 0;
    
    if (mt_used && layout == CblasRowMajor && blas_kernels.dger_k != NULL)
    {
        // Multi-threaded path: partition rows across threads
        CBLAS_INDEX thread_count = CLAMP(cblas_get_num_threads(), 1, MAX_THREADS);
        
        #ifdef _WIN32
            work_queue_t *queue = _malloca(thread_count * sizeof(work_queue_t));
            cblas_args_t *args = _malloca(thread_count * sizeof(cblas_args_t));
        #else
            work_queue_t *queue = alloca(thread_count * sizeof(work_queue_t));
            cblas_args_t *args = alloca(thread_count * sizeof(cblas_args_t));
        #endif
        
        CBLAS_INDEX rows_remaining = m;
        CBLAS_INDEX row_offset = 0;
        
        for (CBLAS_INDEX i = 0; i < thread_count; i++)
        {
            // Compute partition size: distribute rows evenly
            CBLAS_INDEX rows_per_thread = (rows_remaining + thread_count - i - 1) / (thread_count - i);
            
            args[i].m = rows_per_thread;
            args[i].n = n;
            args[i].incx = incx;
            args[i].incy = incy;
            args[i].lda = lda;
            args[i].x = &x[row_offset * incx];
            args[i].y = y;
            args[i].a = &a[row_offset * lda];
            args[i].alpha = &alpha;
            
            queue[i].finished = 0;
            queue[i].args = &args[i];
            queue[i].kernel = blas_kernels.dger_k;
            queue[i].next = &queue[i + 1];
            
            row_offset += rows_per_thread;
            rows_remaining -= rows_per_thread;
        }
        
        // mark end of task queue
        queue[thread_count - 1].next = NULL;
        
        // synchronously execute task queue
        cblas_execute(thread_count, queue);
    }
    else
#else
    int mt_used = 0;
#endif
    {
        // Single-threaded fallback path
        if (layout == CblasRowMajor)
        {
            if (alpha == 1.0)
            {
                // Use cache blocking only for large matrices
                if (m > 2 * GER_BLOCK_SIZE)
                {
                    // Process rows in cache-friendly blocks
                    for (CBLAS_INDEX i = 0; i < m; i += GER_BLOCK_SIZE)
                    {
                        CBLAS_INDEX ib = (i + GER_BLOCK_SIZE < m) ? GER_BLOCK_SIZE : (m - i);
                        
                        for (CBLAS_INDEX row = i; row < i + ib; row++)
                        {
                            for (CBLAS_INDEX col = 0; col < n; col++)
                            {
                                a[row * n + col] += x[row] * y[col];
                            }
                        }
                    }
                }
                else
                {
                    // Direct processing for small matrices
                    for (CBLAS_INDEX row = 0; row < m; row++)
                    {
                        for (CBLAS_INDEX col = 0; col < n; col++)
                        {
                            a[row * n + col] += x[row] * y[col];
                        }
                    }
                }
            }
            else
            {
                // Use cache blocking only for large matrices
                if (m > 2 * GER_BLOCK_SIZE)
                {
                    // Process rows in cache-friendly blocks
                    for (CBLAS_INDEX i = 0; i < m; i += GER_BLOCK_SIZE)
                    {
                        CBLAS_INDEX ib = (i + GER_BLOCK_SIZE < m) ? GER_BLOCK_SIZE : (m - i);
                        
                        for (CBLAS_INDEX row = i; row < i + ib; row++)
                        {
                            for (CBLAS_INDEX col = 0; col < n; col++)
                            {
                                a[row * n + col] += alpha * x[row] * y[col];
                            }
                        }
                    }
                }
                else
                {
                    // Direct processing for small matrices
                    for (CBLAS_INDEX row = 0; row < m; row++)
                    {
                        for (CBLAS_INDEX col = 0; col < n; col++)
                        {
                            a[row * n + col] += alpha * x[row] * y[col];
                        }
                    }
                }
            }
        } else
        {
            if (alpha == 1.0)
            {
                // Use cache blocking only for large matrices
                if (n > 2 * GER_BLOCK_SIZE)
                {
                    // Process columns in cache-friendly blocks
                    for (CBLAS_INDEX j = 0; j < n; j += GER_BLOCK_SIZE)
                    {
                        CBLAS_INDEX jb = (j + GER_BLOCK_SIZE < n) ? GER_BLOCK_SIZE : (n - j);
                        
                        for (CBLAS_INDEX col = j; col < j + jb; col++)
                        {
                            for (CBLAS_INDEX row = 0; row < m; row++)
                            {
                                a[col * m + row] += x[row] * y[col];
                            }
                        }
                    }
                }
                else
                {
                    // Direct processing for small matrices
                    for (CBLAS_INDEX col = 0; col < n; col++)
                    {
                        for (CBLAS_INDEX row = 0; row < m; row++)
                        {
                            a[col * m + row] += x[row] * y[col];
                        }
                    }
                }
            }
            else
            {
                // Use cache blocking only for large matrices
                if (n > 2 * GER_BLOCK_SIZE)
                {
                    // Process columns in cache-friendly blocks
                    for (CBLAS_INDEX j = 0; j < n; j += GER_BLOCK_SIZE)
                    {
                        CBLAS_INDEX jb = (j + GER_BLOCK_SIZE < n) ? GER_BLOCK_SIZE : (n - j);
                        
                        for (CBLAS_INDEX col = j; col < j + jb; col++)
                        {
                            for (CBLAS_INDEX row = 0; row < m; row++)
                            {
                                a[col * m + row] += alpha * x[row] * y[col];
                            }
                        }
                    }
                }
                else
                {
                    // Direct processing for small matrices
                    for (CBLAS_INDEX col = 0; col < n; col++)
                    {
                        for (CBLAS_INDEX row = 0; row < m; row++)
                        {
                            a[col * m + row] += alpha * x[row] * y[col];
                        }
                    }
                }
            }
        }
    }

    CBLAS_STATS_END("dger", m * n, mt_used);
}
