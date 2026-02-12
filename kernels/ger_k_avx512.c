//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------
// GER kernel - AVX-512 implementation
// A = alpha * x * y' + A (rank-1 update)
// Uses 512-bit SIMD for 16-column processing (single)
// and 8-column processing (double)
//------------------------------------------------------

#include "cblas.h"
#include "cblas_simd.h"

#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX512F__)

// helpful macros
#define X(i) x[(i) * incx]
#define Y(i) y[(i) * incy]
#define A(col, row) a[(row) * lda + (col)]

//------------------------------------------------------
// compute 16 cols x 4 rows product (AVX-512, single)
// Processes 4 rows at a time, 16 columns per iteration
//------------------------------------------------------
static void AddProd16x4_AVX512(float* x, float* y, float* a, CBLAS_INDEX lda)
{
    __m512 x0 = _mm512_set1_ps(x[0]);
    __m512 x1 = _mm512_set1_ps(x[1]);
    __m512 x2 = _mm512_set1_ps(x[2]);
    __m512 x3 = _mm512_set1_ps(x[3]);
    
    __m512 y0 = _mm512_loadu_ps(y);
    
    __m512 a0 = _mm512_loadu_ps(a);
    __m512 a1 = _mm512_loadu_ps(a + lda);
    __m512 a2 = _mm512_loadu_ps(a + 2 * lda);
    __m512 a3 = _mm512_loadu_ps(a + 3 * lda);

    // compute 16x4 product using FMA
    a0 = _mm512_fmadd_ps(x0, y0, a0);
    a1 = _mm512_fmadd_ps(x1, y0, a1);
    a2 = _mm512_fmadd_ps(x2, y0, a2);
    a3 = _mm512_fmadd_ps(x3, y0, a3);

    // store results
    _mm512_storeu_ps(a, a0);
    _mm512_storeu_ps(a + lda, a1);
    _mm512_storeu_ps(a + 2 * lda, a2);
    _mm512_storeu_ps(a + 3 * lda, a3);
}

//------------------------------------------------------
// compute 8 cols x 4 rows product (AVX-512, double)
// Processes 4 rows at a time, 8 columns per iteration
//------------------------------------------------------
static void AddProd8x4_AVX512_d(double* x, double* y, double* a, CBLAS_INDEX lda)
{
    __m512d x0 = _mm512_set1_pd(x[0]);
    __m512d x1 = _mm512_set1_pd(x[1]);
    __m512d x2 = _mm512_set1_pd(x[2]);
    __m512d x3 = _mm512_set1_pd(x[3]);
    
    __m512d y0 = _mm512_loadu_pd(y);
    
    __m512d a0 = _mm512_loadu_pd(a);
    __m512d a1 = _mm512_loadu_pd(a + lda);
    __m512d a2 = _mm512_loadu_pd(a + 2 * lda);
    __m512d a3 = _mm512_loadu_pd(a + 3 * lda);

    // compute 8x4 product using FMA
    a0 = _mm512_fmadd_pd(x0, y0, a0);
    a1 = _mm512_fmadd_pd(x1, y0, a1);
    a2 = _mm512_fmadd_pd(x2, y0, a2);
    a3 = _mm512_fmadd_pd(x3, y0, a3);

    // store results
    _mm512_storeu_pd(a, a0);
    _mm512_storeu_pd(a + lda, a1);
    _mm512_storeu_pd(a + 2 * lda, a2);
    _mm512_storeu_pd(a + 3 * lda, a3);
}

// Scalar helper for cleanup
static inline void AddProd_s(float x, float y, float* a)
{
    *a += x * y;
}

static inline void AddProd_d(double x, double y, double* a)
{
    *a += x * y;
}

//------------------------------------------------------
// Single-precision GER kernel (AVX-512)
//------------------------------------------------------
void sger_k_avx512(cblas_args_t* args)
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

    if (alpha == 1.0f && incx == 1 && incy == 1)
    {
        float *xr, *yc, *ap;
        CBLAS_INDEX col, row;

        // Main loop: 4 rows x 16 cols
        for (row = 0; row + 4 <= m; row += 4)
        {
            xr = x + row;
            yc = y;
            ap = &A(0, row);

            for (col = 0; col + 16 <= n; col += 16)
            {
                AddProd16x4_AVX512(xr, yc, ap, lda);
                yc += 16;
                ap += 16;
            }

            // Handle leftover columns with AVX (8 cols)
            if (col + 8 <= n)
            {
                __m256 x0 = _mm256_set1_ps(xr[0]);
                __m256 x1 = _mm256_set1_ps(xr[1]);
                __m256 x2 = _mm256_set1_ps(xr[2]);
                __m256 x3 = _mm256_set1_ps(xr[3]);
                __m256 yv = _mm256_loadu_ps(yc);
                
                __m256 a0 = _mm256_loadu_ps(ap);
                __m256 a1 = _mm256_loadu_ps(ap + lda);
                __m256 a2 = _mm256_loadu_ps(ap + 2 * lda);
                __m256 a3 = _mm256_loadu_ps(ap + 3 * lda);
                
                a0 = _mm256_fmadd_ps(x0, yv, a0);
                a1 = _mm256_fmadd_ps(x1, yv, a1);
                a2 = _mm256_fmadd_ps(x2, yv, a2);
                a3 = _mm256_fmadd_ps(x3, yv, a3);
                
                _mm256_storeu_ps(ap, a0);
                _mm256_storeu_ps(ap + lda, a1);
                _mm256_storeu_ps(ap + 2 * lda, a2);
                _mm256_storeu_ps(ap + 3 * lda, a3);
                
                yc += 8;
                ap += 8;
                col += 8;
            }

            // Scalar cleanup for remaining columns
            for (; col < n; col++)
            {
                for (CBLAS_INDEX i = 0; i < 4; i++)
                {
                    AddProd_s(xr[i], yc[0], ap + i * lda);
                }
                yc++;
                ap++;
            }
        }

        // Handle leftover rows (scalar)
        for (; row < m; row++)
        {
            for (col = 0; col < n; col++)
            {
                A(col, row) += x[row] * y[col];
            }
        }
    }
    else
    {
        // Generic path for non-unit alpha or strided access
        for (CBLAS_INDEX row = 0; row < m; row++)
        {
            for (CBLAS_INDEX col = 0; col < n; col++)
            {
                A(col, row) += alpha * X(row) * Y(col);
            }
        }
    }
}

//------------------------------------------------------
// Double-precision GER kernel (AVX-512)
//------------------------------------------------------
void dger_k_avx512(cblas_args_t* args)
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

    if (alpha == 1.0 && incx == 1 && incy == 1)
    {
        double *xr, *yc, *ap;
        CBLAS_INDEX col, row;

        // Main loop: 4 rows x 8 cols
        for (row = 0; row + 4 <= m; row += 4)
        {
            xr = x + row;
            yc = y;
            ap = &A(0, row);

            for (col = 0; col + 8 <= n; col += 8)
            {
                AddProd8x4_AVX512_d(xr, yc, ap, lda);
                yc += 8;
                ap += 8;
            }

            // Handle leftover columns with AVX (4 cols)
            if (col + 4 <= n)
            {
                __m256d x0 = _mm256_set1_pd(xr[0]);
                __m256d x1 = _mm256_set1_pd(xr[1]);
                __m256d x2 = _mm256_set1_pd(xr[2]);
                __m256d x3 = _mm256_set1_pd(xr[3]);
                __m256d yv = _mm256_loadu_pd(yc);
                
                __m256d a0 = _mm256_loadu_pd(ap);
                __m256d a1 = _mm256_loadu_pd(ap + lda);
                __m256d a2 = _mm256_loadu_pd(ap + 2 * lda);
                __m256d a3 = _mm256_loadu_pd(ap + 3 * lda);
                
                a0 = _mm256_fmadd_pd(x0, yv, a0);
                a1 = _mm256_fmadd_pd(x1, yv, a1);
                a2 = _mm256_fmadd_pd(x2, yv, a2);
                a3 = _mm256_fmadd_pd(x3, yv, a3);
                
                _mm256_storeu_pd(ap, a0);
                _mm256_storeu_pd(ap + lda, a1);
                _mm256_storeu_pd(ap + 2 * lda, a2);
                _mm256_storeu_pd(ap + 3 * lda, a3);
                
                yc += 4;
                ap += 4;
                col += 4;
            }

            // Scalar cleanup for remaining columns
            for (; col < n; col++)
            {
                for (CBLAS_INDEX i = 0; i < 4; i++)
                {
                    AddProd_d(xr[i], yc[0], ap + i * lda);
                }
                yc++;
                ap++;
            }
        }

        // Handle leftover rows (scalar)
        for (; row < m; row++)
        {
            for (col = 0; col < n; col++)
            {
                A(col, row) += x[row] * y[col];
            }
        }
    }
    else
    {
        // Generic path for non-unit alpha or strided access
        for (CBLAS_INDEX row = 0; row < m; row++)
        {
            for (CBLAS_INDEX col = 0; col < n; col++)
            {
                a[row * lda + col] += alpha * x[row * incx] * y[col * incy];
            }
        }
    }
}

#else

// Stub implementations for non-AVX512 builds
void sger_k_avx512(cblas_args_t* args) { (void)args; }
void dger_k_avx512(cblas_args_t* args) { (void)args; }

#endif
