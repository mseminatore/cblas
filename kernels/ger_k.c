//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"
#include "cblas_simd.h"

// helpful macros
#define X(i) x[(i) * incx]
#define Y(i) y[(i) * incy]
#define A(col, row) a[(row) * lda + (col)]

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)

//------------------------------------------------------
// compute 4 cols x 4 rows product (SSE, non-FMA)
//------------------------------------------------------
static void AddProd4x4_SSE(float* x, float* y, float* a, CBLAS_INDEX lda)
{
    __m128 x0, x1, x2, x3, y0, a0, a1, a2, a3;

    x0 = _mm_load_ps1(x);
    x1 = _mm_load_ps1(x + 1);
    x2 = _mm_load_ps1(x + 2);
    x3 = _mm_load_ps1(x + 3);
    
    y0 = _mm_loadu_ps(y);
    
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
}

//------------------------------------------------------
// compute 2 cols x 2 rows product (SSE, non-FMA, double)
//------------------------------------------------------
static void AddProd2x2_SSE_d(double* x, double* y, double* a, CBLAS_INDEX lda)
{
    __m128d x0, x1, y0, a0, a1;

    x0 = _mm_load_pd1(x);
    x1 = _mm_load_pd1(x + 1);
    
    y0 = _mm_loadu_pd(y);
    
    a0 = _mm_loadu_pd(a);
    a1 = _mm_loadu_pd(a + lda);

    // compute 2x2 product (non-FMA)
    a0 = _mm_add_pd(a0, _mm_mul_pd(x0, y0));
    a1 = _mm_add_pd(a1, _mm_mul_pd(x1, y0));

    // store results
    _mm_storeu_pd(a, a0);
    _mm_storeu_pd(a + lda, a1);
}

//------------------------------------------------------
// Single-precision GER kernel (SSE, non-FMA)
//------------------------------------------------------
void sger_k_sse(cblas_args_t* args)
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

    if (alpha == 1.0f)
    {
        float *xr, *yc, *ap;
        CBLAS_INDEX col, row;

#if defined(CBLAS_PREFETCH)
        const CBLAS_INDEX prefetch_distance = 64;
#endif

        for (row = 0; row + 4 <= m; row += 4)
        {
            xr = &X(row);
            yc = y;
            ap = &A(0, row);

            for (col = 0; col + 4 <= n; col += 4)
            {
#if defined(CBLAS_PREFETCH)
                if (col + prefetch_distance < n) {
                    CBLAS_PREFETCH(ap + prefetch_distance, 1, 3);
                    CBLAS_PREFETCH(ap + lda + prefetch_distance, 1, 3);
                    CBLAS_PREFETCH(ap + 2*lda + prefetch_distance, 1, 3);
                    CBLAS_PREFETCH(ap + 3*lda + prefetch_distance, 1, 3);
                    CBLAS_PREFETCH(yc + prefetch_distance, 0, 3);
                }
#endif
                AddProd4x4_SSE(xr, yc, ap, lda);
                yc += 4;
                ap += 4;
            }

            // handle leftover cols
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
                case 0:;
                }
                xr++;
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
        case 0:;
        }
    }
    else
    {
        // Generic path for non-unit alpha with cache blocking
        if (m > 2 * GER_BLOCK_SIZE)
        {
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
// Double-precision GER kernel (SSE, non-FMA)
//------------------------------------------------------
void dger_k_sse(cblas_args_t* args)
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

    if (alpha == 1.0)
    {
        double *xr, *yc, *ap;
        CBLAS_INDEX col, row;

#if defined(CBLAS_PREFETCH)
        const CBLAS_INDEX prefetch_distance = 64;
#endif

        for (row = 0; row + 2 <= m; row += 2)
        {
            xr = &X(row);
            yc = y;
            ap = &A(0, row);

            for (col = 0; col + 2 <= n; col += 2)
            {
#if defined(CBLAS_PREFETCH)
                if (col + prefetch_distance < n) {
                    CBLAS_PREFETCH(ap + prefetch_distance, 1, 3);
                    CBLAS_PREFETCH(ap + lda + prefetch_distance, 1, 3);
                    CBLAS_PREFETCH(yc + prefetch_distance, 0, 3);
                }
#endif
                AddProd2x2_SSE_d(xr, yc, ap, lda);
                yc += 2;
                ap += 2;
            }

            // handle leftover cols
            for (CBLAS_INDEX i = 0; i < 2; i++)
            {
                if (n - col == 1)
                {
                    AddProd(*xr, Y(col), &A(col, row + i));
                }
                xr = &X(row + i + 1);
            }
        }

        // handle leftover rows
        if (m - row >= 1)
        {
            for (col = 0; col < n; col++)
                AddProd(X(row), Y(col), &A(col, row));
        }
    }
    else
    {
        // Generic path for non-unit alpha
        if (m > 2 * GER_BLOCK_SIZE)
        {
            for (CBLAS_INDEX i = 0; i < m; i += GER_BLOCK_SIZE)
            {
                CBLAS_INDEX ib = (i + GER_BLOCK_SIZE < m) ? GER_BLOCK_SIZE : (m - i);
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
// compute 4 cols x 4 rows product (AVX 256-bit, single)
//------------------------------------------------------
static void AddProd4x8_AVX(float* x, float* y, float* a, CBLAS_INDEX lda)
{
    __m256 x0, x1, x2, x3, y0, a0, a1, a2, a3;

    x0 = _mm256_broadcast_ss(x);
    x1 = _mm256_broadcast_ss(x + 1);
    x2 = _mm256_broadcast_ss(x + 2);
    x3 = _mm256_broadcast_ss(x + 3);
    
    y0 = _mm256_loadu_ps(y);
    
    a0 = _mm256_loadu_ps(a);
    a1 = _mm256_loadu_ps(a + lda);
    a2 = _mm256_loadu_ps(a + 2 * lda);
    a3 = _mm256_loadu_ps(a + 3 * lda);

    // compute 4x8 product (non-FMA)
    a0 = _mm256_add_ps(a0, _mm256_mul_ps(x0, y0));
    a1 = _mm256_add_ps(a1, _mm256_mul_ps(x1, y0));
    a2 = _mm256_add_ps(a2, _mm256_mul_ps(x2, y0));
    a3 = _mm256_add_ps(a3, _mm256_mul_ps(x3, y0));

    // store results
    _mm256_storeu_ps(a, a0);
    _mm256_storeu_ps(a + lda, a1);
    _mm256_storeu_ps(a + 2 * lda, a2);
    _mm256_storeu_ps(a + 3 * lda, a3);
}

//------------------------------------------------------
// compute 2 cols x 2 rows product (AVX 256-bit, double)
//------------------------------------------------------
static void AddProd2x4_AVX_d(double* x, double* y, double* a, CBLAS_INDEX lda)
{
    __m256d x0, x1, y0, a0, a1;

    x0 = _mm256_broadcast_sd(x);
    x1 = _mm256_broadcast_sd(x + 1);
    
    y0 = _mm256_loadu_pd(y);
    
    a0 = _mm256_loadu_pd(a);
    a1 = _mm256_loadu_pd(a + lda);

    a0 = _mm256_add_pd(a0, _mm256_mul_pd(x0, y0));
    a1 = _mm256_add_pd(a1, _mm256_mul_pd(x1, y0));

    _mm256_storeu_pd(a, a0);
    _mm256_storeu_pd(a + lda, a1);
}

//------------------------------------------------------
// Single-precision GER kernel (AVX 256-bit, non-FMA)
//------------------------------------------------------
void sger_k_avx(cblas_args_t* args)
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

        for (row = 0; row + 4 <= m; row += 4)
        {
            xr = &x[row];
            yc = y;
            ap = &A(0, row);

            for (col = 0; col + 8 <= n; col += 8)
            {
                AddProd4x8_AVX(xr, yc, ap, lda);
                yc += 8;
                ap += 8;
            }

            // Handle remaining columns with SSE
            for (; col + 4 <= n; col += 4)
            {
                AddProd4x4_SSE(xr, yc, ap, lda);
                yc += 4;
                ap += 4;
            }

            // Scalar cleanup
            for (CBLAS_INDEX i = 0; i < 4; i++)
            {
                for (CBLAS_INDEX c = col; c < n; c++)
                {
                    A(c, row + i) += xr[i] * y[c];
                }
            }
        }

        // Handle remaining rows
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
        // Fallback to scalar
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
// Double-precision GER kernel (AVX 256-bit, non-FMA)
//------------------------------------------------------
void dger_k_avx(cblas_args_t* args)
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

        for (row = 0; row + 2 <= m; row += 2)
        {
            xr = &x[row];
            yc = y;
            ap = &A(0, row);

            for (col = 0; col + 4 <= n; col += 4)
            {
                AddProd2x4_AVX_d(xr, yc, ap, lda);
                yc += 4;
                ap += 4;
            }

            // Handle remaining columns with SSE
            for (; col + 2 <= n; col += 2)
            {
                AddProd2x2_SSE_d(xr, yc, ap, lda);
                yc += 2;
                ap += 2;
            }

            // Scalar cleanup
            for (CBLAS_INDEX i = 0; i < 2; i++)
            {
                for (CBLAS_INDEX c = col; c < n; c++)
                {
                    A(c, row + i) += xr[i] * y[c];
                }
            }
        }

        // Handle remaining rows
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
        // Fallback to scalar
        for (CBLAS_INDEX row = 0; row < m; row++)
        {
            for (CBLAS_INDEX col = 0; col < n; col++)
            {
                A(col, row) += alpha * X(row) * Y(col);
            }
        }
    }
}

#endif // x86/x64

//------------------------------------------------------
// Base scalar GER kernel (fallback)
//------------------------------------------------------
void sger_k_base(cblas_args_t* args)
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

    if (alpha == 1.0f)
    {
        for (CBLAS_INDEX row = 0; row < m; row++)
        {
            float xr = X(row);
            for (CBLAS_INDEX col = 0; col < n; col++)
            {
                A(col, row) += xr * Y(col);
            }
        }
    }
    else
    {
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
// Base scalar DGER kernel (fallback)
//------------------------------------------------------
void dger_k_base(cblas_args_t* args)
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

    if (alpha == 1.0)
    {
        for (CBLAS_INDEX row = 0; row < m; row++)
        {
            double xr = X(row);
            for (CBLAS_INDEX col = 0; col < n; col++)
            {
                A(col, row) += xr * Y(col);
            }
        }
    }
    else
    {
        for (CBLAS_INDEX row = 0; row < m; row++)
        {
            for (CBLAS_INDEX col = 0; col < n; col++)
            {
                a[row * lda + col] += alpha * x[row * incx] * y[col * incy];
            }
        }
    }
}
