//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"
#include "cblas_simd.h"

#if defined(__aarch64__) && defined(__ARM_NEON)

// helpful macros
#define X(i) x[(i) * incx]
#define Y(i) y[(i) * incy]
#define A(col, row) a[(row) * lda + (col)]

//------------------------------------------------------
// compute 4 cols x 4 rows product (NEON)
//------------------------------------------------------
static void AddProd4x4_NEON(float* x, float* y, float* a, CBLAS_INDEX lda)
{
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
}

//------------------------------------------------------
// compute 2 cols x 2 rows product (NEON, double)
//------------------------------------------------------
static void AddProd2x2_NEON_d(double* x, double* y, double* a, CBLAS_INDEX lda)
{
    float64x2_t x0, x1, y0, a0, a1;

    x0 = vld1q_dup_f64(x);
    x1 = vld1q_dup_f64(x + 1);

    y0 = vld1q_f64(y);

    a0 = vld1q_f64(a);
    a1 = vld1q_f64(a + lda);

#ifdef __ARM_FEATURE_FMA
    // compute 2x2 product using FMA
    a0 = vfmaq_f64(a0, x0, y0);
    a1 = vfmaq_f64(a1, x1, y0);
#else
    // compute 2x2 product using MUL and ADD
    a0 = vaddq_f64(a0, vmulq_f64(x0, y0));
    a1 = vaddq_f64(a1, vmulq_f64(x1, y0));
#endif

    // store 2x2 doubles
    vst1q_f64(a, a0);
    vst1q_f64(a + lda, a1);
}

//------------------------------------------------------
// Single-precision GER kernel (NEON)
//------------------------------------------------------
void sger_k_neon(cblas_args_t* args)
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
                AddProd4x4_NEON(xr, yc, ap, lda);
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
// Double-precision GER kernel (NEON)
//------------------------------------------------------
void dger_k_neon(cblas_args_t* args)
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

        for (row = 0; row + 2 <= m; row += 2)
        {
            xr = &X(row);
            yc = y;
            ap = &A(0, row);

            for (col = 0; col + 2 <= n; col += 2)
            {
                AddProd2x2_NEON_d(xr, yc, ap, lda);
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

#endif // __aarch64__ && __ARM_NEON
