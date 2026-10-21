//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"
#include "cblas_simd.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)

//------------------------------------------------------
// 128-bit GEMV kernels for x86-64 CPUs without AVX (pre-Sandy Bridge, and
// Atom-class cores such as Goldmont/Tremont). Uses SSE2 + SSE3 (hadd).
//
// Kernels operate on a row-major m x n matrix A (leading dimension lda):
//   [sd]gemv_k_sse    y[0:m] = alpha * A   * x + beta * y   (dot form)
//   [sd]gemv_t_k_sse  y[0:n] = alpha * A^T * x + beta * y   (axpy form)
//
// Both process 4 rows per pass so each x load (dot form) or y load/store
// (axpy form) is shared by 4 rows. See gemv_k_x86.h for details.
//------------------------------------------------------

// number of y elements per axpy-form block (fits comfortably in L1)
#define GEMV_T_BLOCK_S  2048
#define GEMV_T_BLOCK_D  1024

//------------------------------------------------------
// Single-precision GEMV kernel, dot form (SSE)
//------------------------------------------------------
void sgemv_k_sse(cblas_args_t* args)
{
    float* a = (float*)args->a;
    float* x = (float*)args->x;
    float* y = (float*)args->y;
    CBLAS_INDEX m = args->m;
    CBLAS_INDEX n = args->n;
    CBLAS_INDEX lda = args->lda;
    CBLAS_INDEX incx = args->incx;
    CBLAS_INDEX incy = args->incy;
    float alpha = *(float*)args->alpha;
    float beta = *(float*)args->beta;

    if (incx != 1)
    {
        // General strided x - scalar
        for (CBLAS_INDEX row = 0; row < m; row++)
        {
            float sum = 0.0f;
            for (CBLAS_INDEX col = 0; col < n; col++)
                sum += a[row * lda + col] * x[col * incx];

            float* yp = &y[row * incy];
            *yp = (beta == 0.0f ? 0.0f : beta * *yp) + alpha * sum;
        }
        return;
    }

    CBLAS_INDEX n8 = n & ~(CBLAS_INDEX)7;
    CBLAS_INDEX n4 = n & ~(CBLAS_INDEX)3;
    CBLAS_INDEX row = 0;

    // 4 rows per pass: each x vector load is shared by 4 rows
    for (; row + 4 <= m; row += 4)
    {
        const float* a0 = a + row * lda;
        const float* a1 = a0 + lda;
        const float* a2 = a1 + lda;
        const float* a3 = a2 + lda;

        __m128 s00 = _mm_setzero_ps(), s01 = _mm_setzero_ps();
        __m128 s10 = _mm_setzero_ps(), s11 = _mm_setzero_ps();
        __m128 s20 = _mm_setzero_ps(), s21 = _mm_setzero_ps();
        __m128 s30 = _mm_setzero_ps(), s31 = _mm_setzero_ps();

        CBLAS_INDEX col = 0;
        for (; col < n8; col += 8)
        {
            __m128 x0 = _mm_loadu_ps(x + col);
            __m128 x1 = _mm_loadu_ps(x + col + 4);

            s00 = _mm_add_ps(s00, _mm_mul_ps(_mm_loadu_ps(a0 + col), x0));
            s01 = _mm_add_ps(s01, _mm_mul_ps(_mm_loadu_ps(a0 + col + 4), x1));
            s10 = _mm_add_ps(s10, _mm_mul_ps(_mm_loadu_ps(a1 + col), x0));
            s11 = _mm_add_ps(s11, _mm_mul_ps(_mm_loadu_ps(a1 + col + 4), x1));
            s20 = _mm_add_ps(s20, _mm_mul_ps(_mm_loadu_ps(a2 + col), x0));
            s21 = _mm_add_ps(s21, _mm_mul_ps(_mm_loadu_ps(a2 + col + 4), x1));
            s30 = _mm_add_ps(s30, _mm_mul_ps(_mm_loadu_ps(a3 + col), x0));
            s31 = _mm_add_ps(s31, _mm_mul_ps(_mm_loadu_ps(a3 + col + 4), x1));
        }

        for (; col < n4; col += 4)
        {
            __m128 x0 = _mm_loadu_ps(x + col);

            s00 = _mm_add_ps(s00, _mm_mul_ps(_mm_loadu_ps(a0 + col), x0));
            s10 = _mm_add_ps(s10, _mm_mul_ps(_mm_loadu_ps(a1 + col), x0));
            s20 = _mm_add_ps(s20, _mm_mul_ps(_mm_loadu_ps(a2 + col), x0));
            s30 = _mm_add_ps(s30, _mm_mul_ps(_mm_loadu_ps(a3 + col), x0));
        }

        // reduce 4 accumulators to one vector of 4 row sums
        __m128 h01 = _mm_hadd_ps(_mm_add_ps(s00, s01), _mm_add_ps(s10, s11));
        __m128 h23 = _mm_hadd_ps(_mm_add_ps(s20, s21), _mm_add_ps(s30, s31));
        float sum[4];
        _mm_storeu_ps(sum, _mm_hadd_ps(h01, h23));

        for (; col < n; col++)
        {
            float xv = x[col];
            sum[0] += a0[col] * xv;
            sum[1] += a1[col] * xv;
            sum[2] += a2[col] * xv;
            sum[3] += a3[col] * xv;
        }

        for (int r = 0; r < 4; r++)
        {
            float* yp = &y[(row + r) * incy];
            *yp = (beta == 0.0f ? 0.0f : beta * *yp) + alpha * sum[r];
        }
    }

    // remaining rows, one at a time
    for (; row < m; row++)
    {
        const float* a0 = a + row * lda;
        __m128 s0 = _mm_setzero_ps(), s1 = _mm_setzero_ps();

        CBLAS_INDEX col = 0;
        for (; col < n8; col += 8)
        {
            s0 = _mm_add_ps(s0, _mm_mul_ps(_mm_loadu_ps(a0 + col), _mm_loadu_ps(x + col)));
            s1 = _mm_add_ps(s1, _mm_mul_ps(_mm_loadu_ps(a0 + col + 4), _mm_loadu_ps(x + col + 4)));
        }
        for (; col < n4; col += 4)
            s0 = _mm_add_ps(s0, _mm_mul_ps(_mm_loadu_ps(a0 + col), _mm_loadu_ps(x + col)));

        __m128 h = _mm_add_ps(s0, s1);
        h = _mm_hadd_ps(h, h);
        h = _mm_hadd_ps(h, h);
        float sum = _mm_cvtss_f32(h);

        for (; col < n; col++)
            sum += a0[col] * x[col];

        float* yp = &y[row * incy];
        *yp = (beta == 0.0f ? 0.0f : beta * *yp) + alpha * sum;
    }
}

//------------------------------------------------------
// Single-precision GEMV kernel, axpy form (SSE)
//------------------------------------------------------
void sgemv_t_k_sse(cblas_args_t* args)
{
    float* a = (float*)args->a;
    float* x = (float*)args->x;
    float* y = (float*)args->y;
    CBLAS_INDEX m = args->m;
    CBLAS_INDEX n = args->n;
    CBLAS_INDEX lda = args->lda;
    CBLAS_INDEX incx = args->incx;
    CBLAS_INDEX incy = args->incy;
    float alpha = *(float*)args->alpha;
    float beta = *(float*)args->beta;

    // y = beta * y; beta == 0 must not read y (it may hold NaN/Inf)
    if (beta != 1.0f)
    {
        for (CBLAS_INDEX col = 0; col < n; col++)
            y[col * incy] = (beta == 0.0f) ? 0.0f : beta * y[col * incy];
    }

    if (alpha == 0.0f)
        return;

    if (incy != 1)
    {
        // General strided y - scalar, still walking A row by row
        for (CBLAS_INDEX row = 0; row < m; row++)
        {
            float t = alpha * x[row * incx];
            const float* ar = a + row * lda;
            for (CBLAS_INDEX col = 0; col < n; col++)
                y[col * incy] += t * ar[col];
        }
        return;
    }

    for (CBLAS_INDEX jb = 0; jb < n; jb += GEMV_T_BLOCK_S)
    {
        CBLAS_INDEX nb = MIN(GEMV_T_BLOCK_S, n - jb);
        CBLAS_INDEX nb8 = nb & ~(CBLAS_INDEX)7;
        CBLAS_INDEX nb4 = nb & ~(CBLAS_INDEX)3;
        float* yb = y + jb;
        CBLAS_INDEX row = 0;

        // 4 rows per pass: each y load/store is shared by 4 rows
        for (; row + 4 <= m; row += 4)
        {
            const float* a0 = a + row * lda + jb;
            const float* a1 = a0 + lda;
            const float* a2 = a1 + lda;
            const float* a3 = a2 + lda;

            float t0s = alpha * x[(row + 0) * incx];
            float t1s = alpha * x[(row + 1) * incx];
            float t2s = alpha * x[(row + 2) * incx];
            float t3s = alpha * x[(row + 3) * incx];
            __m128 t0 = _mm_set1_ps(t0s);
            __m128 t1 = _mm_set1_ps(t1s);
            __m128 t2 = _mm_set1_ps(t2s);
            __m128 t3 = _mm_set1_ps(t3s);

            CBLAS_INDEX col = 0;
            for (; col < nb8; col += 8)
            {
                __m128 y0 = _mm_loadu_ps(yb + col);
                __m128 y1 = _mm_loadu_ps(yb + col + 4);

                y0 = _mm_add_ps(y0, _mm_mul_ps(_mm_loadu_ps(a0 + col), t0));
                y1 = _mm_add_ps(y1, _mm_mul_ps(_mm_loadu_ps(a0 + col + 4), t0));
                y0 = _mm_add_ps(y0, _mm_mul_ps(_mm_loadu_ps(a1 + col), t1));
                y1 = _mm_add_ps(y1, _mm_mul_ps(_mm_loadu_ps(a1 + col + 4), t1));
                y0 = _mm_add_ps(y0, _mm_mul_ps(_mm_loadu_ps(a2 + col), t2));
                y1 = _mm_add_ps(y1, _mm_mul_ps(_mm_loadu_ps(a2 + col + 4), t2));
                y0 = _mm_add_ps(y0, _mm_mul_ps(_mm_loadu_ps(a3 + col), t3));
                y1 = _mm_add_ps(y1, _mm_mul_ps(_mm_loadu_ps(a3 + col + 4), t3));

                _mm_storeu_ps(yb + col, y0);
                _mm_storeu_ps(yb + col + 4, y1);
            }

            for (; col < nb4; col += 4)
            {
                __m128 y0 = _mm_loadu_ps(yb + col);
                y0 = _mm_add_ps(y0, _mm_mul_ps(_mm_loadu_ps(a0 + col), t0));
                y0 = _mm_add_ps(y0, _mm_mul_ps(_mm_loadu_ps(a1 + col), t1));
                y0 = _mm_add_ps(y0, _mm_mul_ps(_mm_loadu_ps(a2 + col), t2));
                y0 = _mm_add_ps(y0, _mm_mul_ps(_mm_loadu_ps(a3 + col), t3));
                _mm_storeu_ps(yb + col, y0);
            }

            for (; col < nb; col++)
                yb[col] += t0s * a0[col] + t1s * a1[col] + t2s * a2[col] + t3s * a3[col];
        }

        // remaining rows, one at a time
        for (; row < m; row++)
        {
            const float* a0 = a + row * lda + jb;
            float t0s = alpha * x[row * incx];
            __m128 t0 = _mm_set1_ps(t0s);

            CBLAS_INDEX col = 0;
            for (; col < nb4; col += 4)
                _mm_storeu_ps(yb + col, _mm_add_ps(_mm_loadu_ps(yb + col), _mm_mul_ps(_mm_loadu_ps(a0 + col), t0)));

            for (; col < nb; col++)
                yb[col] += t0s * a0[col];
        }
    }
}

//------------------------------------------------------
// Double-precision GEMV kernel, dot form (SSE)
//------------------------------------------------------
void dgemv_k_sse(cblas_args_t* args)
{
    double* a = (double*)args->a;
    double* x = (double*)args->x;
    double* y = (double*)args->y;
    CBLAS_INDEX m = args->m;
    CBLAS_INDEX n = args->n;
    CBLAS_INDEX lda = args->lda;
    CBLAS_INDEX incx = args->incx;
    CBLAS_INDEX incy = args->incy;
    double alpha = *(double*)args->alpha;
    double beta = *(double*)args->beta;

    if (incx != 1)
    {
        // General strided x - scalar
        for (CBLAS_INDEX row = 0; row < m; row++)
        {
            double sum = 0.0;
            for (CBLAS_INDEX col = 0; col < n; col++)
                sum += a[row * lda + col] * x[col * incx];

            double* yp = &y[row * incy];
            *yp = (beta == 0.0 ? 0.0 : beta * *yp) + alpha * sum;
        }
        return;
    }

    CBLAS_INDEX n4 = n & ~(CBLAS_INDEX)3;
    CBLAS_INDEX n2 = n & ~(CBLAS_INDEX)1;
    CBLAS_INDEX row = 0;

    // 4 rows per pass: each x vector load is shared by 4 rows
    for (; row + 4 <= m; row += 4)
    {
        const double* a0 = a + row * lda;
        const double* a1 = a0 + lda;
        const double* a2 = a1 + lda;
        const double* a3 = a2 + lda;

        __m128d s00 = _mm_setzero_pd(), s01 = _mm_setzero_pd();
        __m128d s10 = _mm_setzero_pd(), s11 = _mm_setzero_pd();
        __m128d s20 = _mm_setzero_pd(), s21 = _mm_setzero_pd();
        __m128d s30 = _mm_setzero_pd(), s31 = _mm_setzero_pd();

        CBLAS_INDEX col = 0;
        for (; col < n4; col += 4)
        {
            __m128d x0 = _mm_loadu_pd(x + col);
            __m128d x1 = _mm_loadu_pd(x + col + 2);

            s00 = _mm_add_pd(s00, _mm_mul_pd(_mm_loadu_pd(a0 + col), x0));
            s01 = _mm_add_pd(s01, _mm_mul_pd(_mm_loadu_pd(a0 + col + 2), x1));
            s10 = _mm_add_pd(s10, _mm_mul_pd(_mm_loadu_pd(a1 + col), x0));
            s11 = _mm_add_pd(s11, _mm_mul_pd(_mm_loadu_pd(a1 + col + 2), x1));
            s20 = _mm_add_pd(s20, _mm_mul_pd(_mm_loadu_pd(a2 + col), x0));
            s21 = _mm_add_pd(s21, _mm_mul_pd(_mm_loadu_pd(a2 + col + 2), x1));
            s30 = _mm_add_pd(s30, _mm_mul_pd(_mm_loadu_pd(a3 + col), x0));
            s31 = _mm_add_pd(s31, _mm_mul_pd(_mm_loadu_pd(a3 + col + 2), x1));
        }

        for (; col < n2; col += 2)
        {
            __m128d x0 = _mm_loadu_pd(x + col);

            s00 = _mm_add_pd(s00, _mm_mul_pd(_mm_loadu_pd(a0 + col), x0));
            s10 = _mm_add_pd(s10, _mm_mul_pd(_mm_loadu_pd(a1 + col), x0));
            s20 = _mm_add_pd(s20, _mm_mul_pd(_mm_loadu_pd(a2 + col), x0));
            s30 = _mm_add_pd(s30, _mm_mul_pd(_mm_loadu_pd(a3 + col), x0));
        }

        double sum[4];
        _mm_storeu_pd(sum, _mm_hadd_pd(_mm_add_pd(s00, s01), _mm_add_pd(s10, s11)));
        _mm_storeu_pd(sum + 2, _mm_hadd_pd(_mm_add_pd(s20, s21), _mm_add_pd(s30, s31)));

        for (; col < n; col++)
        {
            double xv = x[col];
            sum[0] += a0[col] * xv;
            sum[1] += a1[col] * xv;
            sum[2] += a2[col] * xv;
            sum[3] += a3[col] * xv;
        }

        for (int r = 0; r < 4; r++)
        {
            double* yp = &y[(row + r) * incy];
            *yp = (beta == 0.0 ? 0.0 : beta * *yp) + alpha * sum[r];
        }
    }

    // remaining rows, one at a time
    for (; row < m; row++)
    {
        const double* a0 = a + row * lda;
        __m128d s0 = _mm_setzero_pd(), s1 = _mm_setzero_pd();

        CBLAS_INDEX col = 0;
        for (; col < n4; col += 4)
        {
            s0 = _mm_add_pd(s0, _mm_mul_pd(_mm_loadu_pd(a0 + col), _mm_loadu_pd(x + col)));
            s1 = _mm_add_pd(s1, _mm_mul_pd(_mm_loadu_pd(a0 + col + 2), _mm_loadu_pd(x + col + 2)));
        }
        for (; col < n2; col += 2)
            s0 = _mm_add_pd(s0, _mm_mul_pd(_mm_loadu_pd(a0 + col), _mm_loadu_pd(x + col)));

        __m128d h = _mm_add_pd(s0, s1);
        double sum = _mm_cvtsd_f64(_mm_hadd_pd(h, h));

        for (; col < n; col++)
            sum += a0[col] * x[col];

        double* yp = &y[row * incy];
        *yp = (beta == 0.0 ? 0.0 : beta * *yp) + alpha * sum;
    }
}

//------------------------------------------------------
// Double-precision GEMV kernel, axpy form (SSE)
//------------------------------------------------------
void dgemv_t_k_sse(cblas_args_t* args)
{
    double* a = (double*)args->a;
    double* x = (double*)args->x;
    double* y = (double*)args->y;
    CBLAS_INDEX m = args->m;
    CBLAS_INDEX n = args->n;
    CBLAS_INDEX lda = args->lda;
    CBLAS_INDEX incx = args->incx;
    CBLAS_INDEX incy = args->incy;
    double alpha = *(double*)args->alpha;
    double beta = *(double*)args->beta;

    // y = beta * y; beta == 0 must not read y (it may hold NaN/Inf)
    if (beta != 1.0)
    {
        for (CBLAS_INDEX col = 0; col < n; col++)
            y[col * incy] = (beta == 0.0) ? 0.0 : beta * y[col * incy];
    }

    if (alpha == 0.0)
        return;

    if (incy != 1)
    {
        // General strided y - scalar, still walking A row by row
        for (CBLAS_INDEX row = 0; row < m; row++)
        {
            double t = alpha * x[row * incx];
            const double* ar = a + row * lda;
            for (CBLAS_INDEX col = 0; col < n; col++)
                y[col * incy] += t * ar[col];
        }
        return;
    }

    for (CBLAS_INDEX jb = 0; jb < n; jb += GEMV_T_BLOCK_D)
    {
        CBLAS_INDEX nb = MIN(GEMV_T_BLOCK_D, n - jb);
        CBLAS_INDEX nb4 = nb & ~(CBLAS_INDEX)3;
        CBLAS_INDEX nb2 = nb & ~(CBLAS_INDEX)1;
        double* yb = y + jb;
        CBLAS_INDEX row = 0;

        // 4 rows per pass: each y load/store is shared by 4 rows
        for (; row + 4 <= m; row += 4)
        {
            const double* a0 = a + row * lda + jb;
            const double* a1 = a0 + lda;
            const double* a2 = a1 + lda;
            const double* a3 = a2 + lda;

            double t0s = alpha * x[(row + 0) * incx];
            double t1s = alpha * x[(row + 1) * incx];
            double t2s = alpha * x[(row + 2) * incx];
            double t3s = alpha * x[(row + 3) * incx];
            __m128d t0 = _mm_set1_pd(t0s);
            __m128d t1 = _mm_set1_pd(t1s);
            __m128d t2 = _mm_set1_pd(t2s);
            __m128d t3 = _mm_set1_pd(t3s);

            CBLAS_INDEX col = 0;
            for (; col < nb4; col += 4)
            {
                __m128d y0 = _mm_loadu_pd(yb + col);
                __m128d y1 = _mm_loadu_pd(yb + col + 2);

                y0 = _mm_add_pd(y0, _mm_mul_pd(_mm_loadu_pd(a0 + col), t0));
                y1 = _mm_add_pd(y1, _mm_mul_pd(_mm_loadu_pd(a0 + col + 2), t0));
                y0 = _mm_add_pd(y0, _mm_mul_pd(_mm_loadu_pd(a1 + col), t1));
                y1 = _mm_add_pd(y1, _mm_mul_pd(_mm_loadu_pd(a1 + col + 2), t1));
                y0 = _mm_add_pd(y0, _mm_mul_pd(_mm_loadu_pd(a2 + col), t2));
                y1 = _mm_add_pd(y1, _mm_mul_pd(_mm_loadu_pd(a2 + col + 2), t2));
                y0 = _mm_add_pd(y0, _mm_mul_pd(_mm_loadu_pd(a3 + col), t3));
                y1 = _mm_add_pd(y1, _mm_mul_pd(_mm_loadu_pd(a3 + col + 2), t3));

                _mm_storeu_pd(yb + col, y0);
                _mm_storeu_pd(yb + col + 2, y1);
            }

            for (; col < nb2; col += 2)
            {
                __m128d y0 = _mm_loadu_pd(yb + col);
                y0 = _mm_add_pd(y0, _mm_mul_pd(_mm_loadu_pd(a0 + col), t0));
                y0 = _mm_add_pd(y0, _mm_mul_pd(_mm_loadu_pd(a1 + col), t1));
                y0 = _mm_add_pd(y0, _mm_mul_pd(_mm_loadu_pd(a2 + col), t2));
                y0 = _mm_add_pd(y0, _mm_mul_pd(_mm_loadu_pd(a3 + col), t3));
                _mm_storeu_pd(yb + col, y0);
            }

            for (; col < nb; col++)
                yb[col] += t0s * a0[col] + t1s * a1[col] + t2s * a2[col] + t3s * a3[col];
        }

        // remaining rows, one at a time
        for (; row < m; row++)
        {
            const double* a0 = a + row * lda + jb;
            double t0s = alpha * x[row * incx];
            __m128d t0 = _mm_set1_pd(t0s);

            CBLAS_INDEX col = 0;
            for (; col < nb2; col += 2)
                _mm_storeu_pd(yb + col, _mm_add_pd(_mm_loadu_pd(yb + col), _mm_mul_pd(_mm_loadu_pd(a0 + col), t0)));

            for (; col < nb; col++)
                yb[col] += t0s * a0[col];
        }
    }
}

#endif // x86/x64
