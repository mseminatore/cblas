//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"
#include "cblas_simd.h"

#if defined(__aarch64__) && defined(__ARM_NEON)

//------------------------------------------------------
// Kernels operate on a row-major m x n matrix A (leading dimension lda):
//   [sd]gemv_k_neon    y[0:m] = alpha * A   * x + beta * y   (dot form)
//   [sd]gemv_t_k_neon  y[0:n] = alpha * A^T * x + beta * y   (axpy form)
//
// Both process 4 rows per pass so each x load (dot form) or y load/store
// (axpy form) is shared by 4 rows. See gemv_k_x86.h for details.
//------------------------------------------------------

#ifdef __ARM_FEATURE_FMA
#   define MADD_F32(acc, a, b) vfmaq_f32((acc), (a), (b))
#   define MADD_F64(acc, a, b) vfmaq_f64((acc), (a), (b))
#else
#   define MADD_F32(acc, a, b) vaddq_f32((acc), vmulq_f32((a), (b)))
#   define MADD_F64(acc, a, b) vaddq_f64((acc), vmulq_f64((a), (b)))
#endif

// number of y elements per axpy-form block (fits comfortably in L1)
#define GEMV_T_BLOCK_S  2048
#define GEMV_T_BLOCK_D  1024

//------------------------------------------------------
// Single-precision GEMV kernel, dot form (NEON)
//------------------------------------------------------
void sgemv_k_neon(cblas_args_t* args)
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

        float32x4_t s00 = vdupq_n_f32(0.0f), s01 = vdupq_n_f32(0.0f);
        float32x4_t s10 = vdupq_n_f32(0.0f), s11 = vdupq_n_f32(0.0f);
        float32x4_t s20 = vdupq_n_f32(0.0f), s21 = vdupq_n_f32(0.0f);
        float32x4_t s30 = vdupq_n_f32(0.0f), s31 = vdupq_n_f32(0.0f);

        CBLAS_INDEX col = 0;
        for (; col < n8; col += 8)
        {
            float32x4_t x0 = vld1q_f32(x + col);
            float32x4_t x1 = vld1q_f32(x + col + 4);

            s00 = MADD_F32(s00, vld1q_f32(a0 + col), x0);
            s01 = MADD_F32(s01, vld1q_f32(a0 + col + 4), x1);
            s10 = MADD_F32(s10, vld1q_f32(a1 + col), x0);
            s11 = MADD_F32(s11, vld1q_f32(a1 + col + 4), x1);
            s20 = MADD_F32(s20, vld1q_f32(a2 + col), x0);
            s21 = MADD_F32(s21, vld1q_f32(a2 + col + 4), x1);
            s30 = MADD_F32(s30, vld1q_f32(a3 + col), x0);
            s31 = MADD_F32(s31, vld1q_f32(a3 + col + 4), x1);
        }

        for (; col < n4; col += 4)
        {
            float32x4_t x0 = vld1q_f32(x + col);

            s00 = MADD_F32(s00, vld1q_f32(a0 + col), x0);
            s10 = MADD_F32(s10, vld1q_f32(a1 + col), x0);
            s20 = MADD_F32(s20, vld1q_f32(a2 + col), x0);
            s30 = MADD_F32(s30, vld1q_f32(a3 + col), x0);
        }

        // pairwise reduce 4 accumulators to one vector of 4 row sums
        float32x4_t s01v = vpaddq_f32(vaddq_f32(s00, s01), vaddq_f32(s10, s11));
        float32x4_t s23v = vpaddq_f32(vaddq_f32(s20, s21), vaddq_f32(s30, s31));
        float sum[4];
        vst1q_f32(sum, vpaddq_f32(s01v, s23v));

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
        float32x4_t s0 = vdupq_n_f32(0.0f), s1 = vdupq_n_f32(0.0f);

        CBLAS_INDEX col = 0;
        for (; col < n8; col += 8)
        {
            s0 = MADD_F32(s0, vld1q_f32(a0 + col), vld1q_f32(x + col));
            s1 = MADD_F32(s1, vld1q_f32(a0 + col + 4), vld1q_f32(x + col + 4));
        }
        for (; col < n4; col += 4)
            s0 = MADD_F32(s0, vld1q_f32(a0 + col), vld1q_f32(x + col));

        float sum = vaddvq_f32(vaddq_f32(s0, s1));
        for (; col < n; col++)
            sum += a0[col] * x[col];

        float* yp = &y[row * incy];
        *yp = (beta == 0.0f ? 0.0f : beta * *yp) + alpha * sum;
    }
}

//------------------------------------------------------
// Single-precision GEMV kernel, axpy form (NEON)
//------------------------------------------------------
void sgemv_t_k_neon(cblas_args_t* args)
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

    if (beta == 0.0f)
    {
        for (CBLAS_INDEX col = 0; col < n; col++)
            y[col * incy] = 0.0f;
    }
    else if (beta != 1.0f)
    {
        for (CBLAS_INDEX col = 0; col < n; col++)
            y[col * incy] *= beta;
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
            float32x4_t t0 = vdupq_n_f32(t0s);
            float32x4_t t1 = vdupq_n_f32(t1s);
            float32x4_t t2 = vdupq_n_f32(t2s);
            float32x4_t t3 = vdupq_n_f32(t3s);

            CBLAS_INDEX col = 0;
            for (; col < nb8; col += 8)
            {
                float32x4_t y0 = vld1q_f32(yb + col);
                float32x4_t y1 = vld1q_f32(yb + col + 4);

                y0 = MADD_F32(y0, vld1q_f32(a0 + col), t0);
                y1 = MADD_F32(y1, vld1q_f32(a0 + col + 4), t0);
                y0 = MADD_F32(y0, vld1q_f32(a1 + col), t1);
                y1 = MADD_F32(y1, vld1q_f32(a1 + col + 4), t1);
                y0 = MADD_F32(y0, vld1q_f32(a2 + col), t2);
                y1 = MADD_F32(y1, vld1q_f32(a2 + col + 4), t2);
                y0 = MADD_F32(y0, vld1q_f32(a3 + col), t3);
                y1 = MADD_F32(y1, vld1q_f32(a3 + col + 4), t3);

                vst1q_f32(yb + col, y0);
                vst1q_f32(yb + col + 4, y1);
            }

            for (; col < nb4; col += 4)
            {
                float32x4_t y0 = vld1q_f32(yb + col);
                y0 = MADD_F32(y0, vld1q_f32(a0 + col), t0);
                y0 = MADD_F32(y0, vld1q_f32(a1 + col), t1);
                y0 = MADD_F32(y0, vld1q_f32(a2 + col), t2);
                y0 = MADD_F32(y0, vld1q_f32(a3 + col), t3);
                vst1q_f32(yb + col, y0);
            }

            for (; col < nb; col++)
                yb[col] += t0s * a0[col] + t1s * a1[col] + t2s * a2[col] + t3s * a3[col];
        }

        // remaining rows, one at a time
        for (; row < m; row++)
        {
            const float* a0 = a + row * lda + jb;
            float t0s = alpha * x[row * incx];
            float32x4_t t0 = vdupq_n_f32(t0s);

            CBLAS_INDEX col = 0;
            for (; col < nb4; col += 4)
                vst1q_f32(yb + col, MADD_F32(vld1q_f32(yb + col), vld1q_f32(a0 + col), t0));

            for (; col < nb; col++)
                yb[col] += t0s * a0[col];
        }
    }
}

//------------------------------------------------------
// Double-precision GEMV kernel, dot form (NEON)
//------------------------------------------------------
void dgemv_k_neon(cblas_args_t* args)
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

        float64x2_t s00 = vdupq_n_f64(0.0), s01 = vdupq_n_f64(0.0);
        float64x2_t s10 = vdupq_n_f64(0.0), s11 = vdupq_n_f64(0.0);
        float64x2_t s20 = vdupq_n_f64(0.0), s21 = vdupq_n_f64(0.0);
        float64x2_t s30 = vdupq_n_f64(0.0), s31 = vdupq_n_f64(0.0);

        CBLAS_INDEX col = 0;
        for (; col < n4; col += 4)
        {
            float64x2_t x0 = vld1q_f64(x + col);
            float64x2_t x1 = vld1q_f64(x + col + 2);

            s00 = MADD_F64(s00, vld1q_f64(a0 + col), x0);
            s01 = MADD_F64(s01, vld1q_f64(a0 + col + 2), x1);
            s10 = MADD_F64(s10, vld1q_f64(a1 + col), x0);
            s11 = MADD_F64(s11, vld1q_f64(a1 + col + 2), x1);
            s20 = MADD_F64(s20, vld1q_f64(a2 + col), x0);
            s21 = MADD_F64(s21, vld1q_f64(a2 + col + 2), x1);
            s30 = MADD_F64(s30, vld1q_f64(a3 + col), x0);
            s31 = MADD_F64(s31, vld1q_f64(a3 + col + 2), x1);
        }

        for (; col < n2; col += 2)
        {
            float64x2_t x0 = vld1q_f64(x + col);

            s00 = MADD_F64(s00, vld1q_f64(a0 + col), x0);
            s10 = MADD_F64(s10, vld1q_f64(a1 + col), x0);
            s20 = MADD_F64(s20, vld1q_f64(a2 + col), x0);
            s30 = MADD_F64(s30, vld1q_f64(a3 + col), x0);
        }

        double sum[4];
        vst1q_f64(sum, vpaddq_f64(vaddq_f64(s00, s01), vaddq_f64(s10, s11)));
        vst1q_f64(sum + 2, vpaddq_f64(vaddq_f64(s20, s21), vaddq_f64(s30, s31)));

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
        float64x2_t s0 = vdupq_n_f64(0.0), s1 = vdupq_n_f64(0.0);

        CBLAS_INDEX col = 0;
        for (; col < n4; col += 4)
        {
            s0 = MADD_F64(s0, vld1q_f64(a0 + col), vld1q_f64(x + col));
            s1 = MADD_F64(s1, vld1q_f64(a0 + col + 2), vld1q_f64(x + col + 2));
        }
        for (; col < n2; col += 2)
            s0 = MADD_F64(s0, vld1q_f64(a0 + col), vld1q_f64(x + col));

        double sum = vaddvq_f64(vaddq_f64(s0, s1));
        for (; col < n; col++)
            sum += a0[col] * x[col];

        double* yp = &y[row * incy];
        *yp = (beta == 0.0 ? 0.0 : beta * *yp) + alpha * sum;
    }
}

//------------------------------------------------------
// Double-precision GEMV kernel, axpy form (NEON)
//------------------------------------------------------
void dgemv_t_k_neon(cblas_args_t* args)
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

    if (beta == 0.0)
    {
        for (CBLAS_INDEX col = 0; col < n; col++)
            y[col * incy] = 0.0;
    }
    else if (beta != 1.0)
    {
        for (CBLAS_INDEX col = 0; col < n; col++)
            y[col * incy] *= beta;
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
            float64x2_t t0 = vdupq_n_f64(t0s);
            float64x2_t t1 = vdupq_n_f64(t1s);
            float64x2_t t2 = vdupq_n_f64(t2s);
            float64x2_t t3 = vdupq_n_f64(t3s);

            CBLAS_INDEX col = 0;
            for (; col < nb4; col += 4)
            {
                float64x2_t y0 = vld1q_f64(yb + col);
                float64x2_t y1 = vld1q_f64(yb + col + 2);

                y0 = MADD_F64(y0, vld1q_f64(a0 + col), t0);
                y1 = MADD_F64(y1, vld1q_f64(a0 + col + 2), t0);
                y0 = MADD_F64(y0, vld1q_f64(a1 + col), t1);
                y1 = MADD_F64(y1, vld1q_f64(a1 + col + 2), t1);
                y0 = MADD_F64(y0, vld1q_f64(a2 + col), t2);
                y1 = MADD_F64(y1, vld1q_f64(a2 + col + 2), t2);
                y0 = MADD_F64(y0, vld1q_f64(a3 + col), t3);
                y1 = MADD_F64(y1, vld1q_f64(a3 + col + 2), t3);

                vst1q_f64(yb + col, y0);
                vst1q_f64(yb + col + 2, y1);
            }

            for (; col < nb2; col += 2)
            {
                float64x2_t y0 = vld1q_f64(yb + col);
                y0 = MADD_F64(y0, vld1q_f64(a0 + col), t0);
                y0 = MADD_F64(y0, vld1q_f64(a1 + col), t1);
                y0 = MADD_F64(y0, vld1q_f64(a2 + col), t2);
                y0 = MADD_F64(y0, vld1q_f64(a3 + col), t3);
                vst1q_f64(yb + col, y0);
            }

            for (; col < nb; col++)
                yb[col] += t0s * a0[col] + t1s * a1[col] + t2s * a2[col] + t3s * a3[col];
        }

        // remaining rows, one at a time
        for (; row < m; row++)
        {
            const double* a0 = a + row * lda + jb;
            double t0s = alpha * x[row * incx];
            float64x2_t t0 = vdupq_n_f64(t0s);

            CBLAS_INDEX col = 0;
            for (; col < nb2; col += 2)
                vst1q_f64(yb + col, MADD_F64(vld1q_f64(yb + col), vld1q_f64(a0 + col), t0));

            for (; col < nb; col++)
                yb[col] += t0s * a0[col];
        }
    }
}

#endif // __aarch64__ && __ARM_NEON
