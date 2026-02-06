//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"
#include "cblas_simd.h"

#if defined(__aarch64__) && defined(__ARM_NEON)

//------------------------------------------------------
// Optimized single-precision row-wise dot product kernel (NEON)
// Computes one row of matrix-vector product using NEON and multi-accumulator unrolling
//------------------------------------------------------
static void sgemv_row_dot_neon(float* a_row, float* x, CBLAS_INDEX n, float* result)
{
    CBLAS_INDEX i = 0;
    float sum = 0.0f;

    // NEON path: Use 4 independent accumulators to hide latency
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);

    // Process 16 floats per iteration (4 accumulators × 4 floats)
    CBLAS_INDEX unroll_end = (n / 16) * 16;

#if defined(CBLAS_PREFETCH)
    const CBLAS_INDEX prefetch_distance = 64;
#endif

    for (; i < unroll_end; i += 16)
    {
#if defined(CBLAS_PREFETCH)
        if (i + prefetch_distance < n) {
            CBLAS_PREFETCH(a_row + i + prefetch_distance, 0, 3);
            CBLAS_PREFETCH(x + i + prefetch_distance, 0, 3);
        }
#endif

        float32x4_t a0 = vld1q_f32(a_row + i);
        float32x4_t x0 = vld1q_f32(x + i);
        float32x4_t a1 = vld1q_f32(a_row + i + 4);
        float32x4_t x1 = vld1q_f32(x + i + 4);
        float32x4_t a2 = vld1q_f32(a_row + i + 8);
        float32x4_t x2 = vld1q_f32(x + i + 8);
        float32x4_t a3 = vld1q_f32(a_row + i + 12);
        float32x4_t x3 = vld1q_f32(x + i + 12);

#ifdef __ARM_FEATURE_FMA
        sum0 = vfmaq_f32(sum0, a0, x0);
        sum1 = vfmaq_f32(sum1, a1, x1);
        sum2 = vfmaq_f32(sum2, a2, x2);
        sum3 = vfmaq_f32(sum3, a3, x3);
#else
        sum0 = vaddq_f32(sum0, vmulq_f32(a0, x0));
        sum1 = vaddq_f32(sum1, vmulq_f32(a1, x1));
        sum2 = vaddq_f32(sum2, vmulq_f32(a2, x2));
        sum3 = vaddq_f32(sum3, vmulq_f32(a3, x3));
#endif
    }

    // Combine the 4 accumulators
    float32x4_t sum_neon = vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3));

    // Handle remaining blocks of 4
    for (; i + 4 <= n; i += 4)
    {
        float32x4_t a_vec = vld1q_f32(a_row + i);
        float32x4_t x_vec = vld1q_f32(x + i);

#ifdef __ARM_FEATURE_FMA
        sum_neon = vfmaq_f32(sum_neon, a_vec, x_vec);
#else
        sum_neon = vaddq_f32(sum_neon, vmulq_f32(a_vec, x_vec));
#endif
    }

    // Horizontal sum of NEON vector
    sum = vaddvq_f32(sum_neon);

    // Handle remaining elements
    for (; i < n; i++)
    {
        sum += a_row[i] * x[i];
    }

    *result = sum;
}

//------------------------------------------------------
// Optimized double-precision row-wise dot product kernel (NEON)
//------------------------------------------------------
static void dgemv_row_dot_neon(double* a_row, double* x, CBLAS_INDEX n, double* result)
{
    CBLAS_INDEX i = 0;
    double sum = 0.0;

    // NEON path: Use 4 independent accumulators
    float64x2_t sum0 = vdupq_n_f64(0.0);
    float64x2_t sum1 = vdupq_n_f64(0.0);
    float64x2_t sum2 = vdupq_n_f64(0.0);
    float64x2_t sum3 = vdupq_n_f64(0.0);

    // Process 8 doubles per iteration (4 accumulators × 2 doubles)
    CBLAS_INDEX unroll_end = (n / 8) * 8;

#if defined(CBLAS_PREFETCH)
    const CBLAS_INDEX prefetch_distance = 64;
#endif

    for (; i < unroll_end; i += 8)
    {
#if defined(CBLAS_PREFETCH)
        if (i + prefetch_distance < n) {
            CBLAS_PREFETCH(a_row + i + prefetch_distance, 0, 3);
            CBLAS_PREFETCH(x + i + prefetch_distance, 0, 3);
        }
#endif

        float64x2_t a0 = vld1q_f64(a_row + i);
        float64x2_t x0 = vld1q_f64(x + i);
        float64x2_t a1 = vld1q_f64(a_row + i + 2);
        float64x2_t x1 = vld1q_f64(x + i + 2);
        float64x2_t a2 = vld1q_f64(a_row + i + 4);
        float64x2_t x2 = vld1q_f64(x + i + 4);
        float64x2_t a3 = vld1q_f64(a_row + i + 6);
        float64x2_t x3 = vld1q_f64(x + i + 6);

#ifdef __ARM_FEATURE_FMA
        sum0 = vfmaq_f64(sum0, a0, x0);
        sum1 = vfmaq_f64(sum1, a1, x1);
        sum2 = vfmaq_f64(sum2, a2, x2);
        sum3 = vfmaq_f64(sum3, a3, x3);
#else
        sum0 = vaddq_f64(sum0, vmulq_f64(a0, x0));
        sum1 = vaddq_f64(sum1, vmulq_f64(a1, x1));
        sum2 = vaddq_f64(sum2, vmulq_f64(a2, x2));
        sum3 = vaddq_f64(sum3, vmulq_f64(a3, x3));
#endif
    }

    // Combine the 4 accumulators
    float64x2_t sum_neon = vaddq_f64(vaddq_f64(sum0, sum1), vaddq_f64(sum2, sum3));

    // Handle remaining blocks of 2
    for (; i + 2 <= n; i += 2)
    {
        float64x2_t a_vec = vld1q_f64(a_row + i);
        float64x2_t x_vec = vld1q_f64(x + i);

#ifdef __ARM_FEATURE_FMA
        sum_neon = vfmaq_f64(sum_neon, a_vec, x_vec);
#else
        sum_neon = vaddq_f64(sum_neon, vmulq_f64(a_vec, x_vec));
#endif
    }

    // Horizontal sum
    sum = vaddvq_f64(sum_neon);

    // Handle remaining elements
    for (; i < n; i++)
    {
        sum += a_row[i] * x[i];
    }

    *result = sum;
}

//------------------------------------------------------
// Single-precision GEMV kernel (NEON)
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

    float sum;

    // Optimized path for unit strides
    if (incx == 1 && incy == 1)
    {
        for (CBLAS_INDEX row = 0; row < m; row++)
        {
            sgemv_row_dot_neon(&a[row * lda], x, n, &sum);
            y[row] = beta * y[row] + alpha * sum;
        }
    }
    else
    {
        // General case with strides - fall back to scalar
        for (CBLAS_INDEX row = 0; row < m; row++)
        {
            sum = 0.0f;
            for (CBLAS_INDEX col = 0; col < n; col++)
            {
                sum += a[row * lda + col] * x[col * incx];
            }
            y[row * incy] = beta * y[row * incy] + alpha * sum;
        }
    }
}

//------------------------------------------------------
// Double-precision GEMV kernel (NEON)
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

    double sum;

    // Optimized path for unit strides
    if (incx == 1 && incy == 1)
    {
        for (CBLAS_INDEX row = 0; row < m; row++)
        {
            dgemv_row_dot_neon(&a[row * lda], x, n, &sum);
            y[row] = beta * y[row] + alpha * sum;
        }
    }
    else
    {
        // General case with strides - fall back to scalar
        for (CBLAS_INDEX row = 0; row < m; row++)
        {
            sum = 0.0;
            for (CBLAS_INDEX col = 0; col < n; col++)
            {
                sum += a[row * lda + col] * x[col * incx];
            }
            y[row * incy] = beta * y[row * incy] + alpha * sum;
        }
    }
}

#endif // __aarch64__ && __ARM_NEON
