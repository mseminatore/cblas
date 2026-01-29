//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

//#include <math.h>
#include "cblas.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)
#   include <immintrin.h>
#endif

#if defined(__aarch64__) && defined(__ARM_NEON)
#   include <arm_neon.h>
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)

//------------------------------------------------------
// single-precision nrm2 kernel incx == 1 (SSE)
//------------------------------------------------------
static void cblas_snrm2_k_noinc_sse(float *x, CBLAS_INDEX n, float *result)
{
    __m128 sum_vec = _mm_setzero_ps();
    CBLAS_INDEX i = 0;

    // Process 16 elements at a time using 4 SSE registers
    for (; i + 16 <= n; i += 16)
    {
        __m128 a = _mm_loadu_ps(x);
        __m128 b = _mm_loadu_ps(x + 4);
        __m128 c = _mm_loadu_ps(x + 8);
        __m128 d = _mm_loadu_ps(x + 12);

        // Accumulate squares
        sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(a, a));
        sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(b, b));
        sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(c, c));
        sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(d, d));

        x += 16;
    }

    // Horizontal sum of the vector
    float sum[4];
    _mm_storeu_ps(sum, sum_vec);
    float total = sum[0] + sum[1] + sum[2] + sum[3];

    // Handle remaining elements
    for (; i < n; i++)
    {
        total += *x * *x;
        x++;
    }

    *result = sqrtf(total);
}

//------------------------------------------------------
// double-precision nrm2 kernel incx == 1 (SSE)
//------------------------------------------------------
static void cblas_dnrm2_k_noinc_sse(double *x, CBLAS_INDEX n, double *result)
{
    __m128d sum_vec = _mm_setzero_pd();
    CBLAS_INDEX i = 0;

    // Process 8 elements at a time using 4 SSE registers (2 doubles each)
    for (; i + 8 <= n; i += 8)
    {
        __m128d a = _mm_loadu_pd(x);
        __m128d b = _mm_loadu_pd(x + 2);
        __m128d c = _mm_loadu_pd(x + 4);
        __m128d d = _mm_loadu_pd(x + 6);

        // Accumulate squares
        sum_vec = _mm_add_pd(sum_vec, _mm_mul_pd(a, a));
        sum_vec = _mm_add_pd(sum_vec, _mm_mul_pd(b, b));
        sum_vec = _mm_add_pd(sum_vec, _mm_mul_pd(c, c));
        sum_vec = _mm_add_pd(sum_vec, _mm_mul_pd(d, d));

        x += 8;
    }

    // Horizontal sum of the vector
    double sum[2];
    _mm_storeu_pd(sum, sum_vec);
    double total = sum[0] + sum[1];

    // Handle remaining elements
    for (; i < n; i++)
    {
        total += *x * *x;
        x++;
    }

    *result = sqrt(total);
}

#endif

#if defined(__aarch64__) && defined(__ARM_NEON)

//------------------------------------------------------
// single-precision nrm2 kernel incx == 1 (NEON)
//------------------------------------------------------
static void cblas_snrm2_k_noinc_neon(float *x, CBLAS_INDEX n, float *result)
{
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    CBLAS_INDEX i = 0;

    // Process 16 elements at a time using 4 NEON registers
    for (; i + 16 <= n; i += 16)
    {
        float32x4_t a = vld1q_f32(x);
        float32x4_t b = vld1q_f32(x + 4);
        float32x4_t c = vld1q_f32(x + 8);
        float32x4_t d = vld1q_f32(x + 12);

        // Accumulate squares
        sum_vec = vfmaq_f32(sum_vec, a, a);
        sum_vec = vfmaq_f32(sum_vec, b, b);
        sum_vec = vfmaq_f32(sum_vec, c, c);
        sum_vec = vfmaq_f32(sum_vec, d, d);

        x += 16;
    }

    // Horizontal sum of the vector
    float total = vaddvq_f32(sum_vec);

    // Handle remaining elements
    for (; i < n; i++)
    {
        total += *x * *x;
        x++;
    }

    *result = sqrtf(total);
}

//------------------------------------------------------
// double-precision nrm2 kernel incx == 1 (NEON)
//------------------------------------------------------
static void cblas_dnrm2_k_noinc_neon(double *x, CBLAS_INDEX n, double *result)
{
    float64x2_t sum_vec = vdupq_n_f64(0.0);
    CBLAS_INDEX i = 0;

    // Process 8 elements at a time using 4 NEON registers (2 doubles each)
    for (; i + 8 <= n; i += 8)
    {
        float64x2_t a = vld1q_f64(x);
        float64x2_t b = vld1q_f64(x + 2);
        float64x2_t c = vld1q_f64(x + 4);
        float64x2_t d = vld1q_f64(x + 6);

        // Accumulate squares
        sum_vec = vfmaq_f64(sum_vec, a, a);
        sum_vec = vfmaq_f64(sum_vec, b, b);
        sum_vec = vfmaq_f64(sum_vec, c, c);
        sum_vec = vfmaq_f64(sum_vec, d, d);

        x += 8;
    }

    // Horizontal sum of the vector
    double total = vaddvq_f64(sum_vec);

    // Handle remaining elements
    for (; i < n; i++)
    {
        total += *x * *x;
        x++;
    }

    *result = sqrt(total);
}

#endif

//------------------------------------------------------
// Level-1 single-precision euclidean norm
//------------------------------------------------------
float cblas_snrm2(CBLAS_INDEX n, float *x, CBLAS_INDEX incx)
{
    float sum = 0.0f;

    CBLAS_VALIDATE_VEC1(n, x, incx, sum);

    CBLAS_STATS_START();

    int mt_used = 0;

    if (incx == 1)
    {
#if defined(USE_SSE) && defined(USE_SIMD)
#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)
        cblas_snrm2_k_noinc_sse(x, n, &sum);
        return sum;
#elif defined(__aarch64__) && defined(__ARM_NEON)
        cblas_snrm2_k_noinc_neon(x, n, &sum);
        return sum;
#endif
#endif
        // Fallback scalar implementation with unrolling
        CBLAS_INDEX i = 0;
        register float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

        for (; i + 4 <= n; i += 4)
        {
            sum0 += x[i] * x[i];
            sum1 += x[i+1] * x[i+1];
            sum2 += x[i+2] * x[i+2];
            sum3 += x[i+3] * x[i+3];
        }

        sum = sum0 + sum1 + sum2 + sum3;

        for (; i < n; i++)
        {
            sum += x[i] * x[i];
        }
    }
    else
    {
        // incx != 1
        for (CBLAS_INDEX i = 0; i < n; i++)
        {
            sum += *x * *x;
            x += incx;
        }
    }

    CBLAS_STATS_END("snrm2", n, mt_used);

    return sqrtf(sum);
}

//------------------------------------------------------
// Level-1 single-precision euclidean norm
//------------------------------------------------------
double cblas_dnrm2(CBLAS_INDEX n, double *x, CBLAS_INDEX incx)
{
    double sum = 0.0;

    CBLAS_VALIDATE_VEC1(n, x, incx, sum);

    CBLAS_STATS_START();

    int mt_used = 0;

    if (incx == 1)
    {
#if defined(USE_SSE) && defined(USE_SIMD)
#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)
        cblas_dnrm2_k_noinc_sse(x, n, &sum);
        return sum;
#elif defined(__aarch64__) && defined(__ARM_NEON)
        cblas_dnrm2_k_noinc_neon(x, n, &sum);
        return sum;
#endif
#endif
        // Fallback scalar implementation with unrolling
        CBLAS_INDEX i = 0;
        register double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;

        for (; i + 4 <= n; i += 4)
        {
            sum0 += x[i] * x[i];
            sum1 += x[i+1] * x[i+1];
            sum2 += x[i+2] * x[i+2];
            sum3 += x[i+3] * x[i+3];
        }

        sum = sum0 + sum1 + sum2 + sum3;

        for (; i < n; i++)
        {
            sum += x[i] * x[i];
        }
    }
    else
    {
        // incx != 1
        for (CBLAS_INDEX i = 0; i < n; i++)
        {
            sum += *x * *x;
            x += incx;
        }
    }

    CBLAS_STATS_END("dnrm2", n, mt_used);

    return sqrt(sum);
}
