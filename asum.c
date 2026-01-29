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

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)

//------------------------------------------------------
// single-precision asum kernel incx == 1 (SSE)
//------------------------------------------------------
static void cblas_sasum_k_noinc_sse(float *x, CBLAS_INDEX n, float *result)
{
    __m128 sum_vec = _mm_setzero_ps();
    __m128 sign_mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF)); // Mask to clear sign bit
    CBLAS_INDEX i = 0;

    // Process 16 elements at a time using 4 SSE registers
    for (; i + 16 <= n; i += 16)
    {
        __m128 a = _mm_loadu_ps(x);
        __m128 b = _mm_loadu_ps(x + 4);
        __m128 c = _mm_loadu_ps(x + 8);
        __m128 d = _mm_loadu_ps(x + 12);

        // Get absolute values
        a = _mm_and_ps(a, sign_mask);
        b = _mm_and_ps(b, sign_mask);
        c = _mm_and_ps(c, sign_mask);
        d = _mm_and_ps(d, sign_mask);

        // Accumulate
        sum_vec = _mm_add_ps(sum_vec, a);
        sum_vec = _mm_add_ps(sum_vec, b);
        sum_vec = _mm_add_ps(sum_vec, c);
        sum_vec = _mm_add_ps(sum_vec, d);

        x += 16;
    }

    // Horizontal sum of the vector
    float sum[4];
    _mm_storeu_ps(sum, sum_vec);
    float total = sum[0] + sum[1] + sum[2] + sum[3];

    // Handle remaining elements
    for (; i < n; i++)
    {
        total += fabsf(*x);
        x++;
    }

    *result = total;
}

//------------------------------------------------------
// double-precision asum kernel incx == 1 (SSE)
//------------------------------------------------------
static void cblas_dasum_k_noinc_sse(double *x, CBLAS_INDEX n, double *result)
{
    __m128d sum_vec = _mm_setzero_pd();
    __m128d sign_mask = _mm_castsi128_pd(_mm_set1_epi64x(0x7FFFFFFFFFFFFFFFLL)); // Mask to clear sign bit
    CBLAS_INDEX i = 0;

    // Process 8 elements at a time using 4 SSE registers (2 doubles each)
    for (; i + 8 <= n; i += 8)
    {
        __m128d a = _mm_loadu_pd(x);
        __m128d b = _mm_loadu_pd(x + 2);
        __m128d c = _mm_loadu_pd(x + 4);
        __m128d d = _mm_loadu_pd(x + 6);

        // Get absolute values
        a = _mm_and_pd(a, sign_mask);
        b = _mm_and_pd(b, sign_mask);
        c = _mm_and_pd(c, sign_mask);
        d = _mm_and_pd(d, sign_mask);

        // Accumulate
        sum_vec = _mm_add_pd(sum_vec, a);
        sum_vec = _mm_add_pd(sum_vec, b);
        sum_vec = _mm_add_pd(sum_vec, c);
        sum_vec = _mm_add_pd(sum_vec, d);

        x += 8;
    }

    // Horizontal sum of the vector
    double sum[2];
    _mm_storeu_pd(sum, sum_vec);
    double total = sum[0] + sum[1];

    // Handle remaining elements
    for (; i < n; i++)
    {
        total += fabs(*x);
        x++;
    }

    *result = total;
}

#endif

#if defined(__aarch64__) && defined(__ARM_NEON)

//------------------------------------------------------
// single-precision asum kernel incx == 1 (NEON)
//------------------------------------------------------
static void cblas_sasum_k_noinc_neon(float *x, CBLAS_INDEX n, float *result)
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

        // Get absolute values
        a = vabsq_f32(a);
        b = vabsq_f32(b);
        c = vabsq_f32(c);
        d = vabsq_f32(d);

        // Accumulate
        sum_vec = vaddq_f32(sum_vec, a);
        sum_vec = vaddq_f32(sum_vec, b);
        sum_vec = vaddq_f32(sum_vec, c);
        sum_vec = vaddq_f32(sum_vec, d);

        x += 16;
    }

    // Horizontal sum of the vector
    float total = vaddvq_f32(sum_vec);

    // Handle remaining elements
    for (; i < n; i++)
    {
        total += fabsf(*x);
        x++;
    }

    *result = total;
}

//------------------------------------------------------
// double-precision asum kernel incx == 1 (NEON)
//------------------------------------------------------
static void cblas_dasum_k_noinc_neon(double *x, CBLAS_INDEX n, double *result)
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

        // Get absolute values
        a = vabsq_f64(a);
        b = vabsq_f64(b);
        c = vabsq_f64(c);
        d = vabsq_f64(d);

        // Accumulate
        sum_vec = vaddq_f64(sum_vec, a);
        sum_vec = vaddq_f64(sum_vec, b);
        sum_vec = vaddq_f64(sum_vec, c);
        sum_vec = vaddq_f64(sum_vec, d);

        x += 8;
    }

    // Horizontal sum of the vector
    double total = vaddvq_f64(sum_vec);

    // Handle remaining elements
    for (; i < n; i++)
    {
        total += fabs(*x);
        x++;
    }

    *result = total;
}

#endif

//------------------------------------------------------
// Level-1 single-precision vector sum
//------------------------------------------------------
float cblas_sasum(CBLAS_INDEX n, float *x, CBLAS_INDEX incx)
{
    float sum = 0.0f;

#ifdef CBLAS_CHECK_INPUTS

#ifdef CBLAS_XERBLA_INPUTS
    int info = 0;
    if (n < 0)
        info = 1;
    else if (!x)
        info = 2;
    else if (incx <= 0)
        info = 3;

    if (info) {
        XERBLA(info);
        return sum;
    }
#else
    if (n < 0 || !x || incx <= 0)
    {
        assert(n > 0 && x && incx > 0);
        return sum;
    }
#endif  // CBLAS_XERBLA_INPUTS

#endif  // CBLAS_CHECK_INPUTS

    if (incx == 1)
    {
#if defined(USE_SSE) && defined(USE_SIMD)
#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)
        cblas_sasum_k_noinc_sse(x, n, &sum);
        return sum;
#elif defined(__aarch64__) && defined(__ARM_NEON)
        cblas_sasum_k_noinc_neon(x, n, &sum);
        return sum;
#endif
#endif
        // Fallback scalar implementation with unrolling
        CBLAS_INDEX i = 0;
        register float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

        for (; i + 4 <= n; i += 4)
        {
            sum0 += fabsf(x[i]);
            sum1 += fabsf(x[i+1]);
            sum2 += fabsf(x[i+2]);
            sum3 += fabsf(x[i+3]);
        }

        sum = sum0 + sum1 + sum2 + sum3;

        for (; i < n; i++)
        {
            sum += fabsf(x[i]);
        }
    }
    else
    {
        // incx != 1
        for (CBLAS_INDEX i = 0; i < n; i++)
        {
            sum += fabsf(*x);
            x += incx;
        }
    }

    return sum;
}

//------------------------------------------------------
// Level-1 double-precision vector sum
//------------------------------------------------------
double cblas_dasum(CBLAS_INDEX n, double *x, CBLAS_INDEX incx)
{
    double sum = 0.0;

#ifdef CBLAS_CHECK_INPUTS

#ifdef CBLAS_XERBLA_INPUTS
    int info = 0;
    if (n < 0)
        info = 1;
    else if (!x)
        info = 2;
    else if (incx <= 0)
        info = 3;

    if (info) {
        XERBLA(info);
        return sum;
    }
#else
    if (n < 0 || !x || incx <= 0)
    {
        assert(n > 0 && x && incx > 0);
        return sum;
    }
#endif
#endif

    if (incx == 1)
    {
#if defined(USE_SSE) && defined(USE_SIMD)
#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)
        cblas_dasum_k_noinc_sse(x, n, &sum);
        return sum;
#elif defined(__aarch64__) && defined(__ARM_NEON)
        cblas_dasum_k_noinc_neon(x, n, &sum);
        return sum;
#endif
#endif
        // Fallback scalar implementation with unrolling
        CBLAS_INDEX i = 0;
        register double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;

        for (; i + 4 <= n; i += 4)
        {
            sum0 += fabs(x[i]);
            sum1 += fabs(x[i+1]);
            sum2 += fabs(x[i+2]);
            sum3 += fabs(x[i+3]);
        }

        sum = sum0 + sum1 + sum2 + sum3;

        for (; i < n; i++)
        {
            sum += fabs(x[i]);
        }
    }
    else
    {
        // incx != 1
        for (CBLAS_INDEX i = 0; i < n; i++)
        {
            sum += fabs(*x);
            x += incx;
        }
    }

    return sum;
}
