//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"
#include "cblas_simd.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)

//------------------------------------------------------
// single-precision asum kernel incx == 1 (SSE)
//------------------------------------------------------
void cblas_sasum_k_noinc_sse(cblas_args_t* args)
{
    float *x = args->x;
    CBLAS_INDEX n = args->n;
    float *result = args->c;
    CBLAS_INDEX i = 0;

    __m128 sum_vec = _mm_setzero_ps();
    __m128 sign_mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF)); // Mask to clear sign bit

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
void cblas_dasum_k_noinc_sse(cblas_args_t* args)
{
    double *x = args->x;
    CBLAS_INDEX n = args->n;
    double *result = args->c;
    CBLAS_INDEX i = 0;

    __m128d sum_vec = _mm_setzero_pd();
    __m128d sign_mask = _mm_castsi128_pd(_mm_set1_epi64x(0x7FFFFFFFFFFFFFFFLL)); // Mask to clear sign bit

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