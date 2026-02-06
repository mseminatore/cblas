//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"
#include "cblas_simd.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)

//------------------------------------------------------
// single-precision asum kernel incx == 1 (AVX)
//------------------------------------------------------
void cblas_sasum_k_noinc_avx(cblas_args_t* args)
{
    float *x = args->x;
    CBLAS_INDEX n = args->n;
    float *result = args->c;
    CBLAS_INDEX i = 0;

    __m256 sum_vec = _mm256_setzero_ps();
    __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)); // Mask to clear sign bit
    int use_prefetch = (n > CBLAS_PREFETCH_THRESHOLD);

    // Process 32 elements at a time using 4 AVX registers (8 floats each)
    for (; i + 32 <= n; i += 32)
    {
        if (use_prefetch && i + CBLAS_PREFETCH_DISTANCE < n) {
            CBLAS_PREFETCH(x + CBLAS_PREFETCH_DISTANCE, 0, 3);
        }

        __m256 a = _mm256_loadu_ps(x);
        __m256 b = _mm256_loadu_ps(x + 8);
        __m256 c = _mm256_loadu_ps(x + 16);
        __m256 d = _mm256_loadu_ps(x + 24);

        // Get absolute values
        a = _mm256_and_ps(a, sign_mask);
        b = _mm256_and_ps(b, sign_mask);
        c = _mm256_and_ps(c, sign_mask);
        d = _mm256_and_ps(d, sign_mask);

        // Accumulate
        sum_vec = _mm256_add_ps(sum_vec, a);
        sum_vec = _mm256_add_ps(sum_vec, b);
        sum_vec = _mm256_add_ps(sum_vec, c);
        sum_vec = _mm256_add_ps(sum_vec, d);

        x += 32;
    }

    // Horizontal sum: reduce 256-bit to 128-bit, then to scalar
    __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
    __m128 sum_low = _mm256_castps256_ps128(sum_vec);
    __m128 sum128 = _mm_add_ps(sum_low, sum_high);

    float sum[4];
    _mm_storeu_ps(sum, sum128);
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
// double-precision asum kernel incx == 1 (AVX)
//------------------------------------------------------
void cblas_dasum_k_noinc_avx(cblas_args_t* args)
{
    double *x = args->x;
    CBLAS_INDEX n = args->n;
    double *result = args->c;
    CBLAS_INDEX i = 0;

    __m256d sum_vec = _mm256_setzero_pd();
    __m256d sign_mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL)); // Mask to clear sign bit
    int use_prefetch = (n > CBLAS_PREFETCH_THRESHOLD);

    // Process 16 elements at a time using 4 AVX registers (4 doubles each)
    for (; i + 16 <= n; i += 16)
    {
        if (use_prefetch && i + CBLAS_PREFETCH_DISTANCE < n) {
            CBLAS_PREFETCH(x + CBLAS_PREFETCH_DISTANCE, 0, 3);
        }

        __m256d a = _mm256_loadu_pd(x);
        __m256d b = _mm256_loadu_pd(x + 4);
        __m256d c = _mm256_loadu_pd(x + 8);
        __m256d d = _mm256_loadu_pd(x + 12);

        // Get absolute values
        a = _mm256_and_pd(a, sign_mask);
        b = _mm256_and_pd(b, sign_mask);
        c = _mm256_and_pd(c, sign_mask);
        d = _mm256_and_pd(d, sign_mask);

        // Accumulate
        sum_vec = _mm256_add_pd(sum_vec, a);
        sum_vec = _mm256_add_pd(sum_vec, b);
        sum_vec = _mm256_add_pd(sum_vec, c);
        sum_vec = _mm256_add_pd(sum_vec, d);

        x += 16;
    }

    // Horizontal sum: reduce 256-bit to 128-bit, then to scalar
    __m128d sum_high = _mm256_extractf128_pd(sum_vec, 1);
    __m128d sum_low = _mm256_castpd256_pd128(sum_vec);
    __m128d sum128 = _mm_add_pd(sum_low, sum_high);

    double sum[2];
    _mm_storeu_pd(sum, sum128);
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
