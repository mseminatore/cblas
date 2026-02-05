//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"
#include "cblas_simd.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)

//------------------------------------------------------
// single-precision nrm2 kernel incx == 1 (AVX)
//------------------------------------------------------
void cblas_snrm2_k_noinc_avx(cblas_args_t* args)
{
    float* x = args->x;
    float* result = args->c;
    register CBLAS_INDEX n = args->n;
    CBLAS_INDEX i = 0;

    // Use 4 independent AVX accumulators to hide latency
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();

    // Process 32 floats per iteration (4 accumulators × 8 floats)
    for (; i + 32 <= n; i += 32)
    {
        __m256 x0 = _mm256_loadu_ps(x + i);
        __m256 x1 = _mm256_loadu_ps(x + i + 8);
        __m256 x2 = _mm256_loadu_ps(x + i + 16);
        __m256 x3 = _mm256_loadu_ps(x + i + 24);

        // Accumulate squares
        sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(x0, x0));
        sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(x1, x1));
        sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(x2, x2));
        sum3 = _mm256_add_ps(sum3, _mm256_mul_ps(x3, x3));
    }

    // Combine the 4 accumulators
    __m256 sum_avx = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));

    // Handle remaining blocks of 8
    for (; i + 8 <= n; i += 8)
    {
        __m256 x_vec = _mm256_loadu_ps(x + i);
        sum_avx = _mm256_add_ps(sum_avx, _mm256_mul_ps(x_vec, x_vec));
    }

    // Horizontal sum of the AVX vector
    __m128 hi = _mm256_extractf128_ps(sum_avx, 1);
    __m128 lo = _mm256_castps256_ps128(sum_avx);
    __m128 sum_sse = _mm_add_ps(lo, hi);

    float sum[4];
    _mm_storeu_ps(sum, sum_sse);
    float total = sum[0] + sum[1] + sum[2] + sum[3];

    // Handle remaining elements
    for (; i < n; i++)
    {
        total += x[i] * x[i];
    }

    *result = sqrtf(total);
}

//------------------------------------------------------
// double-precision nrm2 kernel incx == 1 (AVX)
//------------------------------------------------------
void cblas_dnrm2_k_noinc_avx(cblas_args_t* args)
{
    double* x = args->x;
    double* result = args->c;
    register CBLAS_INDEX n = args->n;
    CBLAS_INDEX i = 0;

    // Use 4 independent AVX accumulators to hide latency
    __m256d sum0 = _mm256_setzero_pd();
    __m256d sum1 = _mm256_setzero_pd();
    __m256d sum2 = _mm256_setzero_pd();
    __m256d sum3 = _mm256_setzero_pd();

    // Process 16 doubles per iteration (4 accumulators × 4 doubles)
    for (; i + 16 <= n; i += 16)
    {
        __m256d x0 = _mm256_loadu_pd(x + i);
        __m256d x1 = _mm256_loadu_pd(x + i + 4);
        __m256d x2 = _mm256_loadu_pd(x + i + 8);
        __m256d x3 = _mm256_loadu_pd(x + i + 12);

        // Accumulate squares
        sum0 = _mm256_add_pd(sum0, _mm256_mul_pd(x0, x0));
        sum1 = _mm256_add_pd(sum1, _mm256_mul_pd(x1, x1));
        sum2 = _mm256_add_pd(sum2, _mm256_mul_pd(x2, x2));
        sum3 = _mm256_add_pd(sum3, _mm256_mul_pd(x3, x3));
    }

    // Combine the 4 accumulators
    __m256d sum_avx = _mm256_add_pd(_mm256_add_pd(sum0, sum1), _mm256_add_pd(sum2, sum3));

    // Handle remaining blocks of 4
    for (; i + 4 <= n; i += 4)
    {
        __m256d x_vec = _mm256_loadu_pd(x + i);
        sum_avx = _mm256_add_pd(sum_avx, _mm256_mul_pd(x_vec, x_vec));
    }

    // Horizontal sum of the AVX vector
    __m128d hi = _mm256_extractf128_pd(sum_avx, 1);
    __m128d lo = _mm256_castpd256_pd128(sum_avx);
    __m128d sum_sse = _mm_add_pd(lo, hi);

    double sum[2];
    _mm_storeu_pd(sum, sum_sse);
    double total = sum[0] + sum[1];

    // Handle remaining elements
    for (; i < n; i++)
    {
        total += x[i] * x[i];
    }

    *result = sqrt(total);
}

#endif
