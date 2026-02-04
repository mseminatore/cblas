//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"
#include "cblas_simd.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)

//--------------------------------------------------------------------------
// double-precision dot product kernel incx == 1 && incy == 1 (AVX - no FMA)
//--------------------------------------------------------------------------
void cblas_ddot_k_noinc_avx(cblas_args_t* args)
{
    double* x = args->x;
    double* y = args->y;
    double* result = args->c;
    register CBLAS_INDEX n = args->n;
    CBLAS_INDEX i = 0;
    __m128d sum_vec = _mm_setzero_pd();

    // AVX2 path: Use 4 independent accumulators to hide latency
    __m256d sum0 = _mm256_setzero_pd();
    __m256d sum1 = _mm256_setzero_pd();
    __m256d sum2 = _mm256_setzero_pd();
    __m256d sum3 = _mm256_setzero_pd();
    
    // Process 16 doubles per iteration (4 accumulators × 4 doubles)
    CBLAS_INDEX unroll_end = (n / 16) * 16;
    
#if defined(CBLAS_PREFETCH)
    const CBLAS_INDEX prefetch_distance = CBLAS_PREFETCH_DISTANCE * 8; // Prefetch in bytes (32 doubles = 256 bytes)
#endif
    
    for (; i < unroll_end; i += 16)
    {
#if defined(CBLAS_PREFETCH)
        // Prefetch data ahead to hide memory latency
        if (i + prefetch_distance < n) {
            CBLAS_PREFETCH(x + i + prefetch_distance, 0, 3);
            CBLAS_PREFETCH(y + i + prefetch_distance, 0, 3);
        }
#endif
        
        __m256d x0 = _mm256_loadu_pd(x + i);
        __m256d y0 = _mm256_loadu_pd(y + i);
        __m256d x1 = _mm256_loadu_pd(x + i + 4);
        __m256d y1 = _mm256_loadu_pd(y + i + 4);
        __m256d x2 = _mm256_loadu_pd(x + i + 8);
        __m256d y2 = _mm256_loadu_pd(y + i + 8);
        __m256d x3 = _mm256_loadu_pd(x + i + 12);
        __m256d y3 = _mm256_loadu_pd(y + i + 12);
        
        // Use separate multiply and add (no FMA)
        sum0 = _mm256_add_pd(sum0, _mm256_mul_pd(x0, y0));
        sum1 = _mm256_add_pd(sum1, _mm256_mul_pd(x1, y1));
        sum2 = _mm256_add_pd(sum2, _mm256_mul_pd(x2, y2));
        sum3 = _mm256_add_pd(sum3, _mm256_mul_pd(x3, y3));
    }
    
    // Combine the 4 accumulators
    __m256d sum_avx = _mm256_add_pd(_mm256_add_pd(sum0, sum1), _mm256_add_pd(sum2, sum3));
    
    // Handle remaining blocks of 4
    for (; i + 4 <= n; i += 4)
    {
        __m256d x_vec = _mm256_loadu_pd(x + i);
        __m256d y_vec = _mm256_loadu_pd(y + i);
        sum_avx = _mm256_add_pd(sum_avx, _mm256_mul_pd(x_vec, y_vec));
    }
    
    // Convert AVX to SSE for final reduction
    __m128d low = _mm256_castpd256_pd128(sum_avx);
    __m128d high = _mm256_extractf128_pd(sum_avx, 1);
    sum_vec = _mm_add_pd(low, high);

    // SSE path: process 2 doubles at a time
    for (; i + 2 <= n; i += 2)
    {
        __m128d x_vec = _mm_loadu_pd(x + i);
        __m128d y_vec = _mm_loadu_pd(y + i);
        sum_vec = _mm_add_pd(sum_vec, _mm_mul_pd(x_vec, y_vec));
    }
    
    // Horizontal sum of the vector
    double sum[2];
    _mm_storeu_pd(sum, sum_vec);
    double total = sum[0] + sum[1];
    
    // Handle remaining elements
    for (; i < n; i++)
    {
        total += x[i] * y[i];
    }
    
    *result = total;
}

#endif
