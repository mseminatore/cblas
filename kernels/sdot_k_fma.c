//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"
#include "cblas_simd.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)

//-----------------------------------------------------------------
// single-precision dot product kernel incx == 1 && incy == 1 (FMA)
//-----------------------------------------------------------------
void cblas_sdot_k_noinc_fma(cblas_args_t* args)
{
    float* x = args->x;
    float* y = args->y;
    float* result = args->c;
    register CBLAS_INDEX n = args->n;
    CBLAS_INDEX i = 0;
    __m128 sum_vec = _mm_setzero_ps();

    // AVX2 path: Use 4 independent accumulators to hide latency
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();
    
    // Process 32 floats per iteration (4 accumulators × 8 floats)
    // This reduces dependency chains and improves memory-level parallelism
    CBLAS_INDEX unroll_end = (n / 32) * 32;
    
#if defined(CBLAS_PREFETCH)
    const CBLAS_INDEX prefetch_distance = CBLAS_PREFETCH_DISTANCE * 4; // Prefetch in bytes (64 floats = 256 bytes)
#endif
    
    for (; i < unroll_end; i += 32)
    {
#if defined(CBLAS_PREFETCH)
        // Prefetch data ahead to hide memory latency
        if (i + prefetch_distance < n) {
            CBLAS_PREFETCH(x + i + prefetch_distance, 0, 3);
            CBLAS_PREFETCH(y + i + prefetch_distance, 0, 3);
        }
#endif
        
        __m256 x0 = _mm256_loadu_ps(x + i);
        __m256 y0 = _mm256_loadu_ps(y + i);
        __m256 x1 = _mm256_loadu_ps(x + i + 8);
        __m256 y1 = _mm256_loadu_ps(y + i + 8);
        __m256 x2 = _mm256_loadu_ps(x + i + 16);
        __m256 y2 = _mm256_loadu_ps(y + i + 16);
        __m256 x3 = _mm256_loadu_ps(x + i + 24);
        __m256 y3 = _mm256_loadu_ps(y + i + 24);
        
        // Use FMA instructions: result = (x * y) + accumulator
        sum0 = _mm256_fmadd_ps(x0, y0, sum0);
        sum1 = _mm256_fmadd_ps(x1, y1, sum1);
        sum2 = _mm256_fmadd_ps(x2, y2, sum2);
        sum3 = _mm256_fmadd_ps(x3, y3, sum3);
    }
    
    // Combine the 4 accumulators
    __m256 sum_avx = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
    
    // Handle remaining blocks of 8
    for (; i + 8 <= n; i += 8)
    {
        __m256 x_vec = _mm256_loadu_ps(x + i);
        __m256 y_vec = _mm256_loadu_ps(y + i);
        sum_avx = _mm256_fmadd_ps(x_vec, y_vec, sum_avx);
    }
    
    // Convert AVX to SSE for final reduction
    __m128 low = _mm256_castps256_ps128(sum_avx);
    __m128 high = _mm256_extractf128_ps(sum_avx, 1);
    sum_vec = _mm_add_ps(low, high);

    // SSE path: process 4 floats at a time
    for (; i + 4 <= n; i += 4)
    {
        __m128 x_vec = _mm_loadu_ps(x + i);
        __m128 y_vec = _mm_loadu_ps(y + i);
        sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(x_vec, y_vec));
    }
    
    // Horizontal sum of the vector
    float sum[4];
    _mm_storeu_ps(sum, sum_vec);
    float total = sum[0] + sum[1] + sum[2] + sum[3];
    
    // Handle remaining elements
    for (; i < n; i++)
    {
        total += x[i] * y[i];
    }
    
    *result = total;
}

#endif
