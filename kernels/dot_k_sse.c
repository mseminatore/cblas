//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"
#include "cblas_simd.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)

//--------------------------------------------------------------------------
// single-precision dot product kernel incx == 1 && incy == 1 (SSE only)
//--------------------------------------------------------------------------
void cblas_sdot_k_noinc_sse(cblas_args_t* args)
{
    float* x = args->x;
    float* y = args->y;
    float* result = args->c;
    register CBLAS_INDEX n = args->n;
    CBLAS_INDEX i = 0;

    // Use 4 independent SSE accumulators to hide latency
    __m128 sum0 = _mm_setzero_ps();
    __m128 sum1 = _mm_setzero_ps();
    __m128 sum2 = _mm_setzero_ps();
    __m128 sum3 = _mm_setzero_ps();
    
    // Process 16 floats per iteration (4 accumulators × 4 floats)
    // This reduces dependency chains and improves memory-level parallelism
    CBLAS_INDEX unroll_end = (n / 16) * 16;
    
#if defined(CBLAS_PREFETCH)
    const CBLAS_INDEX prefetch_distance = CBLAS_PREFETCH_DISTANCE * 4; // Prefetch in bytes (64 floats = 256 bytes)
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
        
        __m128 x0 = _mm_loadu_ps(x + i);
        __m128 y0 = _mm_loadu_ps(y + i);
        __m128 x1 = _mm_loadu_ps(x + i + 4);
        __m128 y1 = _mm_loadu_ps(y + i + 4);
        __m128 x2 = _mm_loadu_ps(x + i + 8);
        __m128 y2 = _mm_loadu_ps(y + i + 8);
        __m128 x3 = _mm_loadu_ps(x + i + 12);
        __m128 y3 = _mm_loadu_ps(y + i + 12);
        
        // Multiply and add (no FMA)
        sum0 = _mm_add_ps(sum0, _mm_mul_ps(x0, y0));
        sum1 = _mm_add_ps(sum1, _mm_mul_ps(x1, y1));
        sum2 = _mm_add_ps(sum2, _mm_mul_ps(x2, y2));
        sum3 = _mm_add_ps(sum3, _mm_mul_ps(x3, y3));
    }
    
    // Combine the 4 accumulators
    __m128 sum_sse = _mm_add_ps(_mm_add_ps(sum0, sum1), _mm_add_ps(sum2, sum3));
    
    // Handle remaining blocks of 4
    for (; i + 4 <= n; i += 4)
    {
        __m128 x_vec = _mm_loadu_ps(x + i);
        __m128 y_vec = _mm_loadu_ps(y + i);
        sum_sse = _mm_add_ps(sum_sse, _mm_mul_ps(x_vec, y_vec));
    }
    
    // Horizontal sum of the vector
    float sum[4];
    _mm_storeu_ps(sum, sum_sse);
    float total = sum[0] + sum[1] + sum[2] + sum[3];
    
    // Handle remaining elements
    for (; i < n; i++)
    {
        total += x[i] * y[i];
    }
    
    *result = total;
}

//--------------------------------------------------------------------------
// double-precision dot product kernel incx == 1 && incy == 1 (SSE only)
//--------------------------------------------------------------------------
void cblas_ddot_k_noinc_sse(cblas_args_t* args)
{
    double* x = args->x;
    double* y = args->y;
    double* result = args->c;
    register CBLAS_INDEX n = args->n;
    CBLAS_INDEX i = 0;

    // Use 4 independent SSE accumulators to hide latency
    __m128d sum0 = _mm_setzero_pd();
    __m128d sum1 = _mm_setzero_pd();
    __m128d sum2 = _mm_setzero_pd();
    __m128d sum3 = _mm_setzero_pd();
    
    // Process 8 doubles per iteration (4 accumulators × 2 doubles)
    // This reduces dependency chains and improves memory-level parallelism
    CBLAS_INDEX unroll_end = (n / 8) * 8;
    
#if defined(CBLAS_PREFETCH)
    const CBLAS_INDEX prefetch_distance = CBLAS_PREFETCH_DISTANCE * 8; // Prefetch in bytes (32 doubles = 256 bytes)
#endif
    
    for (; i < unroll_end; i += 8)
    {
#if defined(CBLAS_PREFETCH)
        // Prefetch data ahead to hide memory latency
        if (i + prefetch_distance < n) {
            CBLAS_PREFETCH(x + i + prefetch_distance, 0, 3);
            CBLAS_PREFETCH(y + i + prefetch_distance, 0, 3);
        }
#endif
        
        __m128d x0 = _mm_loadu_pd(x + i);
        __m128d y0 = _mm_loadu_pd(y + i);
        __m128d x1 = _mm_loadu_pd(x + i + 2);
        __m128d y1 = _mm_loadu_pd(y + i + 2);
        __m128d x2 = _mm_loadu_pd(x + i + 4);
        __m128d y2 = _mm_loadu_pd(y + i + 4);
        __m128d x3 = _mm_loadu_pd(x + i + 6);
        __m128d y3 = _mm_loadu_pd(y + i + 6);
        
        // Multiply and add (no FMA)
        sum0 = _mm_add_pd(sum0, _mm_mul_pd(x0, y0));
        sum1 = _mm_add_pd(sum1, _mm_mul_pd(x1, y1));
        sum2 = _mm_add_pd(sum2, _mm_mul_pd(x2, y2));
        sum3 = _mm_add_pd(sum3, _mm_mul_pd(x3, y3));
    }
    
    // Combine the 4 accumulators
    __m128d sum_sse = _mm_add_pd(_mm_add_pd(sum0, sum1), _mm_add_pd(sum2, sum3));
    
    // Handle remaining blocks of 2
    for (; i + 2 <= n; i += 2)
    {
        __m128d x_vec = _mm_loadu_pd(x + i);
        __m128d y_vec = _mm_loadu_pd(y + i);
        sum_sse = _mm_add_pd(sum_sse, _mm_mul_pd(x_vec, y_vec));
    }
    
    // Horizontal sum of the vector
    double sum[2];
    _mm_storeu_pd(sum, sum_sse);
    double total = sum[0] + sum[1];
    
    // Handle remaining elements
    for (; i < n; i++)
    {
        total += x[i] * y[i];
    }
    
    *result = total;
}

#endif
