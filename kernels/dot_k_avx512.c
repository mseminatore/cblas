//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"
#include "cblas_simd.h"

#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX512F__)

//-------------------------------------------------------------------
// single-precision dot product kernel incx == 1 && incy == 1 (AVX-512)
//-------------------------------------------------------------------
void cblas_sdot_k_noinc_avx512(cblas_args_t* args)
{
    float* x = args->x;
    float* y = args->y;
    float* result = args->c;
    register CBLAS_INDEX n = args->n;
    CBLAS_INDEX i = 0;

    // Use 4 independent AVX-512 accumulators to hide FMA latency
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    __m512 sum3 = _mm512_setzero_ps();

    // Process 64 floats per iteration (4 accumulators × 16 floats)
    CBLAS_INDEX unroll_end = (n / 64) * 64;

#if defined(CBLAS_PREFETCH)
    const CBLAS_INDEX prefetch_distance = CBLAS_PREFETCH_DISTANCE * 4;
#endif

    for (; i < unroll_end; i += 64)
    {
#if defined(CBLAS_PREFETCH)
        if (i + prefetch_distance < n) {
            CBLAS_PREFETCH(x + i + prefetch_distance, 0, 3);
            CBLAS_PREFETCH(y + i + prefetch_distance, 0, 3);
        }
#endif

        __m512 x0 = _mm512_loadu_ps(x + i);
        __m512 y0 = _mm512_loadu_ps(y + i);
        __m512 x1 = _mm512_loadu_ps(x + i + 16);
        __m512 y1 = _mm512_loadu_ps(y + i + 16);
        __m512 x2 = _mm512_loadu_ps(x + i + 32);
        __m512 y2 = _mm512_loadu_ps(y + i + 32);
        __m512 x3 = _mm512_loadu_ps(x + i + 48);
        __m512 y3 = _mm512_loadu_ps(y + i + 48);

        sum0 = _mm512_fmadd_ps(x0, y0, sum0);
        sum1 = _mm512_fmadd_ps(x1, y1, sum1);
        sum2 = _mm512_fmadd_ps(x2, y2, sum2);
        sum3 = _mm512_fmadd_ps(x3, y3, sum3);
    }

    // Combine the 4 accumulators
    __m512 sum_avx512 = _mm512_add_ps(_mm512_add_ps(sum0, sum1), _mm512_add_ps(sum2, sum3));

    // Handle remaining blocks of 16
    for (; i + 16 <= n; i += 16)
    {
        __m512 x_vec = _mm512_loadu_ps(x + i);
        __m512 y_vec = _mm512_loadu_ps(y + i);
        sum_avx512 = _mm512_fmadd_ps(x_vec, y_vec, sum_avx512);
    }

    // Horizontal sum using AVX-512 reduce
    float total = _mm512_reduce_add_ps(sum_avx512);

    // Handle remaining elements
    for (; i < n; i++)
    {
        total += x[i] * y[i];
    }

    *result = total;
}

//-------------------------------------------------------------------
// double-precision dot product kernel incx == 1 && incy == 1 (AVX-512)
//-------------------------------------------------------------------
void cblas_ddot_k_noinc_avx512(cblas_args_t* args)
{
    double* x = args->x;
    double* y = args->y;
    double* result = args->c;
    register CBLAS_INDEX n = args->n;
    CBLAS_INDEX i = 0;

    // Use 4 independent AVX-512 accumulators to hide FMA latency
    __m512d sum0 = _mm512_setzero_pd();
    __m512d sum1 = _mm512_setzero_pd();
    __m512d sum2 = _mm512_setzero_pd();
    __m512d sum3 = _mm512_setzero_pd();

    // Process 32 doubles per iteration (4 accumulators × 8 doubles)
    CBLAS_INDEX unroll_end = (n / 32) * 32;

#if defined(CBLAS_PREFETCH)
    const CBLAS_INDEX prefetch_distance = CBLAS_PREFETCH_DISTANCE * 8;
#endif

    for (; i < unroll_end; i += 32)
    {
#if defined(CBLAS_PREFETCH)
        if (i + prefetch_distance < n) {
            CBLAS_PREFETCH(x + i + prefetch_distance, 0, 3);
            CBLAS_PREFETCH(y + i + prefetch_distance, 0, 3);
        }
#endif

        __m512d x0 = _mm512_loadu_pd(x + i);
        __m512d y0 = _mm512_loadu_pd(y + i);
        __m512d x1 = _mm512_loadu_pd(x + i + 8);
        __m512d y1 = _mm512_loadu_pd(y + i + 8);
        __m512d x2 = _mm512_loadu_pd(x + i + 16);
        __m512d y2 = _mm512_loadu_pd(y + i + 16);
        __m512d x3 = _mm512_loadu_pd(x + i + 24);
        __m512d y3 = _mm512_loadu_pd(y + i + 24);

        sum0 = _mm512_fmadd_pd(x0, y0, sum0);
        sum1 = _mm512_fmadd_pd(x1, y1, sum1);
        sum2 = _mm512_fmadd_pd(x2, y2, sum2);
        sum3 = _mm512_fmadd_pd(x3, y3, sum3);
    }

    // Combine the 4 accumulators
    __m512d sum_avx512 = _mm512_add_pd(_mm512_add_pd(sum0, sum1), _mm512_add_pd(sum2, sum3));

    // Handle remaining blocks of 8
    for (; i + 8 <= n; i += 8)
    {
        __m512d x_vec = _mm512_loadu_pd(x + i);
        __m512d y_vec = _mm512_loadu_pd(y + i);
        sum_avx512 = _mm512_fmadd_pd(x_vec, y_vec, sum_avx512);
    }

    // Horizontal sum using AVX-512 reduce
    double total = _mm512_reduce_add_pd(sum_avx512);

    // Handle remaining elements
    for (; i < n; i++)
    {
        total += x[i] * y[i];
    }

    *result = total;
}

#endif
