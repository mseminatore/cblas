//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"
#include "cblas_simd.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)

//------------------------------------------------------
// single-precision copy kernel incx == incy == 1 (AVX)
//------------------------------------------------------
void cblas_scopy_k_noinc_avx(cblas_args_t* args)
{
    float* x = args->x;
    float* y = args->y;
    register CBLAS_INDEX n = args->n;

    register CBLAS_INDEX i = 0;
    int use_prefetch = (n > CBLAS_PREFETCH_THRESHOLD);

    // Process 32 floats per iteration (4x8 with AVX) for better ILP
    // Use 4 independent load/store streams to hide latency
    for (; i + 32 <= n; i += 32)
    {
        if (use_prefetch && i + CBLAS_PREFETCH_DISTANCE < n) {
            CBLAS_PREFETCH(&x[i + CBLAS_PREFETCH_DISTANCE], 0, 3);
            CBLAS_PREFETCH(&y[i + CBLAS_PREFETCH_DISTANCE], 1, 3);
        }

        // Load 32 floats using 4 independent streams
        __m256 a0 = _mm256_loadu_ps(&x[i]);
        __m256 a1 = _mm256_loadu_ps(&x[i + 8]);
        __m256 a2 = _mm256_loadu_ps(&x[i + 16]);
        __m256 a3 = _mm256_loadu_ps(&x[i + 24]);

        // Store 32 floats
        _mm256_storeu_ps(&y[i], a0);
        _mm256_storeu_ps(&y[i + 8], a1);
        _mm256_storeu_ps(&y[i + 16], a2);
        _mm256_storeu_ps(&y[i + 24], a3);
    }

    // Process remaining elements
    for (; i < n; i++)
        y[i] = x[i];
}

//------------------------------------------------------
// double-precision copy kernel incx == incy == 1 (AVX)
//------------------------------------------------------
void cblas_dcopy_k_noinc_avx(cblas_args_t* args)
{
    double *x = args->x;
    double *y = args->y;
    register CBLAS_INDEX n = args->n;

    register CBLAS_INDEX i = 0;
    int use_prefetch = (n > CBLAS_PREFETCH_THRESHOLD);

    // Process 16 doubles per iteration (4x4 with AVX) for better ILP
    for (; i + 16 <= n; i += 16)
    {
        if (use_prefetch && i + CBLAS_PREFETCH_DISTANCE < n) {
            CBLAS_PREFETCH(&x[i + CBLAS_PREFETCH_DISTANCE], 0, 3);
            CBLAS_PREFETCH(&y[i + CBLAS_PREFETCH_DISTANCE], 1, 3);
        }

        // Load 16 doubles using 4 independent streams
        __m256d a0 = _mm256_loadu_pd(&x[i]);
        __m256d a1 = _mm256_loadu_pd(&x[i + 4]);
        __m256d a2 = _mm256_loadu_pd(&x[i + 8]);
        __m256d a3 = _mm256_loadu_pd(&x[i + 12]);

        // Store 16 doubles
        _mm256_storeu_pd(&y[i], a0);
        _mm256_storeu_pd(&y[i + 4], a1);
        _mm256_storeu_pd(&y[i + 8], a2);
        _mm256_storeu_pd(&y[i + 12], a3);
    }

    // Process remaining elements
    for (; i < n; i++)
        y[i] = x[i];
}

#endif
