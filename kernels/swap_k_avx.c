//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"
#include "cblas_simd.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)

//------------------------------------------------------
// single-precision swap kernel incx == 1 && incy == 1 (AVX)
//------------------------------------------------------
void cblas_sswap_k_noinc_avx(cblas_args_t* args)
{
    float* x = args->x;
    float* y = args->y;
    register CBLAS_INDEX n = args->n;
    CBLAS_INDEX i = 0;
    int use_prefetch = (n > CBLAS_PREFETCH_THRESHOLD);

    // Process 32 floats per iteration (4x8 with AVX) for better ILP
    for (; i + 32 <= n; i += 32)
    {
        if (use_prefetch && i + CBLAS_PREFETCH_DISTANCE < n) {
            CBLAS_PREFETCH(&x[i + CBLAS_PREFETCH_DISTANCE], 0, 3);
            CBLAS_PREFETCH(&y[i + CBLAS_PREFETCH_DISTANCE], 0, 3);
        }

        // Load 32 floats from x
        __m256 x0 = _mm256_loadu_ps(&x[i]);
        __m256 x1 = _mm256_loadu_ps(&x[i + 8]);
        __m256 x2 = _mm256_loadu_ps(&x[i + 16]);
        __m256 x3 = _mm256_loadu_ps(&x[i + 24]);

        // Load 32 floats from y
        __m256 y0 = _mm256_loadu_ps(&y[i]);
        __m256 y1 = _mm256_loadu_ps(&y[i + 8]);
        __m256 y2 = _mm256_loadu_ps(&y[i + 16]);
        __m256 y3 = _mm256_loadu_ps(&y[i + 24]);

        // Store y values to x
        _mm256_storeu_ps(&x[i], y0);
        _mm256_storeu_ps(&x[i + 8], y1);
        _mm256_storeu_ps(&x[i + 16], y2);
        _mm256_storeu_ps(&x[i + 24], y3);

        // Store x values to y
        _mm256_storeu_ps(&y[i], x0);
        _mm256_storeu_ps(&y[i + 8], x1);
        _mm256_storeu_ps(&y[i + 16], x2);
        _mm256_storeu_ps(&y[i + 24], x3);
    }

    // Handle remaining elements
    for (; i < n; i++)
    {
        float temp = y[i];
        y[i] = x[i];
        x[i] = temp;
    }
}

//------------------------------------------------------
// double-precision swap kernel incx == 1 && incy == 1 (AVX)
//------------------------------------------------------
void cblas_dswap_k_noinc_avx(cblas_args_t* args)
{
    double* x = args->x;
    double* y = args->y;
    register CBLAS_INDEX n = args->n;
    CBLAS_INDEX i = 0;
    int use_prefetch = (n > CBLAS_PREFETCH_THRESHOLD);

    // Process 16 doubles per iteration (4x4 with AVX) for better ILP
    for (; i + 16 <= n; i += 16)
    {
        if (use_prefetch && i + CBLAS_PREFETCH_DISTANCE < n) {
            CBLAS_PREFETCH(&x[i + CBLAS_PREFETCH_DISTANCE], 0, 3);
            CBLAS_PREFETCH(&y[i + CBLAS_PREFETCH_DISTANCE], 0, 3);
        }

        // Load 16 doubles from x
        __m256d x0 = _mm256_loadu_pd(&x[i]);
        __m256d x1 = _mm256_loadu_pd(&x[i + 4]);
        __m256d x2 = _mm256_loadu_pd(&x[i + 8]);
        __m256d x3 = _mm256_loadu_pd(&x[i + 12]);

        // Load 16 doubles from y
        __m256d y0 = _mm256_loadu_pd(&y[i]);
        __m256d y1 = _mm256_loadu_pd(&y[i + 4]);
        __m256d y2 = _mm256_loadu_pd(&y[i + 8]);
        __m256d y3 = _mm256_loadu_pd(&y[i + 12]);

        // Store y values to x
        _mm256_storeu_pd(&x[i], y0);
        _mm256_storeu_pd(&x[i + 4], y1);
        _mm256_storeu_pd(&x[i + 8], y2);
        _mm256_storeu_pd(&x[i + 12], y3);

        // Store x values to y
        _mm256_storeu_pd(&y[i], x0);
        _mm256_storeu_pd(&y[i + 4], x1);
        _mm256_storeu_pd(&y[i + 8], x2);
        _mm256_storeu_pd(&y[i + 12], x3);
    }

    // Handle remaining elements
    for (; i < n; i++)
    {
        double temp = y[i];
        y[i] = x[i];
        x[i] = temp;
    }
}

#endif
