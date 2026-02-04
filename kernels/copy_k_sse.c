//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"
#include "cblas_simd.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)

//------------------------------------------------------
// single-precision copy kernel incx == incy == 1 (SSE)
//------------------------------------------------------
void cblas_scopy_k_noinc_sse(cblas_args_t* args)
{
    float* x = args->x;
    float* y = args->y;
    register CBLAS_INDEX n = args->n;

    register CBLAS_INDEX i = 0;

    // Process 16 floats per iteration (4x4 with SSE) for better ILP
    for (; i + 16 <= n; i += 16)
    {
        __m128 a0 = _mm_loadu_ps(&x[i]);
        __m128 a1 = _mm_loadu_ps(&x[i + 4]);
        __m128 a2 = _mm_loadu_ps(&x[i + 8]);
        __m128 a3 = _mm_loadu_ps(&x[i + 12]);

        _mm_storeu_ps(&y[i], a0);
        _mm_storeu_ps(&y[i + 4], a1);
        _mm_storeu_ps(&y[i + 8], a2);
        _mm_storeu_ps(&y[i + 12], a3);
    }

    // Process remaining elements
    for (; i < n; i++)
        y[i] = x[i];
}

//------------------------------------------------------
// double-precision copy kernel incx == incy == 1 (SSE)
//------------------------------------------------------
void cblas_dcopy_k_noinc_sse(cblas_args_t* args)
{
    double* x = args->x;
    double* y = args->y;
    register CBLAS_INDEX n = args->n;

    register CBLAS_INDEX i = 0;

    // Process 8 doubles per iteration (4x2 with SSE) for better ILP
    for (; i + 8 <= n; i += 8)
    {
        __m128d a0 = _mm_loadu_pd(&x[i]);
        __m128d a1 = _mm_loadu_pd(&x[i + 2]);
        __m128d a2 = _mm_loadu_pd(&x[i + 4]);
        __m128d a3 = _mm_loadu_pd(&x[i + 6]);

        _mm_storeu_pd(&y[i], a0);
        _mm_storeu_pd(&y[i + 2], a1);
        _mm_storeu_pd(&y[i + 4], a2);
        _mm_storeu_pd(&y[i + 6], a3);
    }

    // Process remaining elements
    for (; i < n; i++)
        y[i] = x[i];
}

#endif
