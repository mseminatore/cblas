//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"
#include "cblas_simd.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)

//------------------------------------------------------
// single-precision swap kernel incx == 1 && incy == 1 (SSE)
//------------------------------------------------------
void cblas_sswap_k_noinc_sse(cblas_args_t* args)
{
    float* x = args->x;
    float* y = args->y;
    register CBLAS_INDEX n = args->n;
    CBLAS_INDEX i = 0;

#if defined(__AVX2__)
    // AVX2 path: process 8 floats at a time
    for (; i + 8 <= n; i += 8)
    {
        __m256 x_vec = _mm256_loadu_ps(x + i);
        __m256 y_vec = _mm256_loadu_ps(y + i);
        _mm256_storeu_ps(x + i, y_vec);
        _mm256_storeu_ps(y + i, x_vec);
    }
#endif

    // SSE path: process 4 floats at a time
    for (; i + 4 <= n; i += 4)
    {
        __m128 x_vec = _mm_loadu_ps(x + i);
        __m128 y_vec = _mm_loadu_ps(y + i);
        _mm_storeu_ps(x + i, y_vec);
        _mm_storeu_ps(y + i, x_vec);
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
// double-precision swap kernel incx == 1 && incy == 1 (SSE)
//------------------------------------------------------
void cblas_dswap_k_noinc_sse(cblas_args_t* args)
{
    double* x = args->x;
    double* y = args->y;
    register CBLAS_INDEX n = args->n;
    CBLAS_INDEX i = 0;

#if defined(__AVX2__)
    // AVX2 path: process 4 doubles at a time
    for (; i + 4 <= n; i += 4)
    {
        __m256d x_vec = _mm256_loadu_pd(x + i);
        __m256d y_vec = _mm256_loadu_pd(y + i);
        _mm256_storeu_pd(x + i, y_vec);
        _mm256_storeu_pd(y + i, x_vec);
    }
#endif

    // SSE path: process 2 doubles at a time
    for (; i + 2 <= n; i += 2)
    {
        __m128d x_vec = _mm_loadu_pd(x + i);
        __m128d y_vec = _mm_loadu_pd(y + i);
        _mm_storeu_pd(x + i, y_vec);
        _mm_storeu_pd(y + i, x_vec);
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
