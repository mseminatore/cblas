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

    // Process 16 floats per iteration (4x4 with SSE) for better ILP
    for (; i + 16 <= n; i += 16)
    {
        __m128 x0 = _mm_loadu_ps(x + i);
        __m128 x1 = _mm_loadu_ps(x + i + 4);
        __m128 x2 = _mm_loadu_ps(x + i + 8);
        __m128 x3 = _mm_loadu_ps(x + i + 12);

        __m128 y0 = _mm_loadu_ps(y + i);
        __m128 y1 = _mm_loadu_ps(y + i + 4);
        __m128 y2 = _mm_loadu_ps(y + i + 8);
        __m128 y3 = _mm_loadu_ps(y + i + 12);

        _mm_storeu_ps(x + i, y0);
        _mm_storeu_ps(x + i + 4, y1);
        _mm_storeu_ps(x + i + 8, y2);
        _mm_storeu_ps(x + i + 12, y3);

        _mm_storeu_ps(y + i, x0);
        _mm_storeu_ps(y + i + 4, x1);
        _mm_storeu_ps(y + i + 8, x2);
        _mm_storeu_ps(y + i + 12, x3);
    }

    // Process 4 floats at a time
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

    // Process 8 doubles per iteration (4x2 with SSE) for better ILP
    for (; i + 8 <= n; i += 8)
    {
        __m128d x0 = _mm_loadu_pd(x + i);
        __m128d x1 = _mm_loadu_pd(x + i + 2);
        __m128d x2 = _mm_loadu_pd(x + i + 4);
        __m128d x3 = _mm_loadu_pd(x + i + 6);

        __m128d y0 = _mm_loadu_pd(y + i);
        __m128d y1 = _mm_loadu_pd(y + i + 2);
        __m128d y2 = _mm_loadu_pd(y + i + 4);
        __m128d y3 = _mm_loadu_pd(y + i + 6);

        _mm_storeu_pd(x + i, y0);
        _mm_storeu_pd(x + i + 2, y1);
        _mm_storeu_pd(x + i + 4, y2);
        _mm_storeu_pd(x + i + 6, y3);

        _mm_storeu_pd(y + i, x0);
        _mm_storeu_pd(y + i + 2, x1);
        _mm_storeu_pd(y + i + 4, x2);
        _mm_storeu_pd(y + i + 6, x3);
    }

    // Process 2 doubles at a time
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
