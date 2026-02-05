//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"
#include "cblas_simd.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)

//------------------------------------------------------
// single-precision scal kernel incx == 1 (SSE)
//------------------------------------------------------
void cblas_sscal_k_noinc_sse(cblas_args_t* args)
{
    float* x = args->x;
    float alpha = *(float*)args->alpha;
    register CBLAS_INDEX n = args->n;
    CBLAS_INDEX i = 0;

    __m128 alpha_vec = _mm_set1_ps(alpha);

    // Process 16 floats per iteration for better ILP
    for (; i + 16 <= n; i += 16)
    {
        __m128 x0 = _mm_loadu_ps(x + i);
        __m128 x1 = _mm_loadu_ps(x + i + 4);
        __m128 x2 = _mm_loadu_ps(x + i + 8);
        __m128 x3 = _mm_loadu_ps(x + i + 12);

        x0 = _mm_mul_ps(alpha_vec, x0);
        x1 = _mm_mul_ps(alpha_vec, x1);
        x2 = _mm_mul_ps(alpha_vec, x2);
        x3 = _mm_mul_ps(alpha_vec, x3);

        _mm_storeu_ps(x + i, x0);
        _mm_storeu_ps(x + i + 4, x1);
        _mm_storeu_ps(x + i + 8, x2);
        _mm_storeu_ps(x + i + 12, x3);
    }

    // Process 4 floats at a time
    for (; i + 4 <= n; i += 4)
    {
        __m128 x_vec = _mm_loadu_ps(x + i);
        x_vec = _mm_mul_ps(alpha_vec, x_vec);
        _mm_storeu_ps(x + i, x_vec);
    }

    // Handle remaining elements
    for (; i < n; i++)
    {
        x[i] = alpha * x[i];
    }
}

//------------------------------------------------------
// double-precision scal kernel incx == 1 (SSE)
//------------------------------------------------------
void cblas_dscal_k_noinc_sse(cblas_args_t* args)
{
    double* x = args->x;
    double alpha = *(double*)args->alpha;
    register CBLAS_INDEX n = args->n;
    CBLAS_INDEX i = 0;

    __m128d alpha_vec = _mm_set1_pd(alpha);

    // Process 8 doubles per iteration for better ILP
    for (; i + 8 <= n; i += 8)
    {
        __m128d x0 = _mm_loadu_pd(x + i);
        __m128d x1 = _mm_loadu_pd(x + i + 2);
        __m128d x2 = _mm_loadu_pd(x + i + 4);
        __m128d x3 = _mm_loadu_pd(x + i + 6);

        x0 = _mm_mul_pd(alpha_vec, x0);
        x1 = _mm_mul_pd(alpha_vec, x1);
        x2 = _mm_mul_pd(alpha_vec, x2);
        x3 = _mm_mul_pd(alpha_vec, x3);

        _mm_storeu_pd(x + i, x0);
        _mm_storeu_pd(x + i + 2, x1);
        _mm_storeu_pd(x + i + 4, x2);
        _mm_storeu_pd(x + i + 6, x3);
    }

    // Process 2 doubles at a time
    for (; i + 2 <= n; i += 2)
    {
        __m128d x_vec = _mm_loadu_pd(x + i);
        x_vec = _mm_mul_pd(alpha_vec, x_vec);
        _mm_storeu_pd(x + i, x_vec);
    }

    // Handle remaining elements
    for (; i < n; i++)
    {
        x[i] = alpha * x[i];
    }
}

#endif
