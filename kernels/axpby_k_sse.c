//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"
#include "cblas_simd.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)

//------------------------------------------------------
// single-precision axpby kernel incx == 1 && incy == 1 (SSE)
// y = alpha * x + beta * y
//------------------------------------------------------
void cblas_saxpby_k_noinc_sse(cblas_args_t* args)
{
    float* x = args->x;
    float* y = args->y;
    float alpha = *(float*)args->alpha;
    float beta = *(float*)args->beta;
    register CBLAS_INDEX n = args->n;
    register CBLAS_INDEX i = 0;

    __m128 alpha_vec = _mm_set1_ps(alpha);
    __m128 beta_vec = _mm_set1_ps(beta);
    
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
        
        // y = alpha*x + beta*y
        y0 = _mm_add_ps(_mm_mul_ps(alpha_vec, x0), _mm_mul_ps(beta_vec, y0));
        y1 = _mm_add_ps(_mm_mul_ps(alpha_vec, x1), _mm_mul_ps(beta_vec, y1));
        y2 = _mm_add_ps(_mm_mul_ps(alpha_vec, x2), _mm_mul_ps(beta_vec, y2));
        y3 = _mm_add_ps(_mm_mul_ps(alpha_vec, x3), _mm_mul_ps(beta_vec, y3));
        
        _mm_storeu_ps(y + i, y0);
        _mm_storeu_ps(y + i + 4, y1);
        _mm_storeu_ps(y + i + 8, y2);
        _mm_storeu_ps(y + i + 12, y3);
    }
    
    for (; i + 4 <= n; i += 4)
    {
        __m128 x_vec = _mm_loadu_ps(x + i);
        __m128 y_vec = _mm_loadu_ps(y + i);
        y_vec = _mm_add_ps(_mm_mul_ps(alpha_vec, x_vec), _mm_mul_ps(beta_vec, y_vec));
        _mm_storeu_ps(y + i, y_vec);
    }
    
    // Scalar cleanup
    for (; i < n; i++)
        y[i] = alpha * x[i] + beta * y[i];
}

//------------------------------------------------------
// double-precision axpby kernel incx == 1 && incy == 1 (SSE)
// y = alpha * x + beta * y
//------------------------------------------------------
void cblas_daxpby_k_noinc_sse(cblas_args_t* args)
{
    double* x = args->x;
    double* y = args->y;
    double alpha = *(double*)args->alpha;
    double beta = *(double*)args->beta;
    register CBLAS_INDEX n = args->n;
    register CBLAS_INDEX i = 0;

    __m128d alpha_vec = _mm_set1_pd(alpha);
    __m128d beta_vec = _mm_set1_pd(beta);
    
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
        
        y0 = _mm_add_pd(_mm_mul_pd(alpha_vec, x0), _mm_mul_pd(beta_vec, y0));
        y1 = _mm_add_pd(_mm_mul_pd(alpha_vec, x1), _mm_mul_pd(beta_vec, y1));
        y2 = _mm_add_pd(_mm_mul_pd(alpha_vec, x2), _mm_mul_pd(beta_vec, y2));
        y3 = _mm_add_pd(_mm_mul_pd(alpha_vec, x3), _mm_mul_pd(beta_vec, y3));
        
        _mm_storeu_pd(y + i, y0);
        _mm_storeu_pd(y + i + 2, y1);
        _mm_storeu_pd(y + i + 4, y2);
        _mm_storeu_pd(y + i + 6, y3);
    }
    
    for (; i + 2 <= n; i += 2)
    {
        __m128d x_vec = _mm_loadu_pd(x + i);
        __m128d y_vec = _mm_loadu_pd(y + i);
        y_vec = _mm_add_pd(_mm_mul_pd(alpha_vec, x_vec), _mm_mul_pd(beta_vec, y_vec));
        _mm_storeu_pd(y + i, y_vec);
    }
    
    // Scalar cleanup
    for (; i < n; i++)
        y[i] = alpha * x[i] + beta * y[i];
}

#endif
