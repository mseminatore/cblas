//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"
#include "cblas_simd.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)

//------------------------------------------------------
// single-precision axpby kernel incx == 1 && incy == 1 (AVX)
// y = alpha * x + beta * y
//------------------------------------------------------
void cblas_saxpby_k_noinc_avx(cblas_args_t* args)
{
    float* x = args->x;
    float* y = args->y;
    float alpha = *(float*)args->alpha;
    float beta = *(float*)args->beta;
    register CBLAS_INDEX n = args->n;
    register CBLAS_INDEX i = 0;

    __m256 alpha_vec = _mm256_set1_ps(alpha);
    __m256 beta_vec = _mm256_set1_ps(beta);
    int use_prefetch = (n > CBLAS_PREFETCH_THRESHOLD);
    
    // Process 32 floats per iteration (4x8 with AVX) for better ILP
    for (; i + 32 <= n; i += 32)
    {
        if (use_prefetch && i + CBLAS_PREFETCH_DISTANCE < n) {
            CBLAS_PREFETCH(x + i + CBLAS_PREFETCH_DISTANCE, 0, 3);
            CBLAS_PREFETCH(y + i + CBLAS_PREFETCH_DISTANCE, 1, 3);
        }
        
        __m256 x0 = _mm256_loadu_ps(x + i);
        __m256 x1 = _mm256_loadu_ps(x + i + 8);
        __m256 x2 = _mm256_loadu_ps(x + i + 16);
        __m256 x3 = _mm256_loadu_ps(x + i + 24);
        
        __m256 y0 = _mm256_loadu_ps(y + i);
        __m256 y1 = _mm256_loadu_ps(y + i + 8);
        __m256 y2 = _mm256_loadu_ps(y + i + 16);
        __m256 y3 = _mm256_loadu_ps(y + i + 24);
        
        // y = alpha*x + beta*y
        y0 = _mm256_add_ps(_mm256_mul_ps(alpha_vec, x0), _mm256_mul_ps(beta_vec, y0));
        y1 = _mm256_add_ps(_mm256_mul_ps(alpha_vec, x1), _mm256_mul_ps(beta_vec, y1));
        y2 = _mm256_add_ps(_mm256_mul_ps(alpha_vec, x2), _mm256_mul_ps(beta_vec, y2));
        y3 = _mm256_add_ps(_mm256_mul_ps(alpha_vec, x3), _mm256_mul_ps(beta_vec, y3));
        
        _mm256_storeu_ps(y + i, y0);
        _mm256_storeu_ps(y + i + 8, y1);
        _mm256_storeu_ps(y + i + 16, y2);
        _mm256_storeu_ps(y + i + 24, y3);
    }
    
    for (; i + 8 <= n; i += 8)
    {
        __m256 x_vec = _mm256_loadu_ps(x + i);
        __m256 y_vec = _mm256_loadu_ps(y + i);
        y_vec = _mm256_add_ps(_mm256_mul_ps(alpha_vec, x_vec), _mm256_mul_ps(beta_vec, y_vec));
        _mm256_storeu_ps(y + i, y_vec);
    }

    // SSE cleanup for remaining 4+ elements
    __m128 alpha_vec_sse = _mm_set1_ps(alpha);
    __m128 beta_vec_sse = _mm_set1_ps(beta);
    for (; i + 4 <= n; i += 4)
    {
        __m128 x_vec = _mm_loadu_ps(x + i);
        __m128 y_vec = _mm_loadu_ps(y + i);
        y_vec = _mm_add_ps(_mm_mul_ps(alpha_vec_sse, x_vec), _mm_mul_ps(beta_vec_sse, y_vec));
        _mm_storeu_ps(y + i, y_vec);
    }
    
    // Scalar cleanup
    for (; i < n; i++)
        y[i] = alpha * x[i] + beta * y[i];
}

//------------------------------------------------------
// double-precision axpby kernel incx == 1 && incy == 1 (AVX)
// y = alpha * x + beta * y
//------------------------------------------------------
void cblas_daxpby_k_noinc_avx(cblas_args_t* args)
{
    double* x = args->x;
    double* y = args->y;
    double alpha = *(double*)args->alpha;
    double beta = *(double*)args->beta;
    register CBLAS_INDEX n = args->n;
    register CBLAS_INDEX i = 0;

    __m256d alpha_vec = _mm256_set1_pd(alpha);
    __m256d beta_vec = _mm256_set1_pd(beta);
    int use_prefetch = (n > CBLAS_PREFETCH_THRESHOLD);
    
    // Process 16 doubles per iteration (4x4 with AVX) for better ILP
    for (; i + 16 <= n; i += 16)
    {
        if (use_prefetch && i + CBLAS_PREFETCH_DISTANCE < n) {
            CBLAS_PREFETCH(x + i + CBLAS_PREFETCH_DISTANCE, 0, 3);
            CBLAS_PREFETCH(y + i + CBLAS_PREFETCH_DISTANCE, 1, 3);
        }
        
        __m256d x0 = _mm256_loadu_pd(x + i);
        __m256d x1 = _mm256_loadu_pd(x + i + 4);
        __m256d x2 = _mm256_loadu_pd(x + i + 8);
        __m256d x3 = _mm256_loadu_pd(x + i + 12);
        
        __m256d y0 = _mm256_loadu_pd(y + i);
        __m256d y1 = _mm256_loadu_pd(y + i + 4);
        __m256d y2 = _mm256_loadu_pd(y + i + 8);
        __m256d y3 = _mm256_loadu_pd(y + i + 12);
        
        y0 = _mm256_add_pd(_mm256_mul_pd(alpha_vec, x0), _mm256_mul_pd(beta_vec, y0));
        y1 = _mm256_add_pd(_mm256_mul_pd(alpha_vec, x1), _mm256_mul_pd(beta_vec, y1));
        y2 = _mm256_add_pd(_mm256_mul_pd(alpha_vec, x2), _mm256_mul_pd(beta_vec, y2));
        y3 = _mm256_add_pd(_mm256_mul_pd(alpha_vec, x3), _mm256_mul_pd(beta_vec, y3));
        
        _mm256_storeu_pd(y + i, y0);
        _mm256_storeu_pd(y + i + 4, y1);
        _mm256_storeu_pd(y + i + 8, y2);
        _mm256_storeu_pd(y + i + 12, y3);
    }
    
    for (; i + 4 <= n; i += 4)
    {
        __m256d x_vec = _mm256_loadu_pd(x + i);
        __m256d y_vec = _mm256_loadu_pd(y + i);
        y_vec = _mm256_add_pd(_mm256_mul_pd(alpha_vec, x_vec), _mm256_mul_pd(beta_vec, y_vec));
        _mm256_storeu_pd(y + i, y_vec);
    }

    // SSE cleanup for remaining 2+ elements
    __m128d alpha_vec_sse = _mm_set1_pd(alpha);
    __m128d beta_vec_sse = _mm_set1_pd(beta);
    for (; i + 2 <= n; i += 2)
    {
        __m128d x_vec = _mm_loadu_pd(x + i);
        __m128d y_vec = _mm_loadu_pd(y + i);
        y_vec = _mm_add_pd(_mm_mul_pd(alpha_vec_sse, x_vec), _mm_mul_pd(beta_vec_sse, y_vec));
        _mm_storeu_pd(y + i, y_vec);
    }
    
    // Scalar cleanup
    for (; i < n; i++)
        y[i] = alpha * x[i] + beta * y[i];
}

#endif
