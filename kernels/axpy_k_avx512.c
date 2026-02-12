//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------
// AXPY kernel - AVX-512 implementation
// y = alpha * x + y using 512-bit SIMD operations
//------------------------------------------------------

#include "cblas.h"
#include "cblas_simd.h"

#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX512F__)

//------------------------------------------------------
// single-precision axpy kernel incx == 1 && incy == 1 (AVX-512)
// y = alpha * x + y
// Processes 64 floats per iteration (4x16) for maximum throughput
//------------------------------------------------------
void cblas_saxpy_k_noinc_avx512(cblas_args_t* args)
{
    float* x = args->x;
    float* y = args->y;
    float alpha = *(float*)args->alpha;
    register CBLAS_INDEX n = args->n;
    register CBLAS_INDEX i = 0;

    __m512 alpha_vec = _mm512_set1_ps(alpha);
    
    // Process 64 floats per iteration (4x16 with AVX-512) for better ILP
    for (; i + 64 <= n; i += 64)
    {
        __m512 x0 = _mm512_loadu_ps(x + i);
        __m512 x1 = _mm512_loadu_ps(x + i + 16);
        __m512 x2 = _mm512_loadu_ps(x + i + 32);
        __m512 x3 = _mm512_loadu_ps(x + i + 48);
        
        __m512 y0 = _mm512_loadu_ps(y + i);
        __m512 y1 = _mm512_loadu_ps(y + i + 16);
        __m512 y2 = _mm512_loadu_ps(y + i + 32);
        __m512 y3 = _mm512_loadu_ps(y + i + 48);
        
        // FMA: y = alpha*x + y in single instruction
        y0 = _mm512_fmadd_ps(alpha_vec, x0, y0);
        y1 = _mm512_fmadd_ps(alpha_vec, x1, y1);
        y2 = _mm512_fmadd_ps(alpha_vec, x2, y2);
        y3 = _mm512_fmadd_ps(alpha_vec, x3, y3);
        
        _mm512_storeu_ps(y + i, y0);
        _mm512_storeu_ps(y + i + 16, y1);
        _mm512_storeu_ps(y + i + 32, y2);
        _mm512_storeu_ps(y + i + 48, y3);
    }
    
    // Process 16 floats at a time
    for (; i + 16 <= n; i += 16)
    {
        __m512 x_vec = _mm512_loadu_ps(x + i);
        __m512 y_vec = _mm512_loadu_ps(y + i);
        y_vec = _mm512_fmadd_ps(alpha_vec, x_vec, y_vec);
        _mm512_storeu_ps(y + i, y_vec);
    }

    // AVX cleanup for remaining 8+ elements
    __m256 alpha_vec_avx = _mm256_set1_ps(alpha);
    for (; i + 8 <= n; i += 8)
    {
        __m256 x_vec = _mm256_loadu_ps(x + i);
        __m256 y_vec = _mm256_loadu_ps(y + i);
        y_vec = _mm256_fmadd_ps(alpha_vec_avx, x_vec, y_vec);
        _mm256_storeu_ps(y + i, y_vec);
    }
    
    // Scalar cleanup
    for (; i < n; i++)
        y[i] = alpha * x[i] + y[i];
}

//------------------------------------------------------
// double-precision axpy kernel incx == 1 && incy == 1 (AVX-512)
// y = alpha * x + y
// Processes 32 doubles per iteration (4x8) for maximum throughput
//------------------------------------------------------
void cblas_daxpy_k_noinc_avx512(cblas_args_t* args)
{
    double* x = args->x;
    double* y = args->y;
    double alpha = *(double*)args->alpha;
    register CBLAS_INDEX n = args->n;
    register CBLAS_INDEX i = 0;

    __m512d alpha_vec = _mm512_set1_pd(alpha);
    
    // Process 32 doubles per iteration (4x8 with AVX-512) for better ILP
    for (; i + 32 <= n; i += 32)
    {
        __m512d x0 = _mm512_loadu_pd(x + i);
        __m512d x1 = _mm512_loadu_pd(x + i + 8);
        __m512d x2 = _mm512_loadu_pd(x + i + 16);
        __m512d x3 = _mm512_loadu_pd(x + i + 24);
        
        __m512d y0 = _mm512_loadu_pd(y + i);
        __m512d y1 = _mm512_loadu_pd(y + i + 8);
        __m512d y2 = _mm512_loadu_pd(y + i + 16);
        __m512d y3 = _mm512_loadu_pd(y + i + 24);
        
        // FMA: y = alpha*x + y in single instruction
        y0 = _mm512_fmadd_pd(alpha_vec, x0, y0);
        y1 = _mm512_fmadd_pd(alpha_vec, x1, y1);
        y2 = _mm512_fmadd_pd(alpha_vec, x2, y2);
        y3 = _mm512_fmadd_pd(alpha_vec, x3, y3);
        
        _mm512_storeu_pd(y + i, y0);
        _mm512_storeu_pd(y + i + 8, y1);
        _mm512_storeu_pd(y + i + 16, y2);
        _mm512_storeu_pd(y + i + 24, y3);
    }
    
    // Process 8 doubles at a time
    for (; i + 8 <= n; i += 8)
    {
        __m512d x_vec = _mm512_loadu_pd(x + i);
        __m512d y_vec = _mm512_loadu_pd(y + i);
        y_vec = _mm512_fmadd_pd(alpha_vec, x_vec, y_vec);
        _mm512_storeu_pd(y + i, y_vec);
    }

    // AVX cleanup for remaining 4+ elements
    __m256d alpha_vec_avx = _mm256_set1_pd(alpha);
    for (; i + 4 <= n; i += 4)
    {
        __m256d x_vec = _mm256_loadu_pd(x + i);
        __m256d y_vec = _mm256_loadu_pd(y + i);
        y_vec = _mm256_fmadd_pd(alpha_vec_avx, x_vec, y_vec);
        _mm256_storeu_pd(y + i, y_vec);
    }
    
    // Scalar cleanup
    for (; i < n; i++)
        y[i] = alpha * x[i] + y[i];
}

#else

// Stub implementations for non-AVX512 builds
void cblas_saxpy_k_noinc_avx512(cblas_args_t* args) { (void)args; }
void cblas_daxpy_k_noinc_avx512(cblas_args_t* args) { (void)args; }

#endif
