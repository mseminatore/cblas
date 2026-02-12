//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------
// SCAL kernel - AVX-512 implementation
// x = alpha * x using 512-bit SIMD operations
//------------------------------------------------------

#include "cblas.h"
#include "cblas_simd.h"

#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX512F__)

//------------------------------------------------------
// single-precision scal kernel incx == 1 (AVX-512)
// Processes 64 floats per iteration (4x16) for maximum throughput
//------------------------------------------------------
void cblas_sscal_k_noinc_avx512(cblas_args_t* args)
{
    float* x = args->x;
    float alpha = *(float*)args->alpha;
    register CBLAS_INDEX n = args->n;
    CBLAS_INDEX i = 0;

    __m512 alpha_vec = _mm512_set1_ps(alpha);

    // Process 64 floats per iteration for better ILP
    for (; i + 64 <= n; i += 64)
    {
        __m512 x0 = _mm512_loadu_ps(x + i);
        __m512 x1 = _mm512_loadu_ps(x + i + 16);
        __m512 x2 = _mm512_loadu_ps(x + i + 32);
        __m512 x3 = _mm512_loadu_ps(x + i + 48);

        x0 = _mm512_mul_ps(alpha_vec, x0);
        x1 = _mm512_mul_ps(alpha_vec, x1);
        x2 = _mm512_mul_ps(alpha_vec, x2);
        x3 = _mm512_mul_ps(alpha_vec, x3);

        _mm512_storeu_ps(x + i, x0);
        _mm512_storeu_ps(x + i + 16, x1);
        _mm512_storeu_ps(x + i + 32, x2);
        _mm512_storeu_ps(x + i + 48, x3);
    }

    // Process 16 floats at a time
    for (; i + 16 <= n; i += 16)
    {
        __m512 x_vec = _mm512_loadu_ps(x + i);
        x_vec = _mm512_mul_ps(alpha_vec, x_vec);
        _mm512_storeu_ps(x + i, x_vec);
    }

    // AVX cleanup for remaining 8+ elements
    __m256 alpha_vec_avx = _mm256_set1_ps(alpha);
    for (; i + 8 <= n; i += 8)
    {
        __m256 x_vec = _mm256_loadu_ps(x + i);
        x_vec = _mm256_mul_ps(alpha_vec_avx, x_vec);
        _mm256_storeu_ps(x + i, x_vec);
    }

    // Handle remaining elements
    for (; i < n; i++)
    {
        x[i] = alpha * x[i];
    }
}

//------------------------------------------------------
// double-precision scal kernel incx == 1 (AVX-512)
// Processes 32 doubles per iteration (4x8) for maximum throughput
//------------------------------------------------------
void cblas_dscal_k_noinc_avx512(cblas_args_t* args)
{
    double* x = args->x;
    double alpha = *(double*)args->alpha;
    register CBLAS_INDEX n = args->n;
    CBLAS_INDEX i = 0;

    __m512d alpha_vec = _mm512_set1_pd(alpha);

    // Process 32 doubles per iteration for better ILP
    for (; i + 32 <= n; i += 32)
    {
        __m512d x0 = _mm512_loadu_pd(x + i);
        __m512d x1 = _mm512_loadu_pd(x + i + 8);
        __m512d x2 = _mm512_loadu_pd(x + i + 16);
        __m512d x3 = _mm512_loadu_pd(x + i + 24);

        x0 = _mm512_mul_pd(alpha_vec, x0);
        x1 = _mm512_mul_pd(alpha_vec, x1);
        x2 = _mm512_mul_pd(alpha_vec, x2);
        x3 = _mm512_mul_pd(alpha_vec, x3);

        _mm512_storeu_pd(x + i, x0);
        _mm512_storeu_pd(x + i + 8, x1);
        _mm512_storeu_pd(x + i + 16, x2);
        _mm512_storeu_pd(x + i + 24, x3);
    }

    // Process 8 doubles at a time
    for (; i + 8 <= n; i += 8)
    {
        __m512d x_vec = _mm512_loadu_pd(x + i);
        x_vec = _mm512_mul_pd(alpha_vec, x_vec);
        _mm512_storeu_pd(x + i, x_vec);
    }

    // AVX cleanup for remaining 4+ elements
    __m256d alpha_vec_avx = _mm256_set1_pd(alpha);
    for (; i + 4 <= n; i += 4)
    {
        __m256d x_vec = _mm256_loadu_pd(x + i);
        x_vec = _mm256_mul_pd(alpha_vec_avx, x_vec);
        _mm256_storeu_pd(x + i, x_vec);
    }

    // Handle remaining elements
    for (; i < n; i++)
    {
        x[i] = alpha * x[i];
    }
}

#else

// Stub implementations for non-AVX512 builds
void cblas_sscal_k_noinc_avx512(cblas_args_t* args) { (void)args; }
void cblas_dscal_k_noinc_avx512(cblas_args_t* args) { (void)args; }

#endif
