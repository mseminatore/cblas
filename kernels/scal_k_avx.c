//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"
#include "cblas_simd.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)

//------------------------------------------------------
// single-precision scal kernel incx == 1 (AVX)
//------------------------------------------------------
void cblas_sscal_k_noinc_avx(cblas_args_t* args)
{
    float* x = args->x;
    float alpha = *(float*)args->alpha;
    register CBLAS_INDEX n = args->n;
    CBLAS_INDEX i = 0;

    __m256 alpha_vec = _mm256_set1_ps(alpha);

    // Process 32 floats per iteration for better ILP
    for (; i + 32 <= n; i += 32)
    {
        __m256 x0 = _mm256_loadu_ps(x + i);
        __m256 x1 = _mm256_loadu_ps(x + i + 8);
        __m256 x2 = _mm256_loadu_ps(x + i + 16);
        __m256 x3 = _mm256_loadu_ps(x + i + 24);

        x0 = _mm256_mul_ps(alpha_vec, x0);
        x1 = _mm256_mul_ps(alpha_vec, x1);
        x2 = _mm256_mul_ps(alpha_vec, x2);
        x3 = _mm256_mul_ps(alpha_vec, x3);

        _mm256_storeu_ps(x + i, x0);
        _mm256_storeu_ps(x + i + 8, x1);
        _mm256_storeu_ps(x + i + 16, x2);
        _mm256_storeu_ps(x + i + 24, x3);
    }

    // Process 8 floats at a time
    for (; i + 8 <= n; i += 8)
    {
        __m256 x_vec = _mm256_loadu_ps(x + i);
        x_vec = _mm256_mul_ps(alpha_vec, x_vec);
        _mm256_storeu_ps(x + i, x_vec);
    }

    // Handle remaining elements
    for (; i < n; i++)
    {
        x[i] = alpha * x[i];
    }
}

//------------------------------------------------------
// double-precision scal kernel incx == 1 (AVX)
//------------------------------------------------------
void cblas_dscal_k_noinc_avx(cblas_args_t* args)
{
    double* x = args->x;
    double alpha = *(double*)args->alpha;
    register CBLAS_INDEX n = args->n;
    CBLAS_INDEX i = 0;

    __m256d alpha_vec = _mm256_set1_pd(alpha);

    // Process 16 doubles per iteration for better ILP
    for (; i + 16 <= n; i += 16)
    {
        __m256d x0 = _mm256_loadu_pd(x + i);
        __m256d x1 = _mm256_loadu_pd(x + i + 4);
        __m256d x2 = _mm256_loadu_pd(x + i + 8);
        __m256d x3 = _mm256_loadu_pd(x + i + 12);

        x0 = _mm256_mul_pd(alpha_vec, x0);
        x1 = _mm256_mul_pd(alpha_vec, x1);
        x2 = _mm256_mul_pd(alpha_vec, x2);
        x3 = _mm256_mul_pd(alpha_vec, x3);

        _mm256_storeu_pd(x + i, x0);
        _mm256_storeu_pd(x + i + 4, x1);
        _mm256_storeu_pd(x + i + 8, x2);
        _mm256_storeu_pd(x + i + 12, x3);
    }

    // Process 4 doubles at a time
    for (; i + 4 <= n; i += 4)
    {
        __m256d x_vec = _mm256_loadu_pd(x + i);
        x_vec = _mm256_mul_pd(alpha_vec, x_vec);
        _mm256_storeu_pd(x + i, x_vec);
    }

    // Handle remaining elements
    for (; i < n; i++)
    {
        x[i] = alpha * x[i];
    }
}

#endif
