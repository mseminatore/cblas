//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)
#   include <immintrin.h>
#endif

#if defined(__aarch64__) && defined(__ARM_NEON)
#   include <arm_neon.h>
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)

//------------------------------------------------------
// single-precision scal kernel incx == 1 (SSE/AVX)
//------------------------------------------------------
static void cblas_sscal_k_noinc_sse(float alpha, float *x, CBLAS_INDEX n)
{
    CBLAS_INDEX i = 0;

#if defined(__AVX2__)
    // AVX2 path: process 8 floats at a time
    __m256 alpha_vec = _mm256_set1_ps(alpha);
    
    for (; i + 8 <= n; i += 8)
    {
        __m256 x_vec = _mm256_loadu_ps(x + i);
        x_vec = _mm256_mul_ps(alpha_vec, x_vec);
        _mm256_storeu_ps(x + i, x_vec);
    }
#endif

    // SSE path: process 4 floats at a time
    __m128 alpha_vec_sse = _mm_set1_ps(alpha);
    
    for (; i + 4 <= n; i += 4)
    {
        __m128 x_vec = _mm_loadu_ps(x + i);
        x_vec = _mm_mul_ps(alpha_vec_sse, x_vec);
        _mm_storeu_ps(x + i, x_vec);
    }
    
    // Handle remaining elements
    for (; i < n; i++)
    {
        x[i] = alpha * x[i];
    }
}

//------------------------------------------------------
// double-precision scal kernel incx == 1 (SSE/AVX)
//------------------------------------------------------
static void cblas_dscal_k_noinc_sse(double alpha, double *x, CBLAS_INDEX n)
{
    CBLAS_INDEX i = 0;

#if defined(__AVX2__)
    // AVX2 path: process 4 doubles at a time
    __m256d alpha_vec = _mm256_set1_pd(alpha);
    
    for (; i + 4 <= n; i += 4)
    {
        __m256d x_vec = _mm256_loadu_pd(x + i);
        x_vec = _mm256_mul_pd(alpha_vec, x_vec);
        _mm256_storeu_pd(x + i, x_vec);
    }
#endif

    // SSE path: process 2 doubles at a time
    __m128d alpha_vec_sse = _mm_set1_pd(alpha);
    
    for (; i + 2 <= n; i += 2)
    {
        __m128d x_vec = _mm_loadu_pd(x + i);
        x_vec = _mm_mul_pd(alpha_vec_sse, x_vec);
        _mm_storeu_pd(x + i, x_vec);
    }
    
    // Handle remaining elements
    for (; i < n; i++)
    {
        x[i] = alpha * x[i];
    }
}

#endif

#if defined(__aarch64__) && defined(__ARM_NEON)

//------------------------------------------------------
// single-precision scal kernel incx == 1 (NEON)
//------------------------------------------------------
static void cblas_sscal_k_noinc_neon(float alpha, float *x, CBLAS_INDEX n)
{
    CBLAS_INDEX i = 0;
    float32x4_t alpha_vec = vdupq_n_f32(alpha);
    
    // Process 16 elements at a time
    for (; i + 16 <= n; i += 16)
    {
        float32x4_t x0 = vld1q_f32(x + i);
        float32x4_t x1 = vld1q_f32(x + i + 4);
        float32x4_t x2 = vld1q_f32(x + i + 8);
        float32x4_t x3 = vld1q_f32(x + i + 12);
        
        x0 = vmulq_f32(alpha_vec, x0);
        x1 = vmulq_f32(alpha_vec, x1);
        x2 = vmulq_f32(alpha_vec, x2);
        x3 = vmulq_f32(alpha_vec, x3);
        
        vst1q_f32(x + i, x0);
        vst1q_f32(x + i + 4, x1);
        vst1q_f32(x + i + 8, x2);
        vst1q_f32(x + i + 12, x3);
    }
    
    // Process 4 elements at a time
    for (; i + 4 <= n; i += 4)
    {
        float32x4_t x_vec = vld1q_f32(x + i);
        x_vec = vmulq_f32(alpha_vec, x_vec);
        vst1q_f32(x + i, x_vec);
    }
    
    // Handle remaining elements
    for (; i < n; i++)
    {
        x[i] = alpha * x[i];
    }
}

//------------------------------------------------------
// double-precision scal kernel incx == 1 (NEON)
//------------------------------------------------------
static void cblas_dscal_k_noinc_neon(double alpha, double *x, CBLAS_INDEX n)
{
    CBLAS_INDEX i = 0;
    float64x2_t alpha_vec = vdupq_n_f64(alpha);
    
    // Process 8 elements at a time
    for (; i + 8 <= n; i += 8)
    {
        float64x2_t x0 = vld1q_f64(x + i);
        float64x2_t x1 = vld1q_f64(x + i + 2);
        float64x2_t x2 = vld1q_f64(x + i + 4);
        float64x2_t x3 = vld1q_f64(x + i + 6);
        
        x0 = vmulq_f64(alpha_vec, x0);
        x1 = vmulq_f64(alpha_vec, x1);
        x2 = vmulq_f64(alpha_vec, x2);
        x3 = vmulq_f64(alpha_vec, x3);
        
        vst1q_f64(x + i, x0);
        vst1q_f64(x + i + 2, x1);
        vst1q_f64(x + i + 4, x2);
        vst1q_f64(x + i + 6, x3);
    }
    
    // Process 2 elements at a time
    for (; i + 2 <= n; i += 2)
    {
        float64x2_t x_vec = vld1q_f64(x + i);
        x_vec = vmulq_f64(alpha_vec, x_vec);
        vst1q_f64(x + i, x_vec);
    }
    
    // Handle remaining elements
    for (; i < n; i++)
    {
        x[i] = alpha * x[i];
    }
}

#endif

//------------------------------------------------------
// Level-1 single-precision vector scale
//------------------------------------------------------
void cblas_sscal(CBLAS_INDEX n, float alpha, float *x, CBLAS_INDEX incx)
{
    CBLAS_VALIDATE_SCAL(n, alpha, x, incx, );

    CBLAS_STATS_START();

    int mt_used = 0;

    if (alpha == 1.0f)
    {
        // nothing to do!!
        return;
    }

    if (incx == 1)
    {
#if defined(USE_SSE) && defined(USE_SIMD)
#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)
        cblas_sscal_k_noinc_sse(alpha, x, n);
        CBLAS_STATS_END("sscal", n, mt_used);
        return;
#elif defined(__aarch64__) && defined(__ARM_NEON)
        cblas_sscal_k_noinc_neon(alpha, x, n);
        CBLAS_STATS_END("sscal", n, mt_used);
        return;
#endif
#endif
        // Fallback scalar implementation
        for (CBLAS_INDEX i = 0; i < n; i++)
        {
            x[i] = alpha * x[i];
        }
    }
    else
    {
        // incx != 1
        for (CBLAS_INDEX i = 0; i < n; i++)
        {
            *x = alpha * *x;
            x += incx;
        }
    }

    CBLAS_STATS_END("sscal", n, mt_used);
}

//------------------------------------------------------
// Level-1 single-precision vector scale
//------------------------------------------------------
void cblas_dscal(CBLAS_INDEX n, double alpha, double* x, CBLAS_INDEX incx)
{
#ifdef CBLAS_CHECK_INPUTS

#ifdef CBLAS_XERBLA_INPUTS
    int info = 0;
    if (n <= 0)
        info = 1;
    else if (!x)
        info = 3;
    else if (incx <= 0)
        info = 4;

    if (info) {
        XERBLA(info);
        return;
    }
#else
    if (n <= 0 || !x || incx <= 0)
    {
        assert(n >= 0 && x && incx > 0);
        return;
    }
#endif
#endif

    CBLAS_STATS_START();

    int mt_used = 0;

    if (alpha == 1.0)
    {
        // nothing to do!!
        return;
    }

    if (incx == 1)
    {
#if defined(USE_SSE) && defined(USE_SIMD)
#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)
        cblas_dscal_k_noinc_sse(alpha, x, n);
        CBLAS_STATS_END("dscal", n, mt_used);
        return;
#elif defined(__aarch64__) && defined(__ARM_NEON)
        cblas_dscal_k_noinc_neon(alpha, x, n);
        CBLAS_STATS_END("dscal", n, mt_used);
        return;
#endif
#endif
        // Fallback scalar implementation
        for (CBLAS_INDEX i = 0; i < n; i++)
        {
            x[i] = alpha * x[i];
        }
    }
    else
    {
        // incx != 1
        for (CBLAS_INDEX i = 0; i < n; i++)
        {
            *x = alpha * *x;
            x += incx;
        }
    }

    CBLAS_STATS_END("dscal", n, mt_used);
}
