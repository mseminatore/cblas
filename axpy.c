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
// single-precision axpy kernel incx == 1 && incy == 1 (SSE/AVX)
//------------------------------------------------------
static void cblas_saxpy_k_noinc_sse(float alpha, float *x, float *y, CBLAS_INDEX n)
{
    CBLAS_INDEX i = 0;

#if defined(__AVX2__)
    // AVX2 path: process 8 floats at a time
    __m256 alpha_vec = _mm256_set1_ps(alpha);
    
    for (; i + 8 <= n; i += 8)
    {
        __m256 x_vec = _mm256_loadu_ps(x + i);
        __m256 y_vec = _mm256_loadu_ps(y + i);
        
#if defined(USE_INTEL_FMA)
        y_vec = _mm256_fmadd_ps(alpha_vec, x_vec, y_vec);
#else
        y_vec = _mm256_add_ps(_mm256_mul_ps(alpha_vec, x_vec), y_vec);
#endif
        
        _mm256_storeu_ps(y + i, y_vec);
    }
#endif

    // SSE path: process 4 floats at a time
    __m128 alpha_vec_sse = _mm_set1_ps(alpha);
    
    for (; i + 4 <= n; i += 4)
    {
        __m128 x_vec = _mm_loadu_ps(x + i);
        __m128 y_vec = _mm_loadu_ps(y + i);
        y_vec = _mm_add_ps(_mm_mul_ps(alpha_vec_sse, x_vec), y_vec);
        _mm_storeu_ps(y + i, y_vec);
    }
    
    // Handle remaining elements
    for (; i < n; i++)
    {
        y[i] = alpha * x[i] + y[i];
    }
}

//------------------------------------------------------
// double-precision axpy kernel incx == 1 && incy == 1 (SSE/AVX)
//------------------------------------------------------
static void cblas_daxpy_k_noinc_sse(double alpha, double *x, double *y, CBLAS_INDEX n)
{
    CBLAS_INDEX i = 0;

#if defined(__AVX2__)
    // AVX2 path: process 4 doubles at a time
    __m256d alpha_vec = _mm256_set1_pd(alpha);
    
    for (; i + 4 <= n; i += 4)
    {
        __m256d x_vec = _mm256_loadu_pd(x + i);
        __m256d y_vec = _mm256_loadu_pd(y + i);
        
#if defined(USE_INTEL_FMA)
        y_vec = _mm256_fmadd_pd(alpha_vec, x_vec, y_vec);
#else
        y_vec = _mm256_add_pd(_mm256_mul_pd(alpha_vec, x_vec), y_vec);
#endif
        
        _mm256_storeu_pd(y + i, y_vec);
    }
#endif

    // SSE path: process 2 doubles at a time
    __m128d alpha_vec_sse = _mm_set1_pd(alpha);
    
    for (; i + 2 <= n; i += 2)
    {
        __m128d x_vec = _mm_loadu_pd(x + i);
        __m128d y_vec = _mm_loadu_pd(y + i);
        y_vec = _mm_add_pd(_mm_mul_pd(alpha_vec_sse, x_vec), y_vec);
        _mm_storeu_pd(y + i, y_vec);
    }
    
    // Handle remaining elements
    for (; i < n; i++)
    {
        y[i] = alpha * x[i] + y[i];
    }
}

#endif

#if defined(__aarch64__) && defined(__ARM_NEON)

//------------------------------------------------------
// single-precision axpy kernel incx == 1 && incy == 1 (NEON)
//------------------------------------------------------
static void cblas_saxpy_k_noinc_neon(float alpha, float *x, float *y, CBLAS_INDEX n)
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
        
        float32x4_t y0 = vld1q_f32(y + i);
        float32x4_t y1 = vld1q_f32(y + i + 4);
        float32x4_t y2 = vld1q_f32(y + i + 8);
        float32x4_t y3 = vld1q_f32(y + i + 12);
        
        y0 = vmlaq_f32(y0, alpha_vec, x0);
        y1 = vmlaq_f32(y1, alpha_vec, x1);
        y2 = vmlaq_f32(y2, alpha_vec, x2);
        y3 = vmlaq_f32(y3, alpha_vec, x3);
        
        vst1q_f32(y + i, y0);
        vst1q_f32(y + i + 4, y1);
        vst1q_f32(y + i + 8, y2);
        vst1q_f32(y + i + 12, y3);
    }
    
    // Process 4 elements at a time
    for (; i + 4 <= n; i += 4)
    {
        float32x4_t x_vec = vld1q_f32(x + i);
        float32x4_t y_vec = vld1q_f32(y + i);
        y_vec = vmlaq_f32(y_vec, alpha_vec, x_vec);
        vst1q_f32(y + i, y_vec);
    }
    
    // Handle remaining elements
    for (; i < n; i++)
    {
        y[i] = alpha * x[i] + y[i];
    }
}

//------------------------------------------------------
// double-precision axpy kernel incx == 1 && incy == 1 (NEON)
//------------------------------------------------------
static void cblas_daxpy_k_noinc_neon(double alpha, double *x, double *y, CBLAS_INDEX n)
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
        
        float64x2_t y0 = vld1q_f64(y + i);
        float64x2_t y1 = vld1q_f64(y + i + 2);
        float64x2_t y2 = vld1q_f64(y + i + 4);
        float64x2_t y3 = vld1q_f64(y + i + 6);
        
        y0 = vmlaq_f64(y0, alpha_vec, x0);
        y1 = vmlaq_f64(y1, alpha_vec, x1);
        y2 = vmlaq_f64(y2, alpha_vec, x2);
        y3 = vmlaq_f64(y3, alpha_vec, x3);
        
        vst1q_f64(y + i, y0);
        vst1q_f64(y + i + 2, y1);
        vst1q_f64(y + i + 4, y2);
        vst1q_f64(y + i + 6, y3);
    }
    
    // Process 2 elements at a time
    for (; i + 2 <= n; i += 2)
    {
        float64x2_t x_vec = vld1q_f64(x + i);
        float64x2_t y_vec = vld1q_f64(y + i);
        y_vec = vmlaq_f64(y_vec, alpha_vec, x_vec);
        vst1q_f64(y + i, y_vec);
    }
    
    // Handle remaining elements
    for (; i < n; i++)
    {
        y[i] = alpha * x[i] + y[i];
    }
}

#endif

//------------------------------------------------------
// Level-1 single-precision y = a * x + y
//------------------------------------------------------
void cblas_saxpy(CBLAS_INDEX n, float alpha, float *x, CBLAS_INDEX incx, float *y, CBLAS_INDEX incy)
{
    CBLAS_VALIDATE_AXPY(n, alpha, x, incx, y, incy, );

    CBLAS_STATS_START();

    int mt_used = 0;

    if (incx == 1 && incy == 1)
    {
#if defined(USE_SSE) && defined(USE_SIMD)
#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)
        cblas_saxpy_k_noinc_sse(alpha, x, y, n);
        CBLAS_STATS_END("saxpy", n, mt_used);
        return;
#elif defined(__aarch64__) && defined(__ARM_NEON)
        cblas_saxpy_k_noinc_neon(alpha, x, y, n);
        CBLAS_STATS_END("saxpy", n, mt_used);
        return;
#endif
#endif
        // Fallback scalar implementation
        if (alpha == 1.0f)
        {
            for (CBLAS_INDEX i = 0; i < n; i++)
            {
                y[i] = x[i] + y[i];
            }
        }
        else
        {
            for (CBLAS_INDEX i = 0; i < n; i++)
            {
                y[i] = alpha * x[i] + y[i];
            }
        }
    }
    else
    {
        // incx and/or incy are not 1
        if (alpha == 1.0f)
        {
            for (CBLAS_INDEX i = 0; i < n; i++)
            {
                *y = *x + *y;
                x += incx;
                y += incy;
            }
        }
        else
        {
            for (CBLAS_INDEX i = 0; i < n; i++)
            {
                *y = alpha * *x + *y;
                x += incx;
                y += incy;
            }
        }
    }

    CBLAS_STATS_END("saxpy", n, mt_used);
}

//------------------------------------------------------
// Level-1 double-precision y = a * x + y
//------------------------------------------------------
void cblas_daxpy(CBLAS_INDEX n, double alpha, double *x, CBLAS_INDEX incx, double *y, CBLAS_INDEX incy)
{
    CBLAS_VALIDATE_AXPY(n, alpha, x, incx, y, incy, );

    CBLAS_STATS_START();

    int mt_used = 0;

    if (incx == 1 && incy == 1)
    {
#if defined(USE_SSE) && defined(USE_SIMD)
#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)
        cblas_daxpy_k_noinc_sse(alpha, x, y, n);
        CBLAS_STATS_END("daxpy", n, mt_used);
        return;
#elif defined(__aarch64__) && defined(__ARM_NEON)
        cblas_daxpy_k_noinc_neon(alpha, x, y, n);
        CBLAS_STATS_END("daxpy", n, mt_used);
        return;
#endif
#endif
        // Fallback scalar implementation
        if (alpha == 1.0)
        {
            for (CBLAS_INDEX i = 0; i < n; i++)
            {
                y[i] = x[i] + y[i];
            }
        }
        else
        {
            for (CBLAS_INDEX i = 0; i < n; i++)
            {
                y[i] = alpha * x[i] + y[i];
            }
        }
    }
    else
    {
        // incx and/or incy are not 1
        if (alpha == 1.0)
        {
            for (CBLAS_INDEX i = 0; i < n; i++)
            {
                *y = *x + *y;
                x += incx;
                y += incy;
            }
        }
        else
        {
            for (CBLAS_INDEX i = 0; i < n; i++)
            {
                *y = alpha * *x + *y;
                x += incx;
                y += incy;
            }
        }
    }

    CBLAS_STATS_END("daxpy", n, mt_used);
}
