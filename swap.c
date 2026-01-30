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
// single-precision swap kernel incx == 1 && incy == 1 (SSE/AVX)
//------------------------------------------------------
static void cblas_sswap_k_noinc_sse(float *x, float *y, CBLAS_INDEX n)
{
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
// double-precision swap kernel incx == 1 && incy == 1 (SSE/AVX)
//------------------------------------------------------
static void cblas_dswap_k_noinc_sse(double *x, double *y, CBLAS_INDEX n)
{
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

#if defined(__aarch64__) && defined(__ARM_NEON)

//------------------------------------------------------
// single-precision swap kernel incx == 1 && incy == 1 (NEON)
//------------------------------------------------------
static void cblas_sswap_k_noinc_neon(float *x, float *y, CBLAS_INDEX n)
{
    CBLAS_INDEX i = 0;
    
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
        
        vst1q_f32(x + i, y0);
        vst1q_f32(x + i + 4, y1);
        vst1q_f32(x + i + 8, y2);
        vst1q_f32(x + i + 12, y3);
        
        vst1q_f32(y + i, x0);
        vst1q_f32(y + i + 4, x1);
        vst1q_f32(y + i + 8, x2);
        vst1q_f32(y + i + 12, x3);
    }
    
    // Process 4 elements at a time
    for (; i + 4 <= n; i += 4)
    {
        float32x4_t x_vec = vld1q_f32(x + i);
        float32x4_t y_vec = vld1q_f32(y + i);
        vst1q_f32(x + i, y_vec);
        vst1q_f32(y + i, x_vec);
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
// double-precision swap kernel incx == 1 && incy == 1 (NEON)
//------------------------------------------------------
static void cblas_dswap_k_noinc_neon(double *x, double *y, CBLAS_INDEX n)
{
    CBLAS_INDEX i = 0;
    
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
        
        vst1q_f64(x + i, y0);
        vst1q_f64(x + i + 2, y1);
        vst1q_f64(x + i + 4, y2);
        vst1q_f64(x + i + 6, y3);
        
        vst1q_f64(y + i, x0);
        vst1q_f64(y + i + 2, x1);
        vst1q_f64(y + i + 4, x2);
        vst1q_f64(y + i + 6, x3);
    }
    
    // Process 2 elements at a time
    for (; i + 2 <= n; i += 2)
    {
        float64x2_t x_vec = vld1q_f64(x + i);
        float64x2_t y_vec = vld1q_f64(y + i);
        vst1q_f64(x + i, y_vec);
        vst1q_f64(y + i, x_vec);
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

//------------------------------------------------------
//
//------------------------------------------------------
static void cblas_sswap_k(cblas_args_t* args)
{
    float temp;
    float* x = args->x;
    float* y = args->y;

    for (CBLAS_INDEX i = 0; i < args->n; i++)
    {
        temp = *y;
        *y = *x;
        *x = temp;

        x += args->incx;
        y += args->incy;
    }
}

//------------------------------------------------------
//
//------------------------------------------------------
static void cblas_dswap_k(cblas_args_t* args)
{
    double temp;
    double* x = args->x;
    double* y = args->y;

    for (CBLAS_INDEX i = 0; i < args->n; i++)
    {
        temp = *y;
        *y = *x;
        *x = temp;

        x += args->incx;
        y += args->incy;
    }
}
//------------------------------------------------------
// Level-1 single-precision swap
//------------------------------------------------------
void cblas_sswap(CBLAS_INDEX n, float *x, CBLAS_INDEX incx, float *y, CBLAS_INDEX incy)
{
    CBLAS_VALIDATE_VEC2(n, x, incx, y, incy, );

    CBLAS_STATS_START();

#ifdef MT_ENABLED
    int mt_used = (n > CBLAS_MT_COPY) ? 1 : 0;
    
    if (mt_used)
    {
        cblas_level1_exec(sizeof(float), cblas_sswap_k, n, x, incx, y, incy, "SSWAP");
    }
    else
    {
        if (incx == 1 && incy == 1)
        {
#if defined(USE_SSE) && defined(USE_SIMD)
#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)
            cblas_sswap_k_noinc_sse(x, y, n);
#elif defined(__aarch64__) && defined(__ARM_NEON)
            cblas_sswap_k_noinc_neon(x, y, n);
#else
            // Fallback: scalar implementation
            float temp;
            for (CBLAS_INDEX i = 0; i < n; i++)
            {
                temp = y[i];
                y[i] = x[i];
                x[i] = temp;
            }
#endif
#else
            // Fallback: scalar implementation
            float temp;
            for (CBLAS_INDEX i = 0; i < n; i++)
            {
                temp = y[i];
                y[i] = x[i];
                x[i] = temp;
            }
#endif
        }
        else
        {
            // incx and/or incy are not 1
            float temp;
            for (CBLAS_INDEX i = 0; i < n; i++)
            {
                temp = *y;
                *y = *x;
                *x = temp;

                x += incx;
                y += incy;
            }
        }
    }
#else
    int mt_used = 0;
    if (incx == 1 && incy == 1)
    {
#if defined(USE_SSE) && defined(USE_SIMD)
#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)
        cblas_sswap_k_noinc_sse(x, y, n);
#elif defined(__aarch64__) && defined(__ARM_NEON)
        cblas_sswap_k_noinc_neon(x, y, n);
#else
        // Fallback: scalar implementation
        float temp;
        for (CBLAS_INDEX i = 0; i < n; i++)
        {
            temp = y[i];
            y[i] = x[i];
            x[i] = temp;
        }
#endif
#else
        // Fallback: scalar implementation
        float temp;
        for (CBLAS_INDEX i = 0; i < n; i++)
        {
            temp = y[i];
            y[i] = x[i];
            x[i] = temp;
        }
#endif
    }
    else
    {
        // incx and/or incy are not 1
        float temp;
        for (CBLAS_INDEX i = 0; i < n; i++)
        {
            temp = *y;
            *y = *x;
            *x = temp;

            x += incx;
            y += incy;
        }
    }
#endif

    CBLAS_STATS_END("sswap", n, mt_used);
}

//------------------------------------------------------
// Level-1 double-precision swap
//------------------------------------------------------
void cblas_dswap(CBLAS_INDEX n, double *x, CBLAS_INDEX incx, double *y, CBLAS_INDEX incy)
{
    CBLAS_VALIDATE_VEC2(n, x, incx, y, incy, );

    CBLAS_STATS_START();

#ifdef MT_ENABLED
    int mt_used = (n > CBLAS_MT_COPY) ? 1 : 0;
    
    if (mt_used)
    {
        cblas_level1_exec(sizeof(double), cblas_dswap_k, n, x, incx, y, incy, "DSWAP");
    }
    else
    {
        if (incx == 1 && incy == 1)
        {
#if defined(USE_SSE) && defined(USE_SIMD)
#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)
            cblas_dswap_k_noinc_sse(x, y, n);
#elif defined(__aarch64__) && defined(__ARM_NEON)
            cblas_dswap_k_noinc_neon(x, y, n);
#else
            // Fallback: scalar implementation
            double temp;
            for (CBLAS_INDEX i = 0; i < n; i++)
            {
                temp = y[i];
                y[i] = x[i];
                x[i] = temp;
            }
#endif
#else
            // Fallback: scalar implementation
            double temp;
            for (CBLAS_INDEX i = 0; i < n; i++)
            {
                temp = y[i];
                y[i] = x[i];
                x[i] = temp;
            }
#endif
        }
        else
        {
            // incx and/or incy are not 1
            double temp;
            for (CBLAS_INDEX i = 0; i < n; i++)
            {
                temp = *y;
                *y = *x;
                *x = temp;

                x += incx;
                y += incy;
            }
        }    
    }
#else
    int mt_used = 0;
    if (incx == 1 && incy == 1)
    {
#if defined(USE_SSE) && defined(USE_SIMD)
#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)
        cblas_dswap_k_noinc_sse(x, y, n);
#elif defined(__aarch64__) && defined(__ARM_NEON)
        cblas_dswap_k_noinc_neon(x, y, n);
#else
        // Fallback: scalar implementation
        double temp;
        for (CBLAS_INDEX i = 0; i < n; i++)
        {
            temp = y[i];
            y[i] = x[i];
            x[i] = temp;
        }
#endif
#else
        // Fallback: scalar implementation
        double temp;
        for (CBLAS_INDEX i = 0; i < n; i++)
        {
            temp = y[i];
            y[i] = x[i];
            x[i] = temp;
        }
#endif
    }
    else
    {
        // incx and/or incy are not 1
        double temp;
        for (CBLAS_INDEX i = 0; i < n; i++)
        {
            temp = *y;
            *y = *x;
            *x = temp;

            x += incx;
            y += incy;
        }    
    }
#endif

    CBLAS_STATS_END("dswap", n, mt_used);
}
