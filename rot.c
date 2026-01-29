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
// single-precision rot kernel incx == incy == 1 (SSE)
//------------------------------------------------------
static void cblas_srot_k_noinc_sse(float *x, float *y, CBLAS_INDEX n, float c, float s)
{
    __m128 c_vec = _mm_set1_ps(c);
    __m128 s_vec = _mm_set1_ps(s);
    CBLAS_INDEX i = 0;

    // Process 16 elements at a time using 4 SSE registers
    for (; i + 16 <= n; i += 16)
    {
        // Load x and y values
        __m128 x0 = _mm_loadu_ps(x);
        __m128 x1 = _mm_loadu_ps(x + 4);
        __m128 x2 = _mm_loadu_ps(x + 8);
        __m128 x3 = _mm_loadu_ps(x + 12);

        __m128 y0 = _mm_loadu_ps(y);
        __m128 y1 = _mm_loadu_ps(y + 4);
        __m128 y2 = _mm_loadu_ps(y + 8);
        __m128 y3 = _mm_loadu_ps(y + 12);

        // Compute temp = c * x + s * y
        __m128 temp0 = _mm_add_ps(_mm_mul_ps(c_vec, x0), _mm_mul_ps(s_vec, y0));
        __m128 temp1 = _mm_add_ps(_mm_mul_ps(c_vec, x1), _mm_mul_ps(s_vec, y1));
        __m128 temp2 = _mm_add_ps(_mm_mul_ps(c_vec, x2), _mm_mul_ps(s_vec, y2));
        __m128 temp3 = _mm_add_ps(_mm_mul_ps(c_vec, x3), _mm_mul_ps(s_vec, y3));

        // Compute y = c * y - s * x
        y0 = _mm_sub_ps(_mm_mul_ps(c_vec, y0), _mm_mul_ps(s_vec, x0));
        y1 = _mm_sub_ps(_mm_mul_ps(c_vec, y1), _mm_mul_ps(s_vec, x1));
        y2 = _mm_sub_ps(_mm_mul_ps(c_vec, y2), _mm_mul_ps(s_vec, x2));
        y3 = _mm_sub_ps(_mm_mul_ps(c_vec, y3), _mm_mul_ps(s_vec, x3));

        // Store results
        _mm_storeu_ps(x, temp0);
        _mm_storeu_ps(x + 4, temp1);
        _mm_storeu_ps(x + 8, temp2);
        _mm_storeu_ps(x + 12, temp3);

        _mm_storeu_ps(y, y0);
        _mm_storeu_ps(y + 4, y1);
        _mm_storeu_ps(y + 8, y2);
        _mm_storeu_ps(y + 12, y3);

        x += 16;
        y += 16;
    }

    // Handle remaining elements
    for (; i < n; i++)
    {
        float temp = c * *x + s * *y;
        *y = c * *y - s * *x;
        *x = temp;
        x++;
        y++;
    }
}

//------------------------------------------------------
// double-precision rot kernel incx == incy == 1 (SSE)
//------------------------------------------------------
static void cblas_drot_k_noinc_sse(double *x, double *y, CBLAS_INDEX n, double c, double s)
{
    __m128d c_vec = _mm_set1_pd(c);
    __m128d s_vec = _mm_set1_pd(s);
    CBLAS_INDEX i = 0;

    // Process 8 elements at a time using 4 SSE registers (2 doubles each)
    for (; i + 8 <= n; i += 8)
    {
        // Load x and y values
        __m128d x0 = _mm_loadu_pd(x);
        __m128d x1 = _mm_loadu_pd(x + 2);
        __m128d x2 = _mm_loadu_pd(x + 4);
        __m128d x3 = _mm_loadu_pd(x + 6);

        __m128d y0 = _mm_loadu_pd(y);
        __m128d y1 = _mm_loadu_pd(y + 2);
        __m128d y2 = _mm_loadu_pd(y + 4);
        __m128d y3 = _mm_loadu_pd(y + 6);

        // Compute temp = c * x + s * y
        __m128d temp0 = _mm_add_pd(_mm_mul_pd(c_vec, x0), _mm_mul_pd(s_vec, y0));
        __m128d temp1 = _mm_add_pd(_mm_mul_pd(c_vec, x1), _mm_mul_pd(s_vec, y1));
        __m128d temp2 = _mm_add_pd(_mm_mul_pd(c_vec, x2), _mm_mul_pd(s_vec, y2));
        __m128d temp3 = _mm_add_pd(_mm_mul_pd(c_vec, x3), _mm_mul_pd(s_vec, y3));

        // Compute y = c * y - s * x
        y0 = _mm_sub_pd(_mm_mul_pd(c_vec, y0), _mm_mul_pd(s_vec, x0));
        y1 = _mm_sub_pd(_mm_mul_pd(c_vec, y1), _mm_mul_pd(s_vec, x1));
        y2 = _mm_sub_pd(_mm_mul_pd(c_vec, y2), _mm_mul_pd(s_vec, x2));
        y3 = _mm_sub_pd(_mm_mul_pd(c_vec, y3), _mm_mul_pd(s_vec, x3));

        // Store results
        _mm_storeu_pd(x, temp0);
        _mm_storeu_pd(x + 2, temp1);
        _mm_storeu_pd(x + 4, temp2);
        _mm_storeu_pd(x + 6, temp3);

        _mm_storeu_pd(y, y0);
        _mm_storeu_pd(y + 2, y1);
        _mm_storeu_pd(y + 4, y2);
        _mm_storeu_pd(y + 6, y3);

        x += 8;
        y += 8;
    }

    // Handle remaining elements
    for (; i < n; i++)
    {
        double temp = c * *x + s * *y;
        *y = c * *y - s * *x;
        *x = temp;
        x++;
        y++;
    }
}

#endif

#if defined(__aarch64__) && defined(__ARM_NEON)

//------------------------------------------------------
// single-precision rot kernel incx == incy == 1 (NEON)
//------------------------------------------------------
static void cblas_srot_k_noinc_neon(float *x, float *y, CBLAS_INDEX n, float c, float s)
{
    float32x4_t c_vec = vdupq_n_f32(c);
    float32x4_t s_vec = vdupq_n_f32(s);
    CBLAS_INDEX i = 0;

    // Process 16 elements at a time using 4 NEON registers
    for (; i + 16 <= n; i += 16)
    {
        // Load x and y values
        float32x4_t x0 = vld1q_f32(x);
        float32x4_t x1 = vld1q_f32(x + 4);
        float32x4_t x2 = vld1q_f32(x + 8);
        float32x4_t x3 = vld1q_f32(x + 12);

        float32x4_t y0 = vld1q_f32(y);
        float32x4_t y1 = vld1q_f32(y + 4);
        float32x4_t y2 = vld1q_f32(y + 8);
        float32x4_t y3 = vld1q_f32(y + 12);

        // Compute temp = c * x + s * y using FMA
        float32x4_t temp0 = vmulq_f32(c_vec, x0);
        float32x4_t temp1 = vmulq_f32(c_vec, x1);
        float32x4_t temp2 = vmulq_f32(c_vec, x2);
        float32x4_t temp3 = vmulq_f32(c_vec, x3);

        temp0 = vfmaq_f32(temp0, s_vec, y0);
        temp1 = vfmaq_f32(temp1, s_vec, y1);
        temp2 = vfmaq_f32(temp2, s_vec, y2);
        temp3 = vfmaq_f32(temp3, s_vec, y3);

        // Compute y = c * y - s * x
        float32x4_t ny0 = vmulq_f32(c_vec, y0);
        float32x4_t ny1 = vmulq_f32(c_vec, y1);
        float32x4_t ny2 = vmulq_f32(c_vec, y2);
        float32x4_t ny3 = vmulq_f32(c_vec, y3);

        ny0 = vfmsq_f32(ny0, s_vec, x0);
        ny1 = vfmsq_f32(ny1, s_vec, x1);
        ny2 = vfmsq_f32(ny2, s_vec, x2);
        ny3 = vfmsq_f32(ny3, s_vec, x3);

        // Store results
        vst1q_f32(x, temp0);
        vst1q_f32(x + 4, temp1);
        vst1q_f32(x + 8, temp2);
        vst1q_f32(x + 12, temp3);

        vst1q_f32(y, ny0);
        vst1q_f32(y + 4, ny1);
        vst1q_f32(y + 8, ny2);
        vst1q_f32(y + 12, ny3);

        x += 16;
        y += 16;
    }

    // Handle remaining elements
    for (; i < n; i++)
    {
        float temp = c * *x + s * *y;
        *y = c * *y - s * *x;
        *x = temp;
        x++;
        y++;
    }
}

//------------------------------------------------------
// double-precision rot kernel incx == incy == 1 (NEON)
//------------------------------------------------------
static void cblas_drot_k_noinc_neon(double *x, double *y, CBLAS_INDEX n, double c, double s)
{
    float64x2_t c_vec = vdupq_n_f64(c);
    float64x2_t s_vec = vdupq_n_f64(s);
    CBLAS_INDEX i = 0;

    // Process 8 elements at a time using 4 NEON registers (2 doubles each)
    for (; i + 8 <= n; i += 8)
    {
        // Load x and y values
        float64x2_t x0 = vld1q_f64(x);
        float64x2_t x1 = vld1q_f64(x + 2);
        float64x2_t x2 = vld1q_f64(x + 4);
        float64x2_t x3 = vld1q_f64(x + 6);

        float64x2_t y0 = vld1q_f64(y);
        float64x2_t y1 = vld1q_f64(y + 2);
        float64x2_t y2 = vld1q_f64(y + 4);
        float64x2_t y3 = vld1q_f64(y + 6);

        // Compute temp = c * x + s * y using FMA
        float64x2_t temp0 = vmulq_f64(c_vec, x0);
        float64x2_t temp1 = vmulq_f64(c_vec, x1);
        float64x2_t temp2 = vmulq_f64(c_vec, x2);
        float64x2_t temp3 = vmulq_f64(c_vec, x3);

        temp0 = vfmaq_f64(temp0, s_vec, y0);
        temp1 = vfmaq_f64(temp1, s_vec, y1);
        temp2 = vfmaq_f64(temp2, s_vec, y2);
        temp3 = vfmaq_f64(temp3, s_vec, y3);

        // Compute y = c * y - s * x
        float64x2_t ny0 = vmulq_f64(c_vec, y0);
        float64x2_t ny1 = vmulq_f64(c_vec, y1);
        float64x2_t ny2 = vmulq_f64(c_vec, y2);
        float64x2_t ny3 = vmulq_f64(c_vec, y3);

        ny0 = vfmsq_f64(ny0, s_vec, x0);
        ny1 = vfmsq_f64(ny1, s_vec, x1);
        ny2 = vfmsq_f64(ny2, s_vec, x2);
        ny3 = vfmsq_f64(ny3, s_vec, x3);

        // Store results
        vst1q_f64(x, temp0);
        vst1q_f64(x + 2, temp1);
        vst1q_f64(x + 4, temp2);
        vst1q_f64(x + 6, temp3);

        vst1q_f64(y, ny0);
        vst1q_f64(y + 2, ny1);
        vst1q_f64(y + 4, ny2);
        vst1q_f64(y + 6, ny3);

        x += 8;
        y += 8;
    }

    // Handle remaining elements
    for (; i < n; i++)
    {
        double temp = c * *x + s * *y;
        *y = c * *y - s * *x;
        *x = temp;
        x++;
        y++;
    }
}

#endif

//------------------------------------------------------
// Level-1 single-precision generate rotation
//------------------------------------------------------
void cblas_srot(CBLAS_INDEX n, float *x, CBLAS_INDEX incx, float *y, CBLAS_INDEX incy, float c, float s)
{
#ifdef CBLAS_CHECK_INPUTS

#ifdef CBLAS_XERBLA_INPUTS
    int info = 0;
    if (n <= 0)
        info = 1;
    else if (!x)
        info = 2;
    else if (!y)
        info = 4;

    if (info) {
        XERBLA(info);
        return;
    }
#else
    if (n <= 0 || !x || !y)
    {
        assert(n > 0 && x && y);
        return;
    }
#endif  // CBLAS_XERBLA_INPUTS
#endif  // CBLAS_CHECK_INPUTS

    float temp;
    if (incx == 1 && incy == 1)
    {
#if defined(USE_SSE) && defined(USE_SIMD)
#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)
        cblas_srot_k_noinc_sse(x, y, n, c, s);
        return;
#elif defined(__aarch64__) && defined(__ARM_NEON)
        cblas_srot_k_noinc_neon(x, y, n, c, s);
        return;
#endif
#endif
        // Fallback scalar implementation with unrolling
        CBLAS_INDEX i = 0;

        for (; i + 4 <= n; i += 4)
        {
            temp = c * x[i] + s * y[i];
            y[i] = c * y[i] - s * x[i];
            x[i] = temp;

            temp = c * x[i+1] + s * y[i+1];
            y[i+1] = c * y[i+1] - s * x[i+1];
            x[i+1] = temp;

            temp = c * x[i+2] + s * y[i+2];
            y[i+2] = c * y[i+2] - s * x[i+2];
            x[i+2] = temp;

            temp = c * x[i+3] + s * y[i+3];
            y[i+3] = c * y[i+3] - s * x[i+3];
            x[i+3] = temp;
        }

        for (; i < n; i++)
        {
            temp = c * x[i] + s * y[i];
            y[i] = c * y[i] - s * x[i];
            x[i] = temp;
        }
    }
    else
    {
        for (CBLAS_INDEX i = 0; i < n; i++)
        {
            temp = c * *x + s * *y;
            *y = c * *y - s * *x;
            *x = temp;

            x += incx;
            y += incy;
        }
    }
}

//------------------------------------------------------
// Level-1 double-precision generate rotation
//------------------------------------------------------
void cblas_drot(CBLAS_INDEX n, double *x, CBLAS_INDEX incx, double *y, CBLAS_INDEX incy, double c, double s)
{
#ifdef CBLAS_CHECK_INPUTS

#ifdef CBLAS_XERBLA_INPUTS
    int info = 0;
    if (n <= 0)
        info = 1;
    else if (!x)
        info = 2;
    else if (!y)
        info = 4;

    if (info) {
        XERBLA(info);
        return;
    }
#else
    if (n <= 0 || !x || !y)
    {
        assert(n > 0 && x && y);
        return;
    }
#endif  // CBLAS_XERBLA_INPUTS
#endif  // CBLAS_CHECK_INPUTS

    double temp;

    if (incx == 1 && incy == 1)
    {
#if defined(USE_SSE) && defined(USE_SIMD)
#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)
        cblas_drot_k_noinc_sse(x, y, n, c, s);
        return;
#elif defined(__aarch64__) && defined(__ARM_NEON)
        cblas_drot_k_noinc_neon(x, y, n, c, s);
        return;
#endif
#endif
        // Fallback scalar implementation with unrolling
        CBLAS_INDEX i = 0;

        for (; i + 4 <= n; i += 4)
        {
            temp = c * x[i] + s * y[i];
            y[i] = c * y[i] - s * x[i];
            x[i] = temp;

            temp = c * x[i+1] + s * y[i+1];
            y[i+1] = c * y[i+1] - s * x[i+1];
            x[i+1] = temp;

            temp = c * x[i+2] + s * y[i+2];
            y[i+2] = c * y[i+2] - s * x[i+2];
            x[i+2] = temp;

            temp = c * x[i+3] + s * y[i+3];
            y[i+3] = c * y[i+3] - s * x[i+3];
            x[i+3] = temp;
        }

        for (; i < n; i++)
        {
            temp = c * x[i] + s * y[i];
            y[i] = c * y[i] - s * x[i];
            x[i] = temp;
        }
    }
    else
    {
        for (CBLAS_INDEX i = 0; i < n; i++)
        {
            temp = c * *x + s * *y;
            *y = c * *y - s * *x;
            *x = temp;
            
            x += incx;
            y += incy;
        }
    }
}
