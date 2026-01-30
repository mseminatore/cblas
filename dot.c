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
// single-precision dot product kernel incx == 1 && incy == 1 (SSE/AVX)
//------------------------------------------------------
static void cblas_sdot_k_noinc_sse(float *x, float *y, CBLAS_INDEX n, float *result)
{
    CBLAS_INDEX i = 0;
    __m128 sum_vec = _mm_setzero_ps();

#if defined(__AVX2__)
    // AVX2 path: process 8 floats at a time
    __m256 sum_vec_avx = _mm256_setzero_ps();
    
    for (; i + 8 <= n; i += 8)
    {
        __m256 x_vec = _mm256_loadu_ps(x + i);
        __m256 y_vec = _mm256_loadu_ps(y + i);
        
#if defined(USE_INTEL_FMA)
        sum_vec_avx = _mm256_fmadd_ps(x_vec, y_vec, sum_vec_avx);
#else
        sum_vec_avx = _mm256_add_ps(sum_vec_avx, _mm256_mul_ps(x_vec, y_vec));
#endif
    }
    
    // Convert AVX to SSE for final reduction
    __m128 low = _mm256_castps256_ps128(sum_vec_avx);
    __m128 high = _mm256_extractf128_ps(sum_vec_avx, 1);
    sum_vec = _mm_add_ps(low, high);
#endif

    // SSE path: process 4 floats at a time
    for (; i + 4 <= n; i += 4)
    {
        __m128 x_vec = _mm_loadu_ps(x + i);
        __m128 y_vec = _mm_loadu_ps(y + i);
        sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(x_vec, y_vec));
    }
    
    // Horizontal sum of the vector
    float sum[4];
    _mm_storeu_ps(sum, sum_vec);
    float total = sum[0] + sum[1] + sum[2] + sum[3];
    
    // Handle remaining elements
    for (; i < n; i++)
    {
        total += x[i] * y[i];
    }
    
    *result = total;
}

//------------------------------------------------------
// double-precision dot product kernel incx == 1 && incy == 1 (SSE/AVX)
//------------------------------------------------------
static void cblas_ddot_k_noinc_sse(double *x, double *y, CBLAS_INDEX n, double *result)
{
    CBLAS_INDEX i = 0;
    __m128d sum_vec = _mm_setzero_pd();

#if defined(__AVX2__)
    // AVX2 path: process 4 doubles at a time
    __m256d sum_vec_avx = _mm256_setzero_pd();
    
    for (; i + 4 <= n; i += 4)
    {
        __m256d x_vec = _mm256_loadu_pd(x + i);
        __m256d y_vec = _mm256_loadu_pd(y + i);
        
#if defined(USE_INTEL_FMA)
        sum_vec_avx = _mm256_fmadd_pd(x_vec, y_vec, sum_vec_avx);
#else
        sum_vec_avx = _mm256_add_pd(sum_vec_avx, _mm256_mul_pd(x_vec, y_vec));
#endif
    }
    
    // Convert AVX to SSE for final reduction
    __m128d low = _mm256_castpd256_pd128(sum_vec_avx);
    __m128d high = _mm256_extractf128_pd(sum_vec_avx, 1);
    sum_vec = _mm_add_pd(low, high);
#endif

    // SSE path: process 2 doubles at a time
    for (; i + 2 <= n; i += 2)
    {
        __m128d x_vec = _mm_loadu_pd(x + i);
        __m128d y_vec = _mm_loadu_pd(y + i);
        sum_vec = _mm_add_pd(sum_vec, _mm_mul_pd(x_vec, y_vec));
    }
    
    // Horizontal sum of the vector
    double sum[2];
    _mm_storeu_pd(sum, sum_vec);
    double total = sum[0] + sum[1];
    
    // Handle remaining elements
    for (; i < n; i++)
    {
        total += x[i] * y[i];
    }
    
    *result = total;
}

#endif

#if defined(__aarch64__) && defined(__ARM_NEON)

//------------------------------------------------------
// single-precision dot product kernel incx == 1 && incy == 1 (NEON)
//------------------------------------------------------
static void cblas_sdot_k_noinc_neon(float *x, float *y, CBLAS_INDEX n, float *result)
{
    CBLAS_INDEX i = 0;
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    
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
        
        sum_vec = vmlaq_f32(sum_vec, x0, y0);
        sum_vec = vmlaq_f32(sum_vec, x1, y1);
        sum_vec = vmlaq_f32(sum_vec, x2, y2);
        sum_vec = vmlaq_f32(sum_vec, x3, y3);
    }
    
    // Process 4 elements at a time
    for (; i + 4 <= n; i += 4)
    {
        float32x4_t x_vec = vld1q_f32(x + i);
        float32x4_t y_vec = vld1q_f32(y + i);
        sum_vec = vmlaq_f32(sum_vec, x_vec, y_vec);
    }
    
    // Horizontal sum
    float total = vaddvq_f32(sum_vec);
    
    // Handle remaining elements
    for (; i < n; i++)
    {
        total += x[i] * y[i];
    }
    
    *result = total;
}

//------------------------------------------------------
// double-precision dot product kernel incx == 1 && incy == 1 (NEON)
//------------------------------------------------------
static void cblas_ddot_k_noinc_neon(double *x, double *y, CBLAS_INDEX n, double *result)
{
    CBLAS_INDEX i = 0;
    float64x2_t sum_vec = vdupq_n_f64(0.0);
    
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
        
        sum_vec = vmlaq_f64(sum_vec, x0, y0);
        sum_vec = vmlaq_f64(sum_vec, x1, y1);
        sum_vec = vmlaq_f64(sum_vec, x2, y2);
        sum_vec = vmlaq_f64(sum_vec, x3, y3);
    }
    
    // Process 2 elements at a time
    for (; i + 2 <= n; i += 2)
    {
        float64x2_t x_vec = vld1q_f64(x + i);
        float64x2_t y_vec = vld1q_f64(y + i);
        sum_vec = vmlaq_f64(sum_vec, x_vec, y_vec);
    }
    
    // Horizontal sum
    double total = vaddvq_f64(sum_vec);
    
    // Handle remaining elements
    for (; i < n; i++)
    {
        total += x[i] * y[i];
    }
    
    *result = total;
}

#endif

//------------------------------------------------------
// single-precision vector dot product kernel
//------------------------------------------------------
static void cblas_sdot_k(cblas_args_t* args)
{
    float sum = 0.0f;
    float* x = args->x;
    float* y = args->y;
    float* result = args->c;
    register CBLAS_INDEX incx = args->incx, incy = args->incy, n = args->n;

    for (CBLAS_INDEX i = 0; i < n; i++)
    {
        sum += *x * *y;
        x += incx;
        y += incy;
    }

    // set return value
    *result = sum;
}

//------------------------------------------------------
// single-precision vector dot product kernel inc=1
//------------------------------------------------------
static void cblas_sdot_k_noinc(cblas_args_t* args)
{
    float* x = args->x;
    float* y = args->y;
    float* result = args->c;
    register CBLAS_INDEX n = args->n;

#if defined(USE_SSE) && defined(USE_SIMD)
#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)
    cblas_sdot_k_noinc_sse(x, y, n, result);
    return;
#elif defined(__aarch64__) && defined(__ARM_NEON)
    cblas_sdot_k_noinc_neon(x, y, n, result);
    return;
#endif
#endif

    // Fallback: scalar implementation with 4-way unrolling
    CBLAS_INDEX i = 0;
    register float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    int use_prefetch = (n > CBLAS_PREFETCH_THRESHOLD);

    for (; i + 4 <= n; i += 4)
    {
        // Prefetch ahead for next iteration if vector is large
        if (use_prefetch && i + CBLAS_PREFETCH_DISTANCE < n) {
            __builtin_prefetch(x + CBLAS_PREFETCH_DISTANCE, 0, 0);
            __builtin_prefetch(y + CBLAS_PREFETCH_DISTANCE, 0, 0);
        }

        sum0 += *x * *y;
        sum1 += *(x + 1) * *(y + 1);
        sum2 += *(x + 2) * *(y + 2);
        sum3 += *(x + 3) * *(y + 3);

        x += 4;
        y += 4;
    }

    register float sum = sum0 + sum1 + sum2 + sum3;

    for (; i < n; i++)
    {
        sum += *x++ * *y++;
    }

    // set return value
    *result = sum;
}

//------------------------------------------------------
// Level-1 single-precision vector dot product
//------------------------------------------------------
float cblas_sdot(CBLAS_INDEX n, float *x, CBLAS_INDEX incx, float *y, CBLAS_INDEX incy)
{
    float sum = 0.0f;

    CBLAS_VALIDATE_VEC2(n, x, incx, y, incy, sum);

    CBLAS_STATS_START();

#if defined(MT_ENABLED)
    int mt_used = (n > CBLAS_MT_DOT) ? 1 : 0;
    
    if (mt_used)
    {
        float thread_partial_sums[MAX_THREADS];

        kernel_function kernel = cblas_sdot_k;

        // special case kernel for no increments
        if (incx == 1 && incy == 1)
            kernel = cblas_sdot_k_noinc;

        cblas_level1_exec_result(sizeof(float), kernel, n, x, incx, y, incy, thread_partial_sums, "SDOT");

        // accumulate results
        CBLAS_INDEX threads = cblas_get_num_threads();
        for (CBLAS_INDEX i = 0; i < threads; i++)
        {
            sum += thread_partial_sums[i];
        }
    }
    else
    {
        if (incx == 1 && incy == 1)
        {
#if defined(USE_SSE) && defined(USE_SIMD)
#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)
            cblas_sdot_k_noinc_sse(x, y, n, &sum);
#elif defined(__aarch64__) && defined(__ARM_NEON)
            cblas_sdot_k_noinc_neon(x, y, n, &sum);
#else
            // Fallback: scalar implementation with 4-way unrolling
            CBLAS_INDEX i = 0;
            register float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
            int use_prefetch = (n > CBLAS_PREFETCH_THRESHOLD);

            for (; i + 4 <= n; i += 4)
            {
                // Prefetch ahead for next iteration if vector is large
                if (use_prefetch && i + CBLAS_PREFETCH_DISTANCE < n) {
                    __builtin_prefetch(&x[i + CBLAS_PREFETCH_DISTANCE], 0, 0);
                    __builtin_prefetch(&y[i + CBLAS_PREFETCH_DISTANCE], 0, 0);
                }

                sum0 += x[i] * y[i];
                sum1 += x[i+1] * y[i+1];
                sum2 += x[i+2] * y[i+2];
                sum3 += x[i+3] * y[i+3];
            }

            sum = sum0 + sum1 + sum2 + sum3;

            for (; i < n; i++)
            {
                sum += x[i] * y[i];
            }
#endif
#else
            // Fallback: scalar implementation with 4-way unrolling
            CBLAS_INDEX i = 0;
            register float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
            int use_prefetch = (n > CBLAS_PREFETCH_THRESHOLD);

            for (; i + 4 <= n; i += 4)
            {
                // Prefetch ahead for next iteration if vector is large
                if (use_prefetch && i + CBLAS_PREFETCH_DISTANCE < n) {
                    __builtin_prefetch(&x[i + CBLAS_PREFETCH_DISTANCE], 0, 0);
                    __builtin_prefetch(&y[i + CBLAS_PREFETCH_DISTANCE], 0, 0);
                }

                sum0 += x[i] * y[i];
                sum1 += x[i+1] * y[i+1];
                sum2 += x[i+2] * y[i+2];
                sum3 += x[i+3] * y[i+3];
            }

            sum = sum0 + sum1 + sum2 + sum3;

            for (; i < n; i++)
            {
                sum += x[i] * y[i];
            }
#endif
        }
        else
        {
            // incx and/or incy are not 1
            for (CBLAS_INDEX i = 0; i < n; i++)
            {
                sum += *x * *y;
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
        cblas_sdot_k_noinc_sse(x, y, n, &sum);
#elif defined(__aarch64__) && defined(__ARM_NEON)
        cblas_sdot_k_noinc_neon(x, y, n, &sum);
#else
        // Fallback: scalar implementation with 4-way unrolling
        CBLAS_INDEX i = 0;
        register float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
        int use_prefetch = (n > CBLAS_PREFETCH_THRESHOLD);

        for (; i + 4 <= n; i += 4)
        {
            if (use_prefetch && i + CBLAS_PREFETCH_DISTANCE < n) {
                __builtin_prefetch(&x[i + CBLAS_PREFETCH_DISTANCE], 0, 0);
                __builtin_prefetch(&y[i + CBLAS_PREFETCH_DISTANCE], 0, 0);
            }

            sum0 += x[i] * y[i];
            sum1 += x[i+1] * y[i+1];
            sum2 += x[i+2] * y[i+2];
            sum3 += x[i+3] * y[i+3];
        }

        sum = sum0 + sum1 + sum2 + sum3;

        for (; i < n; i++)
        {
            sum += x[i] * y[i];
        }
#endif
#else
        // Fallback: scalar implementation with 4-way unrolling
        CBLAS_INDEX i = 0;
        register float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
        int use_prefetch = (n > CBLAS_PREFETCH_THRESHOLD);

        for (; i + 4 <= n; i += 4)
        {
            if (use_prefetch && i + CBLAS_PREFETCH_DISTANCE < n) {
                __builtin_prefetch(&x[i + CBLAS_PREFETCH_DISTANCE], 0, 0);
                __builtin_prefetch(&y[i + CBLAS_PREFETCH_DISTANCE], 0, 0);
            }

            sum0 += x[i] * y[i];
            sum1 += x[i+1] * y[i+1];
            sum2 += x[i+2] * y[i+2];
            sum3 += x[i+3] * y[i+3];
        }

        sum = sum0 + sum1 + sum2 + sum3;

        for (; i < n; i++)
        {
            sum += x[i] * y[i];
        }
#endif
    }
    else
    {
        // incx and/or incy are not 1
        for (CBLAS_INDEX i = 0; i < n; i++)
        {
            sum += *x * *y;
            x += incx;
            y += incy;
        }
    }
#endif

    CBLAS_STATS_END("sdot", n, mt_used);

    return sum;
}

//------------------------------------------------------
// Level-1 double-precision vector dot product
//------------------------------------------------------
double cblas_ddot(CBLAS_INDEX n, double *x, CBLAS_INDEX incx, double *y, CBLAS_INDEX incy)
{
    double sum = 0.0;

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
        return sum;
    }
#else
    if (n <= 0 || !x || !y)
    {
        assert(n > 0 && x && y);
        return 0.0;
    }
#endif  // CBLAS_XERBLA_INPUTS
#endif  // CBLAS_CHECK_INPUTS

    CBLAS_STATS_START();

    if (incx == 1 && incy == 1)
    {
#if defined(USE_SSE) && defined(USE_SIMD)
#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)
        cblas_ddot_k_noinc_sse(x, y, n, &sum);
#elif defined(__aarch64__) && defined(__ARM_NEON)
        cblas_ddot_k_noinc_neon(x, y, n, &sum);
#else
        // Fallback: scalar implementation
        int use_prefetch = (n > CBLAS_PREFETCH_THRESHOLD);
        
        for (CBLAS_INDEX i = 0; i < n; i++)
        {
            if (use_prefetch && i + CBLAS_PREFETCH_DISTANCE < n) {
                __builtin_prefetch(&x[i + CBLAS_PREFETCH_DISTANCE], 0, 0);
                __builtin_prefetch(&y[i + CBLAS_PREFETCH_DISTANCE], 0, 0);
            }
            sum += x[i] * y[i];
        }
#endif
#else
        // Fallback: scalar implementation
        int use_prefetch = (n > CBLAS_PREFETCH_THRESHOLD);
        
        for (CBLAS_INDEX i = 0; i < n; i++)
        {
            if (use_prefetch && i + CBLAS_PREFETCH_DISTANCE < n) {
                __builtin_prefetch(&x[i + CBLAS_PREFETCH_DISTANCE], 0, 0);
                __builtin_prefetch(&y[i + CBLAS_PREFETCH_DISTANCE], 0, 0);
            }
            sum += x[i] * y[i];
        }
#endif
    }
    else
    {
        // incx and/or incy are not 1
        for (CBLAS_INDEX i = 0; i < n; i++)
        {
            sum += *x * *y;
            x += incx;
            y += incy;
        }
    }

    CBLAS_STATS_END("ddot", n, 0);

    return sum;
}
