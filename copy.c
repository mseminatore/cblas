//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"
#include "cblas_simd.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)

//------------------------------------------------------
// single-precision copy kernel incx == incy == 1
//------------------------------------------------------
CBLAS_UNUSED static void cblas_scopy_k_noinc_sse(cblas_args_t* args)
{
    float* x = args->x;
    float* y = args->y;
    register CBLAS_INDEX n = args->n;

    register CBLAS_INDEX i = 0;
    int use_prefetch = (n > CBLAS_PREFETCH_THRESHOLD);

    // Process 32 floats per iteration (4x8 with AVX) for better ILP
    // Use 4 independent load/store streams to hide latency
    for (; i + 32 <= n; i += 32)
    {
        if (use_prefetch && i + CBLAS_PREFETCH_DISTANCE < n) {
            CBLAS_PREFETCH(&x[i + CBLAS_PREFETCH_DISTANCE], 0, 3);
            CBLAS_PREFETCH(&y[i + CBLAS_PREFETCH_DISTANCE], 1, 3);
        }

        // Load 32 floats using 4 independent streams
        __m256 a0 = _mm256_loadu_ps(&x[i]);
        __m256 a1 = _mm256_loadu_ps(&x[i + 8]);
        __m256 a2 = _mm256_loadu_ps(&x[i + 16]);
        __m256 a3 = _mm256_loadu_ps(&x[i + 24]);

        // Store 32 floats
        _mm256_storeu_ps(&y[i], a0);
        _mm256_storeu_ps(&y[i + 8], a1);
        _mm256_storeu_ps(&y[i + 16], a2);
        _mm256_storeu_ps(&y[i + 24], a3);
    }

    // Process remaining elements
    for (; i < n; i++)
        y[i] = x[i];
}

#endif

#if defined(__aarch64__) && defined(__ARM_NEON)

//------------------------------------------------------
// single-precision copy kernel incx == incy == 1
//------------------------------------------------------
CBLAS_UNUSED static void cblas_scopy_k_noinc_neon(cblas_args_t* args)
{
    float* x = args->x;
    float* y = args->y;
    register CBLAS_INDEX n = args->n;

    float32x4_t a, b, c, d;

    register CBLAS_INDEX i = 0;

    for (; i + 16 < n; i += 16)
    {
        a = vld1q_f32(x + i);
        b = vld1q_f32(x + i + 4);
        c = vld1q_f32(x + i + 8);
        d = vld1q_f32(x + i + 12);

        vst1q_f32(y + i, a);
        vst1q_f32(y + i + 4, b);
        vst1q_f32(y + i + 8, c);
        vst1q_f32(y + i + 12, d);
    }

    for (; i < n; i++)
        y[i] = x[i];
}

#endif

//------------------------------------------------------
// single-precision copy kernel incx == incy == 1
//------------------------------------------------------
CBLAS_UNUSED static void cblas_scopy_k_noinc(cblas_args_t* args)
{
    float* x = args->x;
    float* y = args->y;
    register CBLAS_INDEX n = args->n;
    
    register CBLAS_INDEX i = 0;

    for (; i + 4 <= n; i += 4)
    {
        y[i] = x[i];
        y[i + 1] = x[i + 1];
        y[i + 2] = x[i + 2];
        y[i + 3] = x[i + 3];
    }

    for (; i < n; i++)
        y[i] = x[i];
}

//------------------------------------------------------
// single-precision copy kernel
//------------------------------------------------------
static void cblas_scopy_k(cblas_args_t *args)
{
    float *x = args->x;
    float *y = args->y;
    register CBLAS_INDEX incx = args->incx, incy = args->incy, n = args->n;
    
    for (CBLAS_INDEX i = 0; i < n; i++)
    {
        *y = *x;
        x += incx;
        y += incy;
    }
}

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)

//------------------------------------------------------
// double-precision copy kernel incx == incy == 1 (SSE)
//------------------------------------------------------
CBLAS_UNUSED static void cblas_dcopy_k_noinc_sse(cblas_args_t* args)
{
    double *x = args->x;
    double *y = args->y;
    register CBLAS_INDEX n = args->n;

    register CBLAS_INDEX i = 0;
    int use_prefetch = (n > CBLAS_PREFETCH_THRESHOLD);

    // Process 16 doubles per iteration (4x4 with AVX) for better ILP
    for (; i + 16 <= n; i += 16)
    {
        if (use_prefetch && i + CBLAS_PREFETCH_DISTANCE < n) {
            CBLAS_PREFETCH(&x[i + CBLAS_PREFETCH_DISTANCE], 0, 3);
            CBLAS_PREFETCH(&y[i + CBLAS_PREFETCH_DISTANCE], 1, 3);
        }

        // Load 16 doubles using 4 independent streams
        __m256d a0 = _mm256_loadu_pd(&x[i]);
        __m256d a1 = _mm256_loadu_pd(&x[i + 4]);
        __m256d a2 = _mm256_loadu_pd(&x[i + 8]);
        __m256d a3 = _mm256_loadu_pd(&x[i + 12]);

        // Store 16 doubles
        _mm256_storeu_pd(&y[i], a0);
        _mm256_storeu_pd(&y[i + 4], a1);
        _mm256_storeu_pd(&y[i + 8], a2);
        _mm256_storeu_pd(&y[i + 12], a3);
    }

    // Process remaining elements
    for (; i < n; i++)
        y[i] = x[i];
}

#endif

//------------------------------------------------------
// double-precision copy kernel incx == incy == 1
//------------------------------------------------------
CBLAS_UNUSED static void cblas_dcopy_k_noinc(cblas_args_t* args)
{
    double* x = args->x;
    double* y = args->y;
    register CBLAS_INDEX n = args->n;
    
    register CBLAS_INDEX i = 0;

    for (; i + 4 <= n; i += 4)
    {
        y[i] = x[i];
        y[i + 1] = x[i + 1];
        y[i + 2] = x[i + 2];
        y[i + 3] = x[i + 3];
    }

    for (; i < n; i++)
        y[i] = x[i];
}

//------------------------------------------------------
// double-precision copy kernel
//------------------------------------------------------
static void cblas_dcopy_k(cblas_args_t* args)
{
    double *x = args->x;
    double *y = args->y;

    for (CBLAS_INDEX i = 0; i < args->n; i++)
    {
        *y = *x;
        x += args->incx;
        y += args->incy;
    }
}

//------------------------------------------------------
// Level-1 single-precision copy
//------------------------------------------------------
void cblas_scopy(CBLAS_INDEX n, float *x, CBLAS_INDEX incx, float *y, CBLAS_INDEX incy)
{
    CBLAS_VALIDATE_VEC2(n, x, incx, y, incy, );

    CBLAS_STATS_START();

#ifdef MT_ENABLED
    int mt_used = (n > CBLAS_MT_COPY) ? 1 : 0;
    
    if (mt_used)
    {
        kernel_function kernel = cblas_scopy_k;
        if (incx == 1 && incy == 1)
#if defined(USE_SSE) && defined(USE_SIMD)
#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)
            kernel = cblas_scopy_k_noinc_sse;
#elif defined(__aarch64__) && defined(__ARM_NEON)
            kernel = cblas_scopy_k_noinc_neon;
#else
            kernel = cblas_scopy_k_noinc;
#endif
#else
            kernel = cblas_scopy_k_noinc;
#endif

        cblas_level1_exec(sizeof(float), kernel, n, x, incx, y, incy, "SCOPY");
    }
    else
    {
        if (incx == 1 && incy == 1)
        {
#if defined(USE_SSE) && defined(USE_SIMD)
#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)
            cblas_args_t args = {.n = n, .x = x, .y = y};
            cblas_scopy_k_noinc_sse(&args);
            CBLAS_STATS_END("scopy", n, mt_used);
            return;
#elif defined(__aarch64__) && defined(__ARM_NEON)
            cblas_args_t args = {.n = n, .x = x, .y = y};
            cblas_scopy_k_noinc_neon(&args);
            CBLAS_STATS_END("scopy", n, mt_used);
            return;
#endif
#endif
            // Fallback scalar implementation
            for (CBLAS_INDEX i = 0; i < n; i++)
            {
                *y++ = *x++;
            }
        }
        else
        {
            for (CBLAS_INDEX i = 0; i < n; i++)
            {
                *y = *x;
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
        cblas_args_t args = {.n = n, .x = x, .y = y};
        cblas_scopy_k_noinc_sse(&args);
        CBLAS_STATS_END("scopy", n, mt_used);
        return;
#elif defined(__aarch64__) && defined(__ARM_NEON)
        cblas_args_t args = {.n = n, .x = x, .y = y};
        cblas_scopy_k_noinc_neon(&args);
        CBLAS_STATS_END("scopy", n, mt_used);
        return;
#endif
#endif
        // Fallback scalar implementation
        for (CBLAS_INDEX i = 0; i < n; i++)
        {
            *y++ = *x++;
        }
    }
    else
    {
        for (CBLAS_INDEX i = 0; i < n; i++)
        {
            *y = *x;
            x += incx;
            y += incy;
        }
    }
#endif

    CBLAS_STATS_END("scopy", n, mt_used);
}

//------------------------------------------------------
// Level-1 double-precision copy
//------------------------------------------------------
void cblas_dcopy(CBLAS_INDEX n, double *x, CBLAS_INDEX incx, double *y, CBLAS_INDEX incy)
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
    if (n < 0 || !x || !y)
    {
        assert(n > 0 && x && y);
        return;
    }

#endif
#endif

    CBLAS_STATS_START();

#ifdef MT_ENABLED
    int mt_used = (n > CBLAS_MT_COPY) ? 1 : 0;
    
    if (mt_used)
    {
        kernel_function kernel = cblas_dcopy_k;
        if (incx == 1 && incy == 1)
#if defined(USE_SSE) && defined(USE_SIMD)
#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)
            kernel = cblas_dcopy_k_noinc_sse;
#else
            kernel = cblas_dcopy_k_noinc;
#endif
#else
            kernel = cblas_dcopy_k_noinc;
#endif

        cblas_level1_exec(sizeof(double), kernel, n, x, incx, y, incy, "DCOPY");
    }
    else
    {
        if (incx == 1 && incy == 1)
        {
#if defined(USE_SSE) && defined(USE_SIMD)
#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)
            cblas_args_t args = {.n = n, .x = x, .y = y};
            cblas_dcopy_k_noinc_sse(&args);
            CBLAS_STATS_END("dcopy", n, mt_used);
            return;
#endif
#endif
            // Fallback scalar implementation
            for (CBLAS_INDEX i = 0; i < n; i++)
            {
                *y++ = *x++;
            }
        }
        else
        {
            for (CBLAS_INDEX i = 0; i < n; i++)
            {
                *y = *x;
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
        cblas_args_t args = {.n = n, .x = x, .y = y};
        cblas_dcopy_k_noinc_sse(&args);
        CBLAS_STATS_END("dcopy", n, mt_used);
        return;
#endif
#endif
        // Fallback scalar implementation
        for (CBLAS_INDEX i = 0; i < n; i++)
        {
            *y++ = *x++;
        }
    }
    else
    {
        for (CBLAS_INDEX i = 0; i < n; i++)
        {
            *y = *x;
            x += incx;
            y += incy;
        }
    }
#endif

    CBLAS_STATS_END("dcopy", n, mt_used);
}
