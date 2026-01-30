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
// single-precision copy kernel incx == incy == 1
//------------------------------------------------------
CBLAS_UNUSED static void cblas_scopy_k_noinc_sse(cblas_args_t* args)
{
    float* x = args->x;
    float* y = args->y;
    register CBLAS_INDEX n = args->n;

    __m128 a, b, c, d;

    register CBLAS_INDEX i = 0;

    for (; i + 16 < n; i += 16)
    {
        a = _mm_load_ps(x);
        b = _mm_load_ps(x + 4);
        c = _mm_load_ps(x + 8);
        d = _mm_load_ps(x + 12);

        x += 16;

        _mm_store_ps(y, a);
        _mm_store_ps(y + 4, b);
        _mm_store_ps(y + 8, c);
        _mm_store_ps(y + 12, d);

        y += 16;
    }

    // TODO - possibly use switch with fall-through here?
    for (; i < n; i++)
        *y++ = *x++;
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
        a = vld1q_f32(x);
        b = vld1q_f32(x + 4);
        c = vld1q_f32(x + 8);
        d = vld1q_f32(x + 12);

        x += 16;

        vst1q_f32(y, a);
        vst1q_f32(y + 4, b);
        vst1q_f32(y + 8, c);
        vst1q_f32(y + 12, d);

        y += 16;
    }

    // TODO - possibly use switch with fall-through here?
    for (; i < n; i++)
        *y++ = *x++;
}

#endif

//------------------------------------------------------
// single-precision copy kernel incx == incy == 1
//------------------------------------------------------
static void cblas_scopy_k_noinc(cblas_args_t* args)
{
    float* x = args->x;
    float* y = args->y;
    register CBLAS_INDEX n = args->n;
    
    register CBLAS_INDEX i = 0;

    for (; i + 4 < n; i += 4)
    {
        *y = *x;
        *(y + 1) = *(x + 1);
        *(y + 2) = *(x + 2);
        *(y + 3) = *(x + 3);

        x += 4;
        y += 4;
    }

    // TODO - possibly use switch with fall-through here?
    for (; i < n; i++)
        *y++ = *x++;
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

//------------------------------------------------------
// single-precision copy kernel incx == incy == 1
//------------------------------------------------------
CBLAS_UNUSED static void cblas_dcopy_k_noinc(cblas_args_t* args)
{
    double *x = args->x;
    double *y = args->y;
    register CBLAS_INDEX n = args->n;

    register CBLAS_INDEX i = 0;

    for (; i + 4 < n; i += 4)
    {
        *y = *x;
        *(y + 1) = *(x + 1);
        *(y + 2) = *(x + 2);
        *(y + 3) = *(x + 3);

        x += 4;
        y += 4;
    }

    // TODO - possibly use switch with fall-through here?
    for (; i < n; i++)
        *y++ = *x++;
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
            kernel = cblas_scopy_k_noinc;

        cblas_level1_exec(sizeof(float), kernel, n, x, incx, y, incy, "SCOPY");
    }
    else
    {
        if (incx == 1 && incy == 1)
        {
            // TODO - unroll this loop
            if (n > CBLAS_PREFETCH_THRESHOLD)
            {
                // Large vector path with prefetching
                CBLAS_INDEX i = 0;
                for (; i < n; i++)
                {
                    if (i + CBLAS_PREFETCH_DISTANCE < n) {
                        __builtin_prefetch(x + CBLAS_PREFETCH_DISTANCE, 0, 0);
                        __builtin_prefetch(y + CBLAS_PREFETCH_DISTANCE, 1, 0);
                    }
                    *y++ = *x++;
                }
            }
            else
            {
                // Small vector path without prefetching
                for (CBLAS_INDEX i = 0; i < n; i++)
                {
                    *y++ = *x++;
                }
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
        // TODO - unroll this loop
        if (n > CBLAS_PREFETCH_THRESHOLD)
        {
            // Large vector path with prefetching
            CBLAS_INDEX i = 0;
            for (; i < n; i++)
            {
                if (i + CBLAS_PREFETCH_DISTANCE < n) {
                    __builtin_prefetch(x + CBLAS_PREFETCH_DISTANCE, 0, 0);
                    __builtin_prefetch(y + CBLAS_PREFETCH_DISTANCE, 1, 0);
                }
                *y++ = *x++;
            }
        }
        else
        {
            // Small vector path without prefetching
            for (CBLAS_INDEX i = 0; i < n; i++)
            {
                *y++ = *x++;
            }
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
        cblas_level1_exec(sizeof(double), cblas_dcopy_k, n, x, incx, y, incy, "DCOPY");
    }
    else
    {
        if (incx == 1 && incy == 1)
        {
            // TODO - unroll this loop
            if (n > CBLAS_PREFETCH_THRESHOLD)
            {
                // Large vector path with prefetching
                CBLAS_INDEX i = 0;
                for (; i < n; i++)
                {
                    if (i + CBLAS_PREFETCH_DISTANCE < n) {
                        __builtin_prefetch(x + CBLAS_PREFETCH_DISTANCE, 0, 0);
                        __builtin_prefetch(y + CBLAS_PREFETCH_DISTANCE, 1, 0);
                    }
                    *y++ = *x++;
                }
            }
            else
            {
                // Small vector path without prefetching
                for (CBLAS_INDEX i = 0; i < n; i++)
                {
                    *y++ = *x++;
                }
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
        // TODO - unroll this loop
        if (n > CBLAS_PREFETCH_THRESHOLD)
        {
            // Large vector path with prefetching
            CBLAS_INDEX i = 0;
            for (; i < n; i++)
            {
                if (i + CBLAS_PREFETCH_DISTANCE < n) {
                    __builtin_prefetch(x + CBLAS_PREFETCH_DISTANCE, 0, 0);
                    __builtin_prefetch(y + CBLAS_PREFETCH_DISTANCE, 1, 0);
                }
                *y++ = *x++;
            }
        }
        else
        {
            // Small vector path without prefetching
            for (CBLAS_INDEX i = 0; i < n; i++)
            {
                *y++ = *x++;
            }
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
