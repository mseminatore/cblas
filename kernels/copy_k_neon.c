//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"
#include "cblas_simd.h"

#if defined(__aarch64__) && defined(__ARM_NEON)

//------------------------------------------------------
// single-precision copy kernel incx == incy == 1
//------------------------------------------------------
void cblas_scopy_k_noinc_neon(cblas_args_t* args)
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