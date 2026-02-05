//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"
#include "cblas_simd.h"

#if defined(__aarch64__) && defined(__ARM_NEON)

//------------------------------------------------------
// single-precision scal kernel incx == 1 (NEON)
//------------------------------------------------------
void cblas_sscal_k_noinc_neon(cblas_args_t* args)
{
    float* x = args->x;
    float alpha = *(float*)args->alpha;
    register CBLAS_INDEX n = args->n;
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
void cblas_dscal_k_noinc_neon(cblas_args_t* args)
{
    double* x = args->x;
    double alpha = *(double*)args->alpha;
    register CBLAS_INDEX n = args->n;
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
