//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"
#include "cblas_simd.h"

#if defined(__aarch64__) && defined(__ARM_NEON)

//------------------------------------------------------
// single-precision nrm2 kernel incx == 1 (NEON)
//------------------------------------------------------
void cblas_snrm2_k_noinc_neon(cblas_args_t* args)
{
    float* x = args->x;
    float* result = args->c;
    register CBLAS_INDEX n = args->n;
    CBLAS_INDEX i = 0;

    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    int use_prefetch = (n > CBLAS_PREFETCH_THRESHOLD);

    // Process 16 elements at a time
    for (; i + 16 <= n; i += 16)
    {
        if (use_prefetch && i + CBLAS_PREFETCH_DISTANCE < n) {
            CBLAS_PREFETCH(x + i + CBLAS_PREFETCH_DISTANCE, 0, 3);
        }

        float32x4_t x0 = vld1q_f32(x + i);
        float32x4_t x1 = vld1q_f32(x + i + 4);
        float32x4_t x2 = vld1q_f32(x + i + 8);
        float32x4_t x3 = vld1q_f32(x + i + 12);

        // Accumulate squares using FMA
        sum_vec = vfmaq_f32(sum_vec, x0, x0);
        sum_vec = vfmaq_f32(sum_vec, x1, x1);
        sum_vec = vfmaq_f32(sum_vec, x2, x2);
        sum_vec = vfmaq_f32(sum_vec, x3, x3);
    }

    // Process 4 elements at a time
    for (; i + 4 <= n; i += 4)
    {
        float32x4_t x_vec = vld1q_f32(x + i);
        sum_vec = vfmaq_f32(sum_vec, x_vec, x_vec);
    }

    // Horizontal sum
    float total = vaddvq_f32(sum_vec);

    // Handle remaining elements
    for (; i < n; i++)
    {
        total += x[i] * x[i];
    }

    *result = sqrtf(total);
}

//------------------------------------------------------
// double-precision nrm2 kernel incx == 1 (NEON)
//------------------------------------------------------
void cblas_dnrm2_k_noinc_neon(cblas_args_t* args)
{
    double* x = args->x;
    double* result = args->c;
    register CBLAS_INDEX n = args->n;
    CBLAS_INDEX i = 0;

    float64x2_t sum_vec = vdupq_n_f64(0.0);
    int use_prefetch = (n > CBLAS_PREFETCH_THRESHOLD);

    // Process 8 elements at a time
    for (; i + 8 <= n; i += 8)
    {
        if (use_prefetch && i + CBLAS_PREFETCH_DISTANCE < n) {
            CBLAS_PREFETCH(x + i + CBLAS_PREFETCH_DISTANCE, 0, 3);
        }

        float64x2_t x0 = vld1q_f64(x + i);
        float64x2_t x1 = vld1q_f64(x + i + 2);
        float64x2_t x2 = vld1q_f64(x + i + 4);
        float64x2_t x3 = vld1q_f64(x + i + 6);

        // Accumulate squares using FMA
        sum_vec = vfmaq_f64(sum_vec, x0, x0);
        sum_vec = vfmaq_f64(sum_vec, x1, x1);
        sum_vec = vfmaq_f64(sum_vec, x2, x2);
        sum_vec = vfmaq_f64(sum_vec, x3, x3);
    }

    // Process 2 elements at a time
    for (; i + 2 <= n; i += 2)
    {
        float64x2_t x_vec = vld1q_f64(x + i);
        sum_vec = vfmaq_f64(sum_vec, x_vec, x_vec);
    }

    // Horizontal sum
    double total = vaddvq_f64(sum_vec);

    // Handle remaining elements
    for (; i < n; i++)
    {
        total += x[i] * x[i];
    }

    *result = sqrt(total);
}

#endif
