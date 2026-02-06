//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"
#include "cblas_simd.h"

#if defined(__aarch64__) && defined(__ARM_NEON)

//------------------------------------------------------
// single-precision asum kernel incx == 1 (NEON)
//------------------------------------------------------
void cblas_sasum_k_noinc_neon(cblas_args_t* args)
{
    float *x = args->x;
    CBLAS_INDEX n = args->n;
    float *result = args->c;
    CBLAS_INDEX i = 0;

    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    int use_prefetch = (n > CBLAS_PREFETCH_THRESHOLD);

    // Process 16 elements at a time using 4 NEON registers
    for (; i + 16 <= n; i += 16)
    {
        if (use_prefetch && i + CBLAS_PREFETCH_DISTANCE < n) {
            CBLAS_PREFETCH(x + CBLAS_PREFETCH_DISTANCE, 0, 3);
        }

        float32x4_t a = vld1q_f32(x);
        float32x4_t b = vld1q_f32(x + 4);
        float32x4_t c = vld1q_f32(x + 8);
        float32x4_t d = vld1q_f32(x + 12);

        // Get absolute values
        a = vabsq_f32(a);
        b = vabsq_f32(b);
        c = vabsq_f32(c);
        d = vabsq_f32(d);

        // Accumulate
        sum_vec = vaddq_f32(sum_vec, a);
        sum_vec = vaddq_f32(sum_vec, b);
        sum_vec = vaddq_f32(sum_vec, c);
        sum_vec = vaddq_f32(sum_vec, d);

        x += 16;
    }

    // Horizontal sum of the vector
    float total = vaddvq_f32(sum_vec);

    // Handle remaining elements
    for (; i < n; i++)
    {
        total += fabsf(*x);
        x++;
    }

    *result = total;
}

//------------------------------------------------------
// double-precision asum kernel incx == 1 (NEON)
//------------------------------------------------------
void cblas_dasum_k_noinc_neon(cblas_args_t* args)
{
    double *x = args->x;
    CBLAS_INDEX n = args->n;
    double *result = args->c;
    CBLAS_INDEX i = 0;
    
    float64x2_t sum_vec = vdupq_n_f64(0.0);
    int use_prefetch = (n > CBLAS_PREFETCH_THRESHOLD);

    // Process 8 elements at a time using 4 NEON registers (2 doubles each)
    for (; i + 8 <= n; i += 8)
    {
        if (use_prefetch && i + CBLAS_PREFETCH_DISTANCE < n) {
            CBLAS_PREFETCH(x + CBLAS_PREFETCH_DISTANCE, 0, 3);
        }

        float64x2_t a = vld1q_f64(x);
        float64x2_t b = vld1q_f64(x + 2);
        float64x2_t c = vld1q_f64(x + 4);
        float64x2_t d = vld1q_f64(x + 6);

        // Get absolute values
        a = vabsq_f64(a);
        b = vabsq_f64(b);
        c = vabsq_f64(c);
        d = vabsq_f64(d);

        // Accumulate
        sum_vec = vaddq_f64(sum_vec, a);
        sum_vec = vaddq_f64(sum_vec, b);
        sum_vec = vaddq_f64(sum_vec, c);
        sum_vec = vaddq_f64(sum_vec, d);

        x += 8;
    }

    // Horizontal sum of the vector
    double total = vaddvq_f64(sum_vec);

    // Handle remaining elements
    for (; i < n; i++)
    {
        total += fabs(*x);
        x++;
    }

    *result = total;
}

#endif
