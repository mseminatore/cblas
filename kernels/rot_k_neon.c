//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"
#include "cblas_simd.h"

#if defined(__aarch64__) && defined(__ARM_NEON)

//------------------------------------------------------
// single-precision rot kernel incx == incy == 1 (NEON)
//------------------------------------------------------
void cblas_srot_k_noinc_neon(cblas_args_t* args)
{
    float* x = args->x;
    float* y = args->y;
    register CBLAS_INDEX n = args->n;
    float c = *(float*)args->alpha;
    float s = *(float*)args->beta;

    float32x4_t c_vec = vdupq_n_f32(c);
    float32x4_t s_vec = vdupq_n_f32(s);
    CBLAS_INDEX i = 0;

    // Process 16 elements at a time using 4 NEON registers
    for (; i + 16 <= n; i += 16)
    {
        // Load x and y values
        float32x4_t x0 = vld1q_f32(x + i);
        float32x4_t x1 = vld1q_f32(x + i + 4);
        float32x4_t x2 = vld1q_f32(x + i + 8);
        float32x4_t x3 = vld1q_f32(x + i + 12);

        float32x4_t y0 = vld1q_f32(y + i);
        float32x4_t y1 = vld1q_f32(y + i + 4);
        float32x4_t y2 = vld1q_f32(y + i + 8);
        float32x4_t y3 = vld1q_f32(y + i + 12);

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
        vst1q_f32(x + i, temp0);
        vst1q_f32(x + i + 4, temp1);
        vst1q_f32(x + i + 8, temp2);
        vst1q_f32(x + i + 12, temp3);

        vst1q_f32(y + i, ny0);
        vst1q_f32(y + i + 4, ny1);
        vst1q_f32(y + i + 8, ny2);
        vst1q_f32(y + i + 12, ny3);
    }

    // Process 4 elements at a time
    for (; i + 4 <= n; i += 4)
    {
        float32x4_t x_vec = vld1q_f32(x + i);
        float32x4_t y_vec = vld1q_f32(y + i);

        float32x4_t temp = vmulq_f32(c_vec, x_vec);
        temp = vfmaq_f32(temp, s_vec, y_vec);

        float32x4_t ny = vmulq_f32(c_vec, y_vec);
        ny = vfmsq_f32(ny, s_vec, x_vec);

        vst1q_f32(x + i, temp);
        vst1q_f32(y + i, ny);
    }

    // Handle remaining elements
    for (; i < n; i++)
    {
        float temp = c * x[i] + s * y[i];
        y[i] = c * y[i] - s * x[i];
        x[i] = temp;
    }
}

//------------------------------------------------------
// double-precision rot kernel incx == incy == 1 (NEON)
//------------------------------------------------------
void cblas_drot_k_noinc_neon(cblas_args_t* args)
{
    double* x = args->x;
    double* y = args->y;
    register CBLAS_INDEX n = args->n;
    double c = *(double*)args->alpha;
    double s = *(double*)args->beta;

    float64x2_t c_vec = vdupq_n_f64(c);
    float64x2_t s_vec = vdupq_n_f64(s);
    CBLAS_INDEX i = 0;

    // Process 8 elements at a time using 4 NEON registers (2 doubles each)
    for (; i + 8 <= n; i += 8)
    {
        // Load x and y values
        float64x2_t x0 = vld1q_f64(x + i);
        float64x2_t x1 = vld1q_f64(x + i + 2);
        float64x2_t x2 = vld1q_f64(x + i + 4);
        float64x2_t x3 = vld1q_f64(x + i + 6);

        float64x2_t y0 = vld1q_f64(y + i);
        float64x2_t y1 = vld1q_f64(y + i + 2);
        float64x2_t y2 = vld1q_f64(y + i + 4);
        float64x2_t y3 = vld1q_f64(y + i + 6);

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
        vst1q_f64(x + i, temp0);
        vst1q_f64(x + i + 2, temp1);
        vst1q_f64(x + i + 4, temp2);
        vst1q_f64(x + i + 6, temp3);

        vst1q_f64(y + i, ny0);
        vst1q_f64(y + i + 2, ny1);
        vst1q_f64(y + i + 4, ny2);
        vst1q_f64(y + i + 6, ny3);
    }

    // Process 2 elements at a time
    for (; i + 2 <= n; i += 2)
    {
        float64x2_t x_vec = vld1q_f64(x + i);
        float64x2_t y_vec = vld1q_f64(y + i);

        float64x2_t temp = vmulq_f64(c_vec, x_vec);
        temp = vfmaq_f64(temp, s_vec, y_vec);

        float64x2_t ny = vmulq_f64(c_vec, y_vec);
        ny = vfmsq_f64(ny, s_vec, x_vec);

        vst1q_f64(x + i, temp);
        vst1q_f64(y + i, ny);
    }

    // Handle remaining elements
    for (; i < n; i++)
    {
        double temp = c * x[i] + s * y[i];
        y[i] = c * y[i] - s * x[i];
        x[i] = temp;
    }
}

#endif
