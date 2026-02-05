//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"
#include "cblas_simd.h"

#if defined(__aarch64__) && defined(__ARM_NEON)

//------------------------------------------------------
// single-precision axpby kernel incx == 1 && incy == 1 (NEON)
// y = alpha * x + beta * y
//------------------------------------------------------
void cblas_saxpby_k_noinc_neon(cblas_args_t* args)
{
    float* x = args->x;
    float* y = args->y;
    float alpha = *(float*)args->alpha;
    float beta = *(float*)args->beta;
    register CBLAS_INDEX n = args->n;
    register CBLAS_INDEX i = 0;
    
    float32x4_t alpha_vec = vdupq_n_f32(alpha);
    float32x4_t beta_vec = vdupq_n_f32(beta);
    
    // Process 16 elements at a time (4x4)
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
        
        // y = alpha*x + beta*y using FMA: vmlaq(beta*y, alpha, x)
        y0 = vmlaq_f32(vmulq_f32(beta_vec, y0), alpha_vec, x0);
        y1 = vmlaq_f32(vmulq_f32(beta_vec, y1), alpha_vec, x1);
        y2 = vmlaq_f32(vmulq_f32(beta_vec, y2), alpha_vec, x2);
        y3 = vmlaq_f32(vmulq_f32(beta_vec, y3), alpha_vec, x3);
        
        vst1q_f32(y + i, y0);
        vst1q_f32(y + i + 4, y1);
        vst1q_f32(y + i + 8, y2);
        vst1q_f32(y + i + 12, y3);
    }
    
    // Process 4 elements at a time
    for (; i + 4 <= n; i += 4)
    {
        float32x4_t x_vec = vld1q_f32(x + i);
        float32x4_t y_vec = vld1q_f32(y + i);
        y_vec = vmlaq_f32(vmulq_f32(beta_vec, y_vec), alpha_vec, x_vec);
        vst1q_f32(y + i, y_vec);
    }
    
    // Scalar cleanup
    for (; i < n; i++)
        y[i] = alpha * x[i] + beta * y[i];
}

//------------------------------------------------------
// double-precision axpby kernel incx == 1 && incy == 1 (NEON)
// y = alpha * x + beta * y
//------------------------------------------------------
void cblas_daxpby_k_noinc_neon(cblas_args_t* args)
{
    double* x = args->x;
    double* y = args->y;
    double alpha = *(double*)args->alpha;
    double beta = *(double*)args->beta;
    register CBLAS_INDEX n = args->n;
    register CBLAS_INDEX i = 0;
    
    float64x2_t alpha_vec = vdupq_n_f64(alpha);
    float64x2_t beta_vec = vdupq_n_f64(beta);
    
    // Process 8 elements at a time (4x2)
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
        
        // y = alpha*x + beta*y using FMA: vmlaq(beta*y, alpha, x)
        y0 = vmlaq_f64(vmulq_f64(beta_vec, y0), alpha_vec, x0);
        y1 = vmlaq_f64(vmulq_f64(beta_vec, y1), alpha_vec, x1);
        y2 = vmlaq_f64(vmulq_f64(beta_vec, y2), alpha_vec, x2);
        y3 = vmlaq_f64(vmulq_f64(beta_vec, y3), alpha_vec, x3);
        
        vst1q_f64(y + i, y0);
        vst1q_f64(y + i + 2, y1);
        vst1q_f64(y + i + 4, y2);
        vst1q_f64(y + i + 6, y3);
    }
    
    // Process 2 elements at a time
    for (; i + 2 <= n; i += 2)
    {
        float64x2_t x_vec = vld1q_f64(x + i);
        float64x2_t y_vec = vld1q_f64(y + i);
        y_vec = vmlaq_f64(vmulq_f64(beta_vec, y_vec), alpha_vec, x_vec);
        vst1q_f64(y + i, y_vec);
    }
    
    // Scalar cleanup
    for (; i < n; i++)
        y[i] = alpha * x[i] + beta * y[i];
}

#endif
