//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"
#include "cblas_simd.h"

#if defined(__aarch64__) && defined(__ARM_NEON)

//------------------------------------------------------
// single-precision swap kernel incx == 1 && incy == 1 (NEON)
//------------------------------------------------------
void cblas_sswap_k_noinc_neon(cblas_args_t* args)
{
    float* x = args->x;
    float* y = args->y;
    register CBLAS_INDEX n = args->n;
    CBLAS_INDEX i = 0;
    
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
        
        vst1q_f32(x + i, y0);
        vst1q_f32(x + i + 4, y1);
        vst1q_f32(x + i + 8, y2);
        vst1q_f32(x + i + 12, y3);
        
        vst1q_f32(y + i, x0);
        vst1q_f32(y + i + 4, x1);
        vst1q_f32(y + i + 8, x2);
        vst1q_f32(y + i + 12, x3);
    }
    
    // Process 4 elements at a time
    for (; i + 4 <= n; i += 4)
    {
        float32x4_t x_vec = vld1q_f32(x + i);
        float32x4_t y_vec = vld1q_f32(y + i);
        vst1q_f32(x + i, y_vec);
        vst1q_f32(y + i, x_vec);
    }
    
    // Handle remaining elements
    for (; i < n; i++)
    {
        float temp = y[i];
        y[i] = x[i];
        x[i] = temp;
    }
}

//------------------------------------------------------
// double-precision swap kernel incx == 1 && incy == 1 (NEON)
//------------------------------------------------------
void cblas_dswap_k_noinc_neon(cblas_args_t* args)
{
    double* x = args->x;
    double* y = args->y;
    register CBLAS_INDEX n = args->n;
    CBLAS_INDEX i = 0;
    
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
        
        vst1q_f64(x + i, y0);
        vst1q_f64(x + i + 2, y1);
        vst1q_f64(x + i + 4, y2);
        vst1q_f64(x + i + 6, y3);
        
        vst1q_f64(y + i, x0);
        vst1q_f64(y + i + 2, x1);
        vst1q_f64(y + i + 4, x2);
        vst1q_f64(y + i + 6, x3);
    }
    
    // Process 2 elements at a time
    for (; i + 2 <= n; i += 2)
    {
        float64x2_t x_vec = vld1q_f64(x + i);
        float64x2_t y_vec = vld1q_f64(y + i);
        vst1q_f64(x + i, y_vec);
        vst1q_f64(y + i, x_vec);
    }
    
    // Handle remaining elements
    for (; i < n; i++)
    {
        double temp = y[i];
        y[i] = x[i];
        x[i] = temp;
    }
}

#endif
