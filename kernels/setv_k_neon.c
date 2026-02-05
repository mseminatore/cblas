//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"
#include "cblas_simd.h"

#if defined(__aarch64__) && defined(__ARM_NEON)

//------------------------------------------------------
// single-precision setv kernel inc == 1 (NEON)
//------------------------------------------------------
void cblas_ssetv_k_noinc_neon(cblas_args_t* args)
{
    float* x = args->x;
    float v = *(float*)args->alpha;
    register CBLAS_INDEX n = args->n;
    CBLAS_INDEX i = 0;

    // Broadcast value to all 4 lanes
    float32x4_t v_vec = vdupq_n_f32(v);

    // Process 16 floats per iteration for better ILP
    for (; i + 16 <= n; i += 16)
    {
        vst1q_f32(x + i, v_vec);
        vst1q_f32(x + i + 4, v_vec);
        vst1q_f32(x + i + 8, v_vec);
        vst1q_f32(x + i + 12, v_vec);
    }

    // Process 4 floats at a time
    for (; i + 4 <= n; i += 4)
    {
        vst1q_f32(x + i, v_vec);
    }

    // Handle remaining elements
    for (; i < n; i++)
    {
        x[i] = v;
    }
}

//------------------------------------------------------
// double-precision setv kernel inc == 1 (NEON)
//------------------------------------------------------
void cblas_dsetv_k_noinc_neon(cblas_args_t* args)
{
    double* x = args->x;
    double v = *(double*)args->alpha;
    register CBLAS_INDEX n = args->n;
    CBLAS_INDEX i = 0;

    // Broadcast value to both lanes
    float64x2_t v_vec = vdupq_n_f64(v);

    // Process 8 doubles per iteration for better ILP
    for (; i + 8 <= n; i += 8)
    {
        vst1q_f64(x + i, v_vec);
        vst1q_f64(x + i + 2, v_vec);
        vst1q_f64(x + i + 4, v_vec);
        vst1q_f64(x + i + 6, v_vec);
    }

    // Process 2 doubles at a time
    for (; i + 2 <= n; i += 2)
    {
        vst1q_f64(x + i, v_vec);
    }

    // Handle remaining elements
    for (; i < n; i++)
    {
        x[i] = v;
    }
}

#endif
