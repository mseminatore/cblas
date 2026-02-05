//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"
#include "cblas_simd.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)

//------------------------------------------------------
// single-precision setv kernel inc == 1 (SSE)
//------------------------------------------------------
void cblas_ssetv_k_noinc_sse(cblas_args_t* args)
{
    float* x = args->x;
    float v = *(float*)args->alpha;
    register CBLAS_INDEX n = args->n;
    CBLAS_INDEX i = 0;

    // Broadcast value to all 4 lanes
    __m128 v_vec = _mm_set1_ps(v);

    // Process 16 floats per iteration (4x4 with SSE) for better ILP
    for (; i + 16 <= n; i += 16)
    {
        _mm_storeu_ps(x + i, v_vec);
        _mm_storeu_ps(x + i + 4, v_vec);
        _mm_storeu_ps(x + i + 8, v_vec);
        _mm_storeu_ps(x + i + 12, v_vec);
    }

    // Process 4 floats at a time
    for (; i + 4 <= n; i += 4)
    {
        _mm_storeu_ps(x + i, v_vec);
    }

    // Handle remaining elements
    for (; i < n; i++)
    {
        x[i] = v;
    }
}

//------------------------------------------------------
// double-precision setv kernel inc == 1 (SSE)
//------------------------------------------------------
void cblas_dsetv_k_noinc_sse(cblas_args_t* args)
{
    double* x = args->x;
    double v = *(double*)args->alpha;
    register CBLAS_INDEX n = args->n;
    CBLAS_INDEX i = 0;

    // Broadcast value to both lanes
    __m128d v_vec = _mm_set1_pd(v);

    // Process 8 doubles per iteration (4x2 with SSE) for better ILP
    for (; i + 8 <= n; i += 8)
    {
        _mm_storeu_pd(x + i, v_vec);
        _mm_storeu_pd(x + i + 2, v_vec);
        _mm_storeu_pd(x + i + 4, v_vec);
        _mm_storeu_pd(x + i + 6, v_vec);
    }

    // Process 2 doubles at a time
    for (; i + 2 <= n; i += 2)
    {
        _mm_storeu_pd(x + i, v_vec);
    }

    // Handle remaining elements
    for (; i < n; i++)
    {
        x[i] = v;
    }
}

#endif
