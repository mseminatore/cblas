//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"
#include "cblas_simd.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)

//------------------------------------------------------
// single-precision setv kernel inc == 1 (AVX)
//------------------------------------------------------
void cblas_ssetv_k_noinc_avx(cblas_args_t* args)
{
    float* x = args->x;
    float v = *(float*)args->alpha;
    register CBLAS_INDEX n = args->n;
    CBLAS_INDEX i = 0;
    int use_prefetch = (n > CBLAS_PREFETCH_THRESHOLD);

    // Broadcast value to all 8 lanes
    __m256 v_vec = _mm256_set1_ps(v);

    // Process 32 floats per iteration (4x8 with AVX) for better ILP
    for (; i + 32 <= n; i += 32)
    {
        if (use_prefetch && i + CBLAS_PREFETCH_DISTANCE < n) {
            CBLAS_PREFETCH(&x[i + CBLAS_PREFETCH_DISTANCE], 1, 3);
        }

        _mm256_storeu_ps(x + i, v_vec);
        _mm256_storeu_ps(x + i + 8, v_vec);
        _mm256_storeu_ps(x + i + 16, v_vec);
        _mm256_storeu_ps(x + i + 24, v_vec);
    }

    // Process 8 floats at a time
    for (; i + 8 <= n; i += 8)
    {
        _mm256_storeu_ps(x + i, v_vec);
    }

    // Handle remaining elements
    for (; i < n; i++)
    {
        x[i] = v;
    }
}

//------------------------------------------------------
// double-precision setv kernel inc == 1 (AVX)
//------------------------------------------------------
void cblas_dsetv_k_noinc_avx(cblas_args_t* args)
{
    double* x = args->x;
    double v = *(double*)args->alpha;
    register CBLAS_INDEX n = args->n;
    CBLAS_INDEX i = 0;
    int use_prefetch = (n > CBLAS_PREFETCH_THRESHOLD);

    // Broadcast value to all 4 lanes
    __m256d v_vec = _mm256_set1_pd(v);

    // Process 16 doubles per iteration (4x4 with AVX) for better ILP
    for (; i + 16 <= n; i += 16)
    {
        if (use_prefetch && i + CBLAS_PREFETCH_DISTANCE < n) {
            CBLAS_PREFETCH(&x[i + CBLAS_PREFETCH_DISTANCE], 1, 3);
        }

        _mm256_storeu_pd(x + i, v_vec);
        _mm256_storeu_pd(x + i + 4, v_vec);
        _mm256_storeu_pd(x + i + 8, v_vec);
        _mm256_storeu_pd(x + i + 12, v_vec);
    }

    // Process 4 doubles at a time
    for (; i + 4 <= n; i += 4)
    {
        _mm256_storeu_pd(x + i, v_vec);
    }

    // Handle remaining elements
    for (; i < n; i++)
    {
        x[i] = v;
    }
}

#endif
