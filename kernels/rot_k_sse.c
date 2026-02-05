//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"
#include "cblas_simd.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)

//------------------------------------------------------
// single-precision rot kernel incx == incy == 1 (SSE)
//------------------------------------------------------
void cblas_srot_k_noinc_sse(cblas_args_t* args)
{
    float* x = args->x;
    float* y = args->y;
    register CBLAS_INDEX n = args->n;
    float c = *(float*)args->alpha;
    float s = *(float*)args->beta;

    __m128 c_vec = _mm_set1_ps(c);
    __m128 s_vec = _mm_set1_ps(s);
    CBLAS_INDEX i = 0;

    // Process 16 elements at a time using 4 SSE registers
    for (; i + 16 <= n; i += 16)
    {
        // Load x and y values
        __m128 x0 = _mm_loadu_ps(x + i);
        __m128 x1 = _mm_loadu_ps(x + i + 4);
        __m128 x2 = _mm_loadu_ps(x + i + 8);
        __m128 x3 = _mm_loadu_ps(x + i + 12);

        __m128 y0 = _mm_loadu_ps(y + i);
        __m128 y1 = _mm_loadu_ps(y + i + 4);
        __m128 y2 = _mm_loadu_ps(y + i + 8);
        __m128 y3 = _mm_loadu_ps(y + i + 12);

        // Compute temp = c * x + s * y
        __m128 temp0 = _mm_add_ps(_mm_mul_ps(c_vec, x0), _mm_mul_ps(s_vec, y0));
        __m128 temp1 = _mm_add_ps(_mm_mul_ps(c_vec, x1), _mm_mul_ps(s_vec, y1));
        __m128 temp2 = _mm_add_ps(_mm_mul_ps(c_vec, x2), _mm_mul_ps(s_vec, y2));
        __m128 temp3 = _mm_add_ps(_mm_mul_ps(c_vec, x3), _mm_mul_ps(s_vec, y3));

        // Compute y = c * y - s * x
        __m128 ny0 = _mm_sub_ps(_mm_mul_ps(c_vec, y0), _mm_mul_ps(s_vec, x0));
        __m128 ny1 = _mm_sub_ps(_mm_mul_ps(c_vec, y1), _mm_mul_ps(s_vec, x1));
        __m128 ny2 = _mm_sub_ps(_mm_mul_ps(c_vec, y2), _mm_mul_ps(s_vec, x2));
        __m128 ny3 = _mm_sub_ps(_mm_mul_ps(c_vec, y3), _mm_mul_ps(s_vec, x3));

        // Store results
        _mm_storeu_ps(x + i, temp0);
        _mm_storeu_ps(x + i + 4, temp1);
        _mm_storeu_ps(x + i + 8, temp2);
        _mm_storeu_ps(x + i + 12, temp3);

        _mm_storeu_ps(y + i, ny0);
        _mm_storeu_ps(y + i + 4, ny1);
        _mm_storeu_ps(y + i + 8, ny2);
        _mm_storeu_ps(y + i + 12, ny3);
    }

    // Process 4 elements at a time
    for (; i + 4 <= n; i += 4)
    {
        __m128 x_vec = _mm_loadu_ps(x + i);
        __m128 y_vec = _mm_loadu_ps(y + i);

        __m128 temp = _mm_add_ps(_mm_mul_ps(c_vec, x_vec), _mm_mul_ps(s_vec, y_vec));
        __m128 ny = _mm_sub_ps(_mm_mul_ps(c_vec, y_vec), _mm_mul_ps(s_vec, x_vec));

        _mm_storeu_ps(x + i, temp);
        _mm_storeu_ps(y + i, ny);
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
// double-precision rot kernel incx == incy == 1 (SSE)
//------------------------------------------------------
void cblas_drot_k_noinc_sse(cblas_args_t* args)
{
    double* x = args->x;
    double* y = args->y;
    register CBLAS_INDEX n = args->n;
    double c = *(double*)args->alpha;
    double s = *(double*)args->beta;

    __m128d c_vec = _mm_set1_pd(c);
    __m128d s_vec = _mm_set1_pd(s);
    CBLAS_INDEX i = 0;

    // Process 8 elements at a time using 4 SSE registers (2 doubles each)
    for (; i + 8 <= n; i += 8)
    {
        // Load x and y values
        __m128d x0 = _mm_loadu_pd(x + i);
        __m128d x1 = _mm_loadu_pd(x + i + 2);
        __m128d x2 = _mm_loadu_pd(x + i + 4);
        __m128d x3 = _mm_loadu_pd(x + i + 6);

        __m128d y0 = _mm_loadu_pd(y + i);
        __m128d y1 = _mm_loadu_pd(y + i + 2);
        __m128d y2 = _mm_loadu_pd(y + i + 4);
        __m128d y3 = _mm_loadu_pd(y + i + 6);

        // Compute temp = c * x + s * y
        __m128d temp0 = _mm_add_pd(_mm_mul_pd(c_vec, x0), _mm_mul_pd(s_vec, y0));
        __m128d temp1 = _mm_add_pd(_mm_mul_pd(c_vec, x1), _mm_mul_pd(s_vec, y1));
        __m128d temp2 = _mm_add_pd(_mm_mul_pd(c_vec, x2), _mm_mul_pd(s_vec, y2));
        __m128d temp3 = _mm_add_pd(_mm_mul_pd(c_vec, x3), _mm_mul_pd(s_vec, y3));

        // Compute y = c * y - s * x
        __m128d ny0 = _mm_sub_pd(_mm_mul_pd(c_vec, y0), _mm_mul_pd(s_vec, x0));
        __m128d ny1 = _mm_sub_pd(_mm_mul_pd(c_vec, y1), _mm_mul_pd(s_vec, x1));
        __m128d ny2 = _mm_sub_pd(_mm_mul_pd(c_vec, y2), _mm_mul_pd(s_vec, x2));
        __m128d ny3 = _mm_sub_pd(_mm_mul_pd(c_vec, y3), _mm_mul_pd(s_vec, x3));

        // Store results
        _mm_storeu_pd(x + i, temp0);
        _mm_storeu_pd(x + i + 2, temp1);
        _mm_storeu_pd(x + i + 4, temp2);
        _mm_storeu_pd(x + i + 6, temp3);

        _mm_storeu_pd(y + i, ny0);
        _mm_storeu_pd(y + i + 2, ny1);
        _mm_storeu_pd(y + i + 4, ny2);
        _mm_storeu_pd(y + i + 6, ny3);
    }

    // Process 2 elements at a time
    for (; i + 2 <= n; i += 2)
    {
        __m128d x_vec = _mm_loadu_pd(x + i);
        __m128d y_vec = _mm_loadu_pd(y + i);

        __m128d temp = _mm_add_pd(_mm_mul_pd(c_vec, x_vec), _mm_mul_pd(s_vec, y_vec));
        __m128d ny = _mm_sub_pd(_mm_mul_pd(c_vec, y_vec), _mm_mul_pd(s_vec, x_vec));

        _mm_storeu_pd(x + i, temp);
        _mm_storeu_pd(y + i, ny);
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
