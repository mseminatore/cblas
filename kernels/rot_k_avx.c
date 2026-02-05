//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"
#include "cblas_simd.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)

//------------------------------------------------------
// single-precision rot kernel incx == incy == 1 (AVX)
//------------------------------------------------------
void cblas_srot_k_noinc_avx(cblas_args_t* args)
{
    float* x = args->x;
    float* y = args->y;
    register CBLAS_INDEX n = args->n;
    float c = *(float*)args->alpha;
    float s = *(float*)args->beta;

    __m256 c_vec = _mm256_set1_ps(c);
    __m256 s_vec = _mm256_set1_ps(s);
    CBLAS_INDEX i = 0;
    int use_prefetch = (n > CBLAS_PREFETCH_THRESHOLD);

    // Process 32 elements at a time using 4 AVX registers
    for (; i + 32 <= n; i += 32)
    {
        if (use_prefetch && i + CBLAS_PREFETCH_DISTANCE < n) {
            CBLAS_PREFETCH(&x[i + CBLAS_PREFETCH_DISTANCE], 0, 3);
            CBLAS_PREFETCH(&y[i + CBLAS_PREFETCH_DISTANCE], 0, 3);
        }

        // Load x and y values
        __m256 x0 = _mm256_loadu_ps(x + i);
        __m256 x1 = _mm256_loadu_ps(x + i + 8);
        __m256 x2 = _mm256_loadu_ps(x + i + 16);
        __m256 x3 = _mm256_loadu_ps(x + i + 24);

        __m256 y0 = _mm256_loadu_ps(y + i);
        __m256 y1 = _mm256_loadu_ps(y + i + 8);
        __m256 y2 = _mm256_loadu_ps(y + i + 16);
        __m256 y3 = _mm256_loadu_ps(y + i + 24);

        // Compute temp = c * x + s * y
        __m256 temp0 = _mm256_add_ps(_mm256_mul_ps(c_vec, x0), _mm256_mul_ps(s_vec, y0));
        __m256 temp1 = _mm256_add_ps(_mm256_mul_ps(c_vec, x1), _mm256_mul_ps(s_vec, y1));
        __m256 temp2 = _mm256_add_ps(_mm256_mul_ps(c_vec, x2), _mm256_mul_ps(s_vec, y2));
        __m256 temp3 = _mm256_add_ps(_mm256_mul_ps(c_vec, x3), _mm256_mul_ps(s_vec, y3));

        // Compute y = c * y - s * x
        __m256 ny0 = _mm256_sub_ps(_mm256_mul_ps(c_vec, y0), _mm256_mul_ps(s_vec, x0));
        __m256 ny1 = _mm256_sub_ps(_mm256_mul_ps(c_vec, y1), _mm256_mul_ps(s_vec, x1));
        __m256 ny2 = _mm256_sub_ps(_mm256_mul_ps(c_vec, y2), _mm256_mul_ps(s_vec, x2));
        __m256 ny3 = _mm256_sub_ps(_mm256_mul_ps(c_vec, y3), _mm256_mul_ps(s_vec, x3));

        // Store results
        _mm256_storeu_ps(x + i, temp0);
        _mm256_storeu_ps(x + i + 8, temp1);
        _mm256_storeu_ps(x + i + 16, temp2);
        _mm256_storeu_ps(x + i + 24, temp3);

        _mm256_storeu_ps(y + i, ny0);
        _mm256_storeu_ps(y + i + 8, ny1);
        _mm256_storeu_ps(y + i + 16, ny2);
        _mm256_storeu_ps(y + i + 24, ny3);
    }

    // Process 8 elements at a time
    for (; i + 8 <= n; i += 8)
    {
        __m256 x_vec = _mm256_loadu_ps(x + i);
        __m256 y_vec = _mm256_loadu_ps(y + i);

        __m256 temp = _mm256_add_ps(_mm256_mul_ps(c_vec, x_vec), _mm256_mul_ps(s_vec, y_vec));
        __m256 ny = _mm256_sub_ps(_mm256_mul_ps(c_vec, y_vec), _mm256_mul_ps(s_vec, x_vec));

        _mm256_storeu_ps(x + i, temp);
        _mm256_storeu_ps(y + i, ny);
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
// double-precision rot kernel incx == incy == 1 (AVX)
//------------------------------------------------------
void cblas_drot_k_noinc_avx(cblas_args_t* args)
{
    double* x = args->x;
    double* y = args->y;
    register CBLAS_INDEX n = args->n;
    double c = *(double*)args->alpha;
    double s = *(double*)args->beta;

    __m256d c_vec = _mm256_set1_pd(c);
    __m256d s_vec = _mm256_set1_pd(s);
    CBLAS_INDEX i = 0;
    int use_prefetch = (n > CBLAS_PREFETCH_THRESHOLD);

    // Process 16 elements at a time using 4 AVX registers (4 doubles each)
    for (; i + 16 <= n; i += 16)
    {
        if (use_prefetch && i + CBLAS_PREFETCH_DISTANCE < n) {
            CBLAS_PREFETCH(&x[i + CBLAS_PREFETCH_DISTANCE], 0, 3);
            CBLAS_PREFETCH(&y[i + CBLAS_PREFETCH_DISTANCE], 0, 3);
        }

        // Load x and y values
        __m256d x0 = _mm256_loadu_pd(x + i);
        __m256d x1 = _mm256_loadu_pd(x + i + 4);
        __m256d x2 = _mm256_loadu_pd(x + i + 8);
        __m256d x3 = _mm256_loadu_pd(x + i + 12);

        __m256d y0 = _mm256_loadu_pd(y + i);
        __m256d y1 = _mm256_loadu_pd(y + i + 4);
        __m256d y2 = _mm256_loadu_pd(y + i + 8);
        __m256d y3 = _mm256_loadu_pd(y + i + 12);

        // Compute temp = c * x + s * y
        __m256d temp0 = _mm256_add_pd(_mm256_mul_pd(c_vec, x0), _mm256_mul_pd(s_vec, y0));
        __m256d temp1 = _mm256_add_pd(_mm256_mul_pd(c_vec, x1), _mm256_mul_pd(s_vec, y1));
        __m256d temp2 = _mm256_add_pd(_mm256_mul_pd(c_vec, x2), _mm256_mul_pd(s_vec, y2));
        __m256d temp3 = _mm256_add_pd(_mm256_mul_pd(c_vec, x3), _mm256_mul_pd(s_vec, y3));

        // Compute y = c * y - s * x
        __m256d ny0 = _mm256_sub_pd(_mm256_mul_pd(c_vec, y0), _mm256_mul_pd(s_vec, x0));
        __m256d ny1 = _mm256_sub_pd(_mm256_mul_pd(c_vec, y1), _mm256_mul_pd(s_vec, x1));
        __m256d ny2 = _mm256_sub_pd(_mm256_mul_pd(c_vec, y2), _mm256_mul_pd(s_vec, x2));
        __m256d ny3 = _mm256_sub_pd(_mm256_mul_pd(c_vec, y3), _mm256_mul_pd(s_vec, x3));

        // Store results
        _mm256_storeu_pd(x + i, temp0);
        _mm256_storeu_pd(x + i + 4, temp1);
        _mm256_storeu_pd(x + i + 8, temp2);
        _mm256_storeu_pd(x + i + 12, temp3);

        _mm256_storeu_pd(y + i, ny0);
        _mm256_storeu_pd(y + i + 4, ny1);
        _mm256_storeu_pd(y + i + 8, ny2);
        _mm256_storeu_pd(y + i + 12, ny3);
    }

    // Process 4 elements at a time
    for (; i + 4 <= n; i += 4)
    {
        __m256d x_vec = _mm256_loadu_pd(x + i);
        __m256d y_vec = _mm256_loadu_pd(y + i);

        __m256d temp = _mm256_add_pd(_mm256_mul_pd(c_vec, x_vec), _mm256_mul_pd(s_vec, y_vec));
        __m256d ny = _mm256_sub_pd(_mm256_mul_pd(c_vec, y_vec), _mm256_mul_pd(s_vec, x_vec));

        _mm256_storeu_pd(x + i, temp);
        _mm256_storeu_pd(y + i, ny);
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
