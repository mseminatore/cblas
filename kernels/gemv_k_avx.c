//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"
#include "cblas_simd.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)

//------------------------------------------------------
// Optimized single-precision row-wise dot product kernel (incx==1)
// Computes one row of matrix-vector product using AVX and multi-accumulator unrolling
//------------------------------------------------------
static void sgemv_row_dot_avx(float* a_row, float* x, CBLAS_INDEX n, float* result)
{
    CBLAS_INDEX i = 0;
    float sum = 0.0f;

#if defined(__AVX2__)
    // AVX2 path: Use 4 independent accumulators to hide latency
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();

    // Process 32 floats per iteration (4 accumulators × 8 floats)
    CBLAS_INDEX unroll_end = (n / 32) * 32;

#if defined(CBLAS_PREFETCH)
    const CBLAS_INDEX prefetch_distance = CBLAS_PREFETCH_DISTANCE;
#endif

    for (; i < unroll_end; i += 32)
    {
#if defined(CBLAS_PREFETCH)
        // Prefetch data ahead to hide memory latency
        if (i + prefetch_distance < n) {
            CBLAS_PREFETCH(a_row + i + prefetch_distance, 0, 3);
            CBLAS_PREFETCH(x + i + prefetch_distance, 0, 3);
        }
#endif

        __m256 a0 = _mm256_loadu_ps(a_row + i);
        __m256 x0 = _mm256_loadu_ps(x + i);
        __m256 a1 = _mm256_loadu_ps(a_row + i + 8);
        __m256 x1 = _mm256_loadu_ps(x + i + 8);
        __m256 a2 = _mm256_loadu_ps(a_row + i + 16);
        __m256 x2 = _mm256_loadu_ps(x + i + 16);
        __m256 a3 = _mm256_loadu_ps(a_row + i + 24);
        __m256 x3 = _mm256_loadu_ps(x + i + 24);

#if defined(USE_INTEL_FMA)
        sum0 = _mm256_fmadd_ps(a0, x0, sum0);
        sum1 = _mm256_fmadd_ps(a1, x1, sum1);
        sum2 = _mm256_fmadd_ps(a2, x2, sum2);
        sum3 = _mm256_fmadd_ps(a3, x3, sum3);
#else
        sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(a0, x0));
        sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(a1, x1));
        sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(a2, x2));
        sum3 = _mm256_add_ps(sum3, _mm256_mul_ps(a3, x3));
#endif
    }

    // Combine the 4 accumulators
    __m256 sum_avx = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));

    // Handle remaining blocks of 8
    for (; i + 8 <= n; i += 8)
    {
        __m256 a_vec = _mm256_loadu_ps(a_row + i);
        __m256 x_vec = _mm256_loadu_ps(x + i);

#if defined(USE_INTEL_FMA)
        sum_avx = _mm256_fmadd_ps(a_vec, x_vec, sum_avx);
#else
        sum_avx = _mm256_add_ps(sum_avx, _mm256_mul_ps(a_vec, x_vec));
#endif
    }

    // Horizontal sum of AVX vector
    float sum_array[8];
    _mm256_storeu_ps(sum_array, sum_avx);
    sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3] +
          sum_array[4] + sum_array[5] + sum_array[6] + sum_array[7];
#endif

    // Handle remaining elements
    for (; i < n; i++)
    {
        sum += a_row[i] * x[i];
    }

    *result = sum;
}

//------------------------------------------------------
// Optimized double-precision row-wise dot product kernel (incx==1)
//------------------------------------------------------
static void dgemv_row_dot_avx(double* a_row, double* x, CBLAS_INDEX n, double* result)
{
    CBLAS_INDEX i = 0;
    double sum = 0.0;

#if defined(__AVX2__)
    // AVX2 path: Use 4 independent accumulators
    __m256d sum0 = _mm256_setzero_pd();
    __m256d sum1 = _mm256_setzero_pd();
    __m256d sum2 = _mm256_setzero_pd();
    __m256d sum3 = _mm256_setzero_pd();

    // Process 16 doubles per iteration (4 accumulators × 4 doubles)
    CBLAS_INDEX unroll_end = (n / 16) * 16;

#if defined(CBLAS_PREFETCH)
    const CBLAS_INDEX prefetch_distance = CBLAS_PREFETCH_DISTANCE;
#endif

    for (; i < unroll_end; i += 16)
    {
#if defined(CBLAS_PREFETCH)
        if (i + prefetch_distance < n) {
            CBLAS_PREFETCH(a_row + i + prefetch_distance, 0, 3);
            CBLAS_PREFETCH(x + i + prefetch_distance, 0, 3);
        }
#endif

        __m256d a0 = _mm256_loadu_pd(a_row + i);
        __m256d x0 = _mm256_loadu_pd(x + i);
        __m256d a1 = _mm256_loadu_pd(a_row + i + 4);
        __m256d x1 = _mm256_loadu_pd(x + i + 4);
        __m256d a2 = _mm256_loadu_pd(a_row + i + 8);
        __m256d x2 = _mm256_loadu_pd(x + i + 8);
        __m256d a3 = _mm256_loadu_pd(a_row + i + 12);
        __m256d x3 = _mm256_loadu_pd(x + i + 12);

#if defined(USE_INTEL_FMA)
        sum0 = _mm256_fmadd_pd(a0, x0, sum0);
        sum1 = _mm256_fmadd_pd(a1, x1, sum1);
        sum2 = _mm256_fmadd_pd(a2, x2, sum2);
        sum3 = _mm256_fmadd_pd(a3, x3, sum3);
#else
        sum0 = _mm256_add_pd(sum0, _mm256_mul_pd(a0, x0));
        sum1 = _mm256_add_pd(sum1, _mm256_mul_pd(a1, x1));
        sum2 = _mm256_add_pd(sum2, _mm256_mul_pd(a2, x2));
        sum3 = _mm256_add_pd(sum3, _mm256_mul_pd(a3, x3));
#endif
    }

    // Combine the 4 accumulators
    __m256d sum_avx = _mm256_add_pd(_mm256_add_pd(sum0, sum1), _mm256_add_pd(sum2, sum3));

    // Handle remaining blocks of 4
    for (; i + 4 <= n; i += 4)
    {
        __m256d a_vec = _mm256_loadu_pd(a_row + i);
        __m256d x_vec = _mm256_loadu_pd(x + i);

#if defined(USE_INTEL_FMA)
        sum_avx = _mm256_fmadd_pd(a_vec, x_vec, sum_avx);
#else
        sum_avx = _mm256_add_pd(sum_avx, _mm256_mul_pd(a_vec, x_vec));
#endif
    }

    // Horizontal sum
    double sum_array[4];
    _mm256_storeu_pd(sum_array, sum_avx);
    sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];
#endif

    // Handle remaining elements
    for (; i < n; i++)
    {
        sum += a_row[i] * x[i];
    }

    *result = sum;
}

//------------------------------------------------------
// Single-precision GEMV kernel (AVX/FMA)
//------------------------------------------------------
void sgemv_k_avx(cblas_args_t* args)
{
    float* a = (float*)args->a;
    float* x = (float*)args->x;
    float* y = (float*)args->y;
    CBLAS_INDEX m = args->m;
    CBLAS_INDEX n = args->n;
    CBLAS_INDEX lda = args->lda;
    CBLAS_INDEX incx = args->incx;
    CBLAS_INDEX incy = args->incy;
    float alpha = *(float*)args->alpha;
    float beta = *(float*)args->beta;

    float sum;

    // Optimized path for unit strides
    if (incx == 1 && incy == 1)
    {
        for (CBLAS_INDEX row = 0; row < m; row++)
        {
            sgemv_row_dot_avx(&a[row * lda], x, n, &sum);
            y[row] = beta * y[row] + alpha * sum;
        }
    }
    else
    {
        // General case with strides - fall back to scalar
        for (CBLAS_INDEX row = 0; row < m; row++)
        {
            sum = 0.0f;
            for (CBLAS_INDEX col = 0; col < n; col++)
            {
                sum += a[row * lda + col] * x[col * incx];
            }
            y[row * incy] = beta * y[row * incy] + alpha * sum;
        }
    }
}

//------------------------------------------------------
// Double-precision GEMV kernel (AVX/FMA)
//------------------------------------------------------
void dgemv_k_avx(cblas_args_t* args)
{
    double* a = (double*)args->a;
    double* x = (double*)args->x;
    double* y = (double*)args->y;
    CBLAS_INDEX m = args->m;
    CBLAS_INDEX n = args->n;
    CBLAS_INDEX lda = args->lda;
    CBLAS_INDEX incx = args->incx;
    CBLAS_INDEX incy = args->incy;
    double alpha = *(double*)args->alpha;
    double beta = *(double*)args->beta;

    double sum;

    // Optimized path for unit strides
    if (incx == 1 && incy == 1)
    {
        for (CBLAS_INDEX row = 0; row < m; row++)
        {
            dgemv_row_dot_avx(&a[row * lda], x, n, &sum);
            y[row] = beta * y[row] + alpha * sum;
        }
    }
    else
    {
        // General case with strides - fall back to scalar
        for (CBLAS_INDEX row = 0; row < m; row++)
        {
            sum = 0.0;
            for (CBLAS_INDEX col = 0; col < n; col++)
            {
                sum += a[row * lda + col] * x[col * incx];
            }
            y[row * incy] = beta * y[row * incy] + alpha * sum;
        }
    }
}

#endif // x86/x64
