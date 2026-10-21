//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"

//------------------------------------------------------
// Base scalar SGEMV kernel (fallback)
//------------------------------------------------------
void sgemv_k_base(cblas_args_t* args)
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
            sum = 0.0f;
            for (CBLAS_INDEX col = 0; col < n; col++)
            {
                sum += a[row * lda + col] * x[col];
            }
            y[row] = beta * y[row] + alpha * sum;
        }
    }
    else
    {
        // General case with strides
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
// Base scalar DGEMV kernel (fallback)
//------------------------------------------------------
void dgemv_k_base(cblas_args_t* args)
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
            sum = 0.0;
            for (CBLAS_INDEX col = 0; col < n; col++)
            {
                sum += a[row * lda + col] * x[col];
            }
            y[row] = beta * y[row] + alpha * sum;
        }
    }
    else
    {
        // General case with strides
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

//------------------------------------------------------
// Base scalar SGEMV transpose kernel (fallback)
// y[0:n] = alpha * A^T * x + beta * y, A row-major m x n
// Walks A row by row (axpy form) to keep accesses contiguous
//------------------------------------------------------
void sgemv_t_k_base(cblas_args_t* args)
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

    if (beta == 0.0f)
    {
        for (CBLAS_INDEX col = 0; col < n; col++)
            y[col * incy] = 0.0f;
    }
    else if (beta != 1.0f)
    {
        for (CBLAS_INDEX col = 0; col < n; col++)
            y[col * incy] *= beta;
    }

    for (CBLAS_INDEX row = 0; row < m; row++)
    {
        float t = alpha * x[row * incx];
        for (CBLAS_INDEX col = 0; col < n; col++)
        {
            y[col * incy] += t * a[row * lda + col];
        }
    }
}

//------------------------------------------------------
// Base scalar DGEMV transpose kernel (fallback)
//------------------------------------------------------
void dgemv_t_k_base(cblas_args_t* args)
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

    if (beta == 0.0)
    {
        for (CBLAS_INDEX col = 0; col < n; col++)
            y[col * incy] = 0.0;
    }
    else if (beta != 1.0)
    {
        for (CBLAS_INDEX col = 0; col < n; col++)
            y[col * incy] *= beta;
    }

    for (CBLAS_INDEX row = 0; row < m; row++)
    {
        double t = alpha * x[row * incx];
        for (CBLAS_INDEX col = 0; col < n; col++)
        {
            y[col * incy] += t * a[row * lda + col];
        }
    }
}
