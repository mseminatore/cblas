//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"

//------------------------------------------------------
// single-precision rot kernel incx == incy == 1
//------------------------------------------------------
void cblas_srot_k_noinc(cblas_args_t* args)
{
    float* x = args->x;
    float* y = args->y;
    register CBLAS_INDEX n = args->n;
    float c = *(float*)args->alpha;
    float s = *(float*)args->beta;
    CBLAS_INDEX i = 0;

    for (; i + 4 <= n; i += 4)
    {
        float temp0 = c * x[i] + s * y[i];
        float temp1 = c * x[i+1] + s * y[i+1];
        float temp2 = c * x[i+2] + s * y[i+2];
        float temp3 = c * x[i+3] + s * y[i+3];

        y[i] = c * y[i] - s * x[i];
        y[i+1] = c * y[i+1] - s * x[i+1];
        y[i+2] = c * y[i+2] - s * x[i+2];
        y[i+3] = c * y[i+3] - s * x[i+3];

        x[i] = temp0;
        x[i+1] = temp1;
        x[i+2] = temp2;
        x[i+3] = temp3;
    }

    for (; i < n; i++)
    {
        float temp = c * x[i] + s * y[i];
        y[i] = c * y[i] - s * x[i];
        x[i] = temp;
    }
}

//------------------------------------------------------
// single-precision rot kernel
//------------------------------------------------------
void cblas_srot_k(cblas_args_t* args)
{
    float* x = args->x;
    float* y = args->y;
    register CBLAS_INDEX incx = args->incx, incy = args->incy, n = args->n;
    float c = *(float*)args->alpha;
    float s = *(float*)args->beta;

    for (CBLAS_INDEX i = 0; i < n; i++)
    {
        float temp = c * *x + s * *y;
        *y = c * *y - s * *x;
        *x = temp;

        x += incx;
        y += incy;
    }
}

//------------------------------------------------------
// double-precision rot kernel incx == incy == 1
//------------------------------------------------------
void cblas_drot_k_noinc(cblas_args_t* args)
{
    double* x = args->x;
    double* y = args->y;
    register CBLAS_INDEX n = args->n;
    double c = *(double*)args->alpha;
    double s = *(double*)args->beta;
    CBLAS_INDEX i = 0;

    for (; i + 4 <= n; i += 4)
    {
        double temp0 = c * x[i] + s * y[i];
        double temp1 = c * x[i+1] + s * y[i+1];
        double temp2 = c * x[i+2] + s * y[i+2];
        double temp3 = c * x[i+3] + s * y[i+3];

        y[i] = c * y[i] - s * x[i];
        y[i+1] = c * y[i+1] - s * x[i+1];
        y[i+2] = c * y[i+2] - s * x[i+2];
        y[i+3] = c * y[i+3] - s * x[i+3];

        x[i] = temp0;
        x[i+1] = temp1;
        x[i+2] = temp2;
        x[i+3] = temp3;
    }

    for (; i < n; i++)
    {
        double temp = c * x[i] + s * y[i];
        y[i] = c * y[i] - s * x[i];
        x[i] = temp;
    }
}

//------------------------------------------------------
// double-precision rot kernel
//------------------------------------------------------
void cblas_drot_k(cblas_args_t* args)
{
    double* x = args->x;
    double* y = args->y;
    register CBLAS_INDEX incx = args->incx, incy = args->incy, n = args->n;
    double c = *(double*)args->alpha;
    double s = *(double*)args->beta;

    for (CBLAS_INDEX i = 0; i < n; i++)
    {
        double temp = c * *x + s * *y;
        *y = c * *y - s * *x;
        *x = temp;

        x += incx;
        y += incy;
    }
}

//------------------------------------------------------
// Level-1 single-precision apply rotation
//------------------------------------------------------
void cblas_srot(CBLAS_INDEX n, float *x, CBLAS_INDEX incx, float *y, CBLAS_INDEX incy, float c, float s)
{
    CBLAS_VALIDATE_VEC2(n, x, incx, y, incy, );
    CBLAS_STATS_START();

#ifdef MT_ENABLED
    int mt_used = (n > CBLAS_MT_COPY && cblas_get_num_threads() > 1) ? 1 : 0;
#else
    int mt_used = 0;
#endif

    kernel_function kernel = blas_kernels.srot_k;

    // special case kernel for no increments
    if (incx == 1 && incy == 1)
    {
        kernel = blas_kernels.srot_k_noinc;
    }

    mt_used = 0;    // disable multithreading!

    if (mt_used)
    {
        cblas_level1_exec(sizeof(float), kernel, n, x, incx, y, incy, &c, &s, "SROT");
    }
    else
    {
        cblas_args_t args;
        args.n = n;
        args.x = x;
        args.incx = incx;
        args.y = y;
        args.incy = incy;
        args.alpha = &c;
        args.beta = &s;

        kernel(&args);
    }

    CBLAS_STATS_END("srot", n, mt_used);
}

//------------------------------------------------------
// Level-1 double-precision apply rotation
//------------------------------------------------------
void cblas_drot(CBLAS_INDEX n, double *x, CBLAS_INDEX incx, double *y, CBLAS_INDEX incy, double c, double s)
{
    CBLAS_VALIDATE_VEC2(n, x, incx, y, incy, );
    CBLAS_STATS_START();

#ifdef MT_ENABLED
    int mt_used = (n > CBLAS_MT_COPY && cblas_get_num_threads() > 1) ? 1 : 0;
#else
    int mt_used = 0;
#endif

    kernel_function kernel = blas_kernels.drot_k;

    // special case kernel for no increments
    if (incx == 1 && incy == 1)
    {
        kernel = blas_kernels.drot_k_noinc;
    }

    mt_used = 0;    // disable multithreading!

    if (mt_used)
    {
        cblas_level1_exec(sizeof(double), kernel, n, x, incx, y, incy, &c, &s, "DROT");
    }
    else
    {
        cblas_args_t args;
        args.n = n;
        args.x = x;
        args.incx = incx;
        args.y = y;
        args.incy = incy;
        args.alpha = &c;
        args.beta = &s;

        kernel(&args);
    }

    CBLAS_STATS_END("drot", n, mt_used);
}
