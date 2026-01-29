//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"

//------------------------------------------------------
//
//------------------------------------------------------
static void cblas_sswap_k(cblas_args_t* args)
{
    float temp;
    float* x = args->x;
    float* y = args->y;

    for (CBLAS_INDEX i = 0; i < args->n; i++)
    {
        temp = *y;
        *y = *x;
        *x = temp;

        x += args->incx;
        y += args->incy;
    }
}

//------------------------------------------------------
//
//------------------------------------------------------
static void cblas_dswap_k(cblas_args_t* args)
{
    double temp;
    double* x = args->x;
    double* y = args->y;

    for (CBLAS_INDEX i = 0; i < args->n; i++)
    {
        temp = *y;
        *y = *x;
        *x = temp;

        x += args->incx;
        y += args->incy;
    }
}
//------------------------------------------------------
// Level-1 single-precision swap
//------------------------------------------------------
void cblas_sswap(CBLAS_INDEX n, float *x, CBLAS_INDEX incx, float *y, CBLAS_INDEX incy)
{
    CBLAS_VALIDATE_VEC2(n, x, incx, y, incy, );

    CBLAS_STATS_START();

#ifdef MT_ENABLED
    int mt_used = (n > CBLAS_MT_COPY) ? 1 : 0;
    cblas_level1_exec(sizeof(float), cblas_sswap_k, n, x, incx, y, incy);
#else
    int mt_used = 0;
    float temp;
    for (CBLAS_INDEX i = 0; i < n; i++)
    {
        temp = *y;
        *y = *x;
        *x = temp;

        x += incx;
        y += incy;
    }
#endif

    CBLAS_STATS_END("sswap", n, mt_used);
}

//------------------------------------------------------
// Level-1 double-precision swap
//------------------------------------------------------
void cblas_dswap(CBLAS_INDEX n, double *x, CBLAS_INDEX incx, double *y, CBLAS_INDEX incy)
{
    CBLAS_VALIDATE_VEC2(n, x, incx, y, incy, );

    CBLAS_STATS_START();

#ifdef MT_ENABLED
    int mt_used = (n > CBLAS_MT_COPY) ? 1 : 0;
    cblas_level1_exec(sizeof(double), cblas_dswap_k, n, x, incx, y, incy);
#else
    int mt_used = 0;
    double temp;
    for (CBLAS_INDEX i = 0; i < n; i++)
    {
        temp = *y;
        *y = *x;
        *x = temp;

        x += incx;
        y += incy;
    }    
#endif

    CBLAS_STATS_END("dswap", n, mt_used);
}
