//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"

//------------------------------------------------------
// single-precision copy kernel
//------------------------------------------------------
static void cblas_scopy_k(cblas_args_t *args)
{
    float *x = args->x;
    float *y = args->y;

    for (CBLAS_INDEX i = 0; i < args->n; i++)
    {
        *x = *y;
        x += args->incx;
        y += args->incy;
    }
}

//------------------------------------------------------
// double-precision copy kernel
//------------------------------------------------------
static void cblas_dcopy_k(cblas_args_t* args)
{
    double *x = args->x;
    double *y = args->y;

    for (CBLAS_INDEX i = 0; i < args->n; i++)
    {
        *x = *y;
        x += args->incx;
        y += args->incy;
    }
}

//------------------------------------------------------
// single-precision copy
//------------------------------------------------------
void cblas_scopy(CBLAS_INDEX n, float *x, CBLAS_INDEX incx, const float *y, CBLAS_INDEX incy)
{
    if (n < 0 || !x || !y)
    {
        assert(n > 0 && x && y);
        return;
    }

#ifdef MT_ENABLED
    cblas_level1_exec(sizeof(float), cblas_scopy_k, n, x, incx, y, incy);
#else
    for (CBLAS_INDEX i = 0; i < n; i++)
    {
        *x = *y;
        x += incx;
        y += incy;
    }
#endif
}

//------------------------------------------------------
// double-precision copy
//------------------------------------------------------
void cblas_dcopy(CBLAS_INDEX n, double *x, CBLAS_INDEX incx, const double *y, CBLAS_INDEX incy)
{
    if (n < 0 || !x || !y)
    {
        assert(n > 0 && x && y);
        return;
    }

#ifdef MT_ENABLED
    cblas_level1_exec(sizeof(double), cblas_dcopy_k, n, x, incx, y, incy);
#else
    for (CBLAS_INDEX i = 0; i < n; i++)
    {
        *x = *y;
        x += incx;
        y += incy;
    }
#endif
}
