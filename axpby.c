//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include <assert.h>
#include "cblas.h"

//------------------------------------------------------
//
//------------------------------------------------------
void cblas_saxpby(CBLAS_INDEX n, float alpha, float *x, CBLAS_INDEX incx, float beta, float *y, CBLAS_INDEX incy)
{
    if (n < 0 || !x || !y)
    {
        assert(n > 0 && x && y);
        return;
    }

    for (CBLAS_INDEX i = 0; i < n; i++)
    {
        *y = alpha * *x + beta * *y;
        x += incx;
        y += incy;
    }
}

//------------------------------------------------------
//
//------------------------------------------------------
void cblas_daxpby(CBLAS_INDEX n, double alpha, double *x, CBLAS_INDEX incx, double beta, double *y, CBLAS_INDEX incy)
{
    if (n < 0 || !x || !y)
    {
        assert(n > 0 && x && y);
        return;
    }

    for (CBLAS_INDEX i = 0; i < n; i++)
    {
        *y = alpha * *x + beta * *y;
        x += incx;
        y += incy;
    }
}
