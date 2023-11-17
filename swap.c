//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"

//------------------------------------------------------
//
//------------------------------------------------------
void cblas_sswap(CBLAS_INDEX n, float *x, CBLAS_INDEX incx, float *y, CBLAS_INDEX incy)
{
    if (n < 0 || !x || !y)
    {
        assert(n > 0 && x && y);
        return;
    }

    float temp;
    for (CBLAS_INDEX i = 0; i < n; i++)
    {
        temp = *y;
        *y = *x;
        *x = temp;

        x += incx;
        y += incy;
    }
}

//------------------------------------------------------
//
//------------------------------------------------------
void cblas_dswap(CBLAS_INDEX n, double *x, CBLAS_INDEX incx, double *y, CBLAS_INDEX incy)
{
    if (n < 0 || !x || !y)
    {
        assert(n > 0 && x && y);
        return;
    }

    double temp;
    for (CBLAS_INDEX i = 0; i < n; i++)
    {
        temp = *y;
        *y = *x;
        *x = temp;

        x += incx;
        y += incy;
    }    
}
