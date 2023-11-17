//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"

//------------------------------------------------------
//
//------------------------------------------------------
void cblas_srot(CBLAS_INDEX n, float *x, CBLAS_INDEX incx, float *y, CBLAS_INDEX incy, float c, float s)
{
    if (n < 0 || !x || !y)
    {
        assert(n > 0 && x && y);
        return;
    }

    float temp;
    for (CBLAS_INDEX i = 0; i < n; i++)
    {
        temp = c * *x + s * *y;
        *y = c * *y - s * *x;
        *x = temp;
        x += incx;
        y += incy;
    }
}

//------------------------------------------------------
//
//------------------------------------------------------
void cblas_drot(CBLAS_INDEX n, double *x, CBLAS_INDEX incx, double *y, CBLAS_INDEX incy, double c, double s)
{
    if (n < 0 || !x || !y)
    {
        assert(n > 0 && x && y);
        return;
    }

    double temp;
    for (CBLAS_INDEX i = 0; i < n; i++)
    {
        temp = c * *x + s * *y;
        *y = c * *y - s * *x;
        *x = temp;
        x += incx;
        y += incy;
    }
}
