//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"

//------------------------------------------------------
//
//------------------------------------------------------
void cblas_saxpy(CBLAS_INDEX n, float alpha, float *x, CBLAS_INDEX incx, float *y, CBLAS_INDEX incy)
{
    if (n <= 0 || !x || !y || alpha == 0.0f)
    {
        assert(n > 0 && x && y);
        return;
    }

    if (alpha == 1.0f)
    {
        for (CBLAS_INDEX i = 0; i < n; i++)
        {
            *y = *x + *y;
            x += incx;
            y += incy;
        }
    }
    else
    {
        for (CBLAS_INDEX i = 0; i < n; i++)
        {
            *y = alpha * *x + *y;
            x += incx;
            y += incy;
        }
    }
}

//------------------------------------------------------
//
//------------------------------------------------------
void cblas_daxpy(CBLAS_INDEX n, double alpha, double *x, CBLAS_INDEX incx, double *y, CBLAS_INDEX incy)
{
    if (n <= 0 || !x || !y || alpha == 0.0)
    {
        assert(n > 0 && x && y);
        return;
    }

    if (alpha == 1.0)
    {
        for (CBLAS_INDEX i = 0; i < n; i++)
        {
            *y = *x + *y;
            x += incx;
            y += incy;
        }
    }
    else
    {
        for (CBLAS_INDEX i = 0; i < n; i++)
        {
            *y = alpha * *x + *y;
            x += incx;
            y += incy;
        }
    }
}
