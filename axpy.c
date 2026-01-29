//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"

//------------------------------------------------------
// Level-1 single-precision y = a * x + y
//------------------------------------------------------
void cblas_saxpy(CBLAS_INDEX n, float alpha, float *x, CBLAS_INDEX incx, float *y, CBLAS_INDEX incy)
{
    CBLAS_VALIDATE_AXPY(n, alpha, x, incx, y, incy, );

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
// Level-1 double-precision y = a * x + y
//------------------------------------------------------
void cblas_daxpy(CBLAS_INDEX n, double alpha, double *x, CBLAS_INDEX incx, double *y, CBLAS_INDEX incy)
{
    CBLAS_VALIDATE_AXPY(n, alpha, x, incx, y, incy, );

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
