//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"

//------------------------------------------------------
// Level-1 single-precision y = A * x + B * y
//------------------------------------------------------
void cblas_saxpby(CBLAS_INDEX n, float alpha, float *x, CBLAS_INDEX incx, float beta, float *y, CBLAS_INDEX incy)
{
    CBLAS_VALIDATE_AXPBY(n, alpha, x, incx, beta, y, incy, );

    CBLAS_STATS_START();

    int mt_used = 0;

    if (alpha == 1.0 && beta == 1.0)
    {
        for (CBLAS_INDEX i = 0; i < n; i++)
        {
            *y += *x;
            x += incx;
            y += incy;
        }
    }
    else if (alpha == 1.0)
    {
        for (CBLAS_INDEX i = 0; i < n; i++)
        {
            *y = *x + beta * *y;
            x += incx;
            y += incy;
        }
    }
    else if (beta == 1.0)
    {
        for (CBLAS_INDEX i = 0; i < n; i++)
        {
            *y = alpha * *x + *y;
            x += incx;
            y += incy;
        }
    }
    else // general case
    {
        for (CBLAS_INDEX i = 0; i < n; i++)
        {
            *y = alpha * *x + beta * *y;
            x += incx;
            y += incy;
        }
    }

    CBLAS_STATS_END("saxpby", n, mt_used);
}

//------------------------------------------------------
// Level-1 double-precision y = A * x + B * y
//------------------------------------------------------
void cblas_daxpby(CBLAS_INDEX n, double alpha, double *x, CBLAS_INDEX incx, double beta, double *y, CBLAS_INDEX incy)
{
    CBLAS_VALIDATE_AXPBY(n, alpha, x, incx, beta, y, incy, );

    CBLAS_STATS_START();

    int mt_used = 0;

    for (CBLAS_INDEX i = 0; i < n; i++)
    {
        *y = alpha * *x + beta * *y;
        x += incx;
        y += incy;
    }

    CBLAS_STATS_END("daxpby", n, mt_used);
}
