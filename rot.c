//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"

//------------------------------------------------------
// Level-1 single-precision generate rotation
//------------------------------------------------------
void cblas_srot(CBLAS_INDEX n, float *x, CBLAS_INDEX incx, float *y, CBLAS_INDEX incy, float c, float s)
{
#ifdef CBLAS_CHECK_INPUTS

#ifdef CBLAS_XERBLA_INPUTS
    int info = 0;
    if (n <= 0)
        info = 1;
    else if (!x)
        info = 2;
    else if (!y)
        info = 4;

    if (info) {
        XERBLA(info);
        return;
    }
#else
    if (n <= 0 || !x || !y)
    {
        assert(n > 0 && x && y);
        return;
    }
#endif  // CBLAS_XERBLA_INPUTS
#endif  // CBLAS_CHECK_INPUTS

    float temp;
    if (incx == 1 && incy == 1)
    {
        for (CBLAS_INDEX i = 0; i < n; i++)
        {
            temp = c * *x + s * *y;
            *y = c * *y - s * *x;
            *x = temp;

            x++;
            y++;
        }
    }
    else
    {
        for (CBLAS_INDEX i = 0; i < n; i++)
        {
            temp = c * *x + s * *y;
            *y = c * *y - s * *x;
            *x = temp;

            x += incx;
            y += incy;
        }
    }
}

//------------------------------------------------------
// Level-1 double-precision generate rotation
//------------------------------------------------------
void cblas_drot(CBLAS_INDEX n, double *x, CBLAS_INDEX incx, double *y, CBLAS_INDEX incy, double c, double s)
{
#ifdef CBLAS_CHECK_INPUTS

#ifdef CBLAS_XERBLA_INPUTS
    int info = 0;
    if (n <= 0)
        info = 1;
    else if (!x)
        info = 2;
    else if (!y)
        info = 4;

    if (info) {
        XERBLA(info);
        return;
    }
#else
    if (n <= 0 || !x || !y)
    {
        assert(n > 0 && x && y);
        return;
    }
#endif  // CBLAS_XERBLA_INPUTS
#endif  // CBLAS_CHECK_INPUTS

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
