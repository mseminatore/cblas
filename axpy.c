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
#ifdef CBLAS_CHECK_INPUTS

#ifdef CBLAS_XERBLA_INPUTS
    int info = 0;
    if (n <= 0)
        info = 1;
    else if (alpha == 0.0f)
        info = 2;
    else if (!x)
        info = 3;
    else if (!y)
        info = 5;

    if (info) {
        XERBLA(info);
        return;
    }
#else
    if (n <= 0 || !x || !y || alpha == 0.0f)
    {
        assert(n > 0 && x && y && alpha != 0.0f);
        return;
    }
#endif
#endif

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
#ifdef CBLAS_CHECK_INPUTS

#ifdef CBLAS_XERBLA_INPUTS
    int info = 0;
    if (n <= 0)
        info = 1;
    else if (alpha == 0.0)
        info = 2;
    else if (!x)
        info = 3;
    else if (!y)
        info = 5;

    if (info) {
        XERBLA(info);
        return;
    }
#else
    if (n <= 0 || !x || !y || alpha == 0.0)
    {
        assert(n > 0 && x && y && alpha != 0.0);
        return;
    }
#endif
#endif

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
