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
#ifdef CBLAS_CHECK_INPUTS

#ifdef CBLAS_XERBLA_INPUTS
    int info = 0;
    if (n <= 0)
        info = 1;
    else if (alpha == 0.0f)
        info = 2;
    else if (!x)
        info = 3;
    else if (beta == 0.0f)
        info = 5;
    else if (!y)
        info = 6;

    if (info) {
        XERBLA(info);
        return;
    }
#else
    if (n <= 0 || !x || !y || alpha == 0.0f || beta == 0.0f)
    {
        assert(n > 0 && x && y && alpha != 0.0f && beta != 0.0f);
        return;
    }
#endif
#endif

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
}

//------------------------------------------------------
// Level-1 double-precision y = A * x + B * y
//------------------------------------------------------
void cblas_daxpby(CBLAS_INDEX n, double alpha, double *x, CBLAS_INDEX incx, double beta, double *y, CBLAS_INDEX incy)
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
    else if (beta == 0.0)
        info = 5;
    else if (!y)
        info = 6;

    if (info) {
        XERBLA(info);
        return;
    }
#else
    if (n <= 0 || !x || !y || alpha == 0.0 || beta == 0.0)
    {
        assert(n > 0 && x && y && alpha != 0.0 && beta != 0.0);
        return;
    }
#endif
#endif

    for (CBLAS_INDEX i = 0; i < n; i++)
    {
        *y = alpha * *x + beta * *y;
        x += incx;
        y += incy;
    }
}
