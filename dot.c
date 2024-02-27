//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"

//------------------------------------------------------
// Level-1 single-precision vector dot product
//------------------------------------------------------
float cblas_sdot(CBLAS_INDEX n, float *x, CBLAS_INDEX incx, float *y, CBLAS_INDEX incy)
{
    float sum = 0.0f;

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
        return sum;
    }
#else
    if (n <= 0 || !x || !y)
    {
        assert(n > 0 && x && y);
        return 0.0f;
    }
#endif  // CBLAS_XERBLA_INPUTS
#endif  // CBLAS_CHECK_INPUTS

    for (CBLAS_INDEX i = 0; i < n; i++)
    {
        sum += *x * *y;
        x += incx;
        y += incy;
    }

    return sum;
}

//------------------------------------------------------
// Level-1 double-precision vector dot product
//------------------------------------------------------
double cblas_ddot(CBLAS_INDEX n, double *x, CBLAS_INDEX incx, double *y, CBLAS_INDEX incy)
{
    double sum = 0.0;

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
        return sum;
    }
#else
    if (n <= 0 || !x || !y)
    {
        assert(n > 0 && x && y);
        return 0.0;
    }
#endif  // CBLAS_XERBLA_INPUTS
#endif  // CBLAS_CHECK_INPUTS

    for (CBLAS_INDEX i = 0; i < n; i++)
    {
        sum += *x * *y;
        x += incx;
        y += incy;
    }

    return sum;
}
