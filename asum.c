//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"

//------------------------------------------------------
// Level-1 single-precision vector sum
//------------------------------------------------------
float cblas_sasum(CBLAS_INDEX n, float *x, CBLAS_INDEX incx)
{
    float sum = 0.0f;

#ifdef CBLAS_XERBLA_INPUTS
    int info = 0;
    if (n < 0)
        info = 1;
    else if (!x)
        info = 2;
    else if (incx == 0)
        info = 3;

    if (info) {
        XERBLA(info);
        return sum;
    }
#endif

    if (n < 0 || !x || incx == 0)
    {
        assert(n > 0 && x && incx != 0);
        return sum;
    }

    for (CBLAS_INDEX i = 0; i < n; i++)
    {
        sum += fabsf(*x);
        x += incx;
    }

    return sum;
}

//------------------------------------------------------
// Level-1 double-precision vector sum
//------------------------------------------------------
double cblas_dasum(CBLAS_INDEX n, double *x, CBLAS_INDEX incx)
{
    double sum = 0.0;

#ifdef CBLAS_XERBLA_INPUTS
    int info = 0;
    if (n < 0)
        info = 1;
    else if (!x)
        info = 2;
    else if (incx == 0)
        info = 3;

    if (info) {
        XERBLA(info);
        return sum;
    }
#endif

    if (n < 0 || !x || incx == 0)
    {
        assert(n > 0 && x && incx != 0);
        return sum;
    }

    for (CBLAS_INDEX i = 0; i < n; i++)
    {
        sum += fabs(*x);
        x += incx;
    }

    return sum;
}
