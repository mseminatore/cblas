//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include <assert.h>
#include "cblas.h"

//------------------------------------------------------
//
//------------------------------------------------------
float cblas_sasum(CBLAS_INDEX n, float *x, CBLAS_INDEX incx)
{
    float sum = 0.0f;

    if (n < 0 || !x)
    {
        assert(n > 0 && x);
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
//
//------------------------------------------------------
double cblas_dasum(CBLAS_INDEX n, double *x, CBLAS_INDEX incx)
{
    double sum = 0.0;

    if (n < 0 || !x)
    {
        assert(n > 0 && x);
        return sum;
    }

    for (CBLAS_INDEX i = 0; i < n; i++)
    {
        sum += fabs(*x);
        x += incx;
    }

    return sum;
}
