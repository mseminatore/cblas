//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include <math.h>
#include "cblas.h"

//------------------------------------------------------
//
//------------------------------------------------------
float cblas_snrm2(CBLAS_INDEX n, float *x, CBLAS_INDEX incx)
{
    float sum = 0.0f;

    if (n < 0 || !x)
    {
        assert(n > 0 && x);
        return sum;
    }

    for (CBLAS_INDEX i = 0; i < n; i++)
    {
        sum += *x * *x;
        x += incx;
    }

    return sqrtf(sum);
}

//------------------------------------------------------
//
//------------------------------------------------------
double cblas_dnrm2(CBLAS_INDEX n, double *x, CBLAS_INDEX incx)
{
    double sum = 0.0;

    if (n < 0 || !x)
    {
        assert(n > 0 && x);
        return sum;
    }

    for (CBLAS_INDEX i = 0; i < n; i++)
    {
        sum += *x * *x;
        x += incx;
    }

    return sqrt(sum);
}
