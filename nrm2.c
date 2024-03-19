//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

//#include <math.h>
#include "cblas.h"

//------------------------------------------------------
// Level-1 single-precision euclidean norm
//------------------------------------------------------
float cblas_snrm2(CBLAS_INDEX n, float *x, CBLAS_INDEX incx)
{
    float sum = 0.0f;

#ifdef CBLAS_CHECK_INPUTS

#ifdef CBLAS_XERBLA_INPUTS
    int info = 0;
    if (n <= 0)
        info = 1;
    else if (!x)
        info = 2;

    if (info) {
        XERBLA(info);
        return sum;
    }
#else

    if (n <= 0 || !x)
    {
        assert(n > 0 && x);
        return sum;
    }
#endif  // CBLAS_XERBLA_INPUTS
#endif  // CBLAS_CHECK_INPUTS

    for (CBLAS_INDEX i = 0; i < n; i++)
    {
        sum += *x * *x;
        x += incx;
    }

    return sqrtf(sum);
}

//------------------------------------------------------
// Level-1 single-precision euclidean norm
//------------------------------------------------------
double cblas_dnrm2(CBLAS_INDEX n, double *x, CBLAS_INDEX incx)
{
    double sum = 0.0;

#ifdef CBLAS_CHECK_INPUTS

#ifdef CBLAS_XERBLA_INPUTS
    int info = 0;
    if (n <= 0)
        info = 1;
    else if (!x)
        info = 2;

    if (info) {
        XERBLA(info);
        return sum;
    }
#else
    if (n <= 0 || !x)
    {
        assert(n > 0 && x);
        return sum;
    }
#endif
#endif

    for (CBLAS_INDEX i = 0; i < n; i++)
    {
        sum += *x * *x;
        x += incx;
    }

    return sqrt(sum);
}
