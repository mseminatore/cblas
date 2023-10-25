//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include <assert.h>
#include "cblas.h"

//------------------------------------------------------
//
//------------------------------------------------------
void cblas_sscal(CBLAS_INDEX n, float a, float *x, CBLAS_INDEX incx)
{
    if (n <= 0 || !x || incx <= 0 || a == 1.0f)
    {
        assert(n >= 0 && x && incx > 0);
        return;
    }

    for (CBLAS_INDEX i = 0; i < n; i++)
    {
        *x = a * *x;
        x += incx;
    }
}

//------------------------------------------------------
//
//------------------------------------------------------
void cblas_dscal(CBLAS_INDEX n, double a, double *x, CBLAS_INDEX incx)
{
    if (n <= 0 || !x || incx <= 0 || a == 1.0)
    {
        assert(n >= 0 && x);
        return;
    }

    for (CBLAS_INDEX i = 0; i < n; i++)
    {
        *x = a * *x;
        x += incx;
    }    
}
