//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include <assert.h>
#include "cblas.h"

//------------------------------------------------------
//
//------------------------------------------------------
void cblas_scopy(CBLAS_INDEX n, float *x, CBLAS_INDEX incx, const float *y, CBLAS_INDEX incy)
{
    if (n < 0 || !x || !y)
    {
        assert(n > 0 && x && y);
        return;
    }

    for (CBLAS_INDEX i = 0; i < n; i++)
    {
        *x = *y;
        x += incx;
        y += incy;
    }
}

//------------------------------------------------------
//
//------------------------------------------------------
void cblas_dcopy(CBLAS_INDEX n, double *x, CBLAS_INDEX incx, const double *y, CBLAS_INDEX incy)
{
    if (n < 0 || !x || !y)
    {
        assert(n > 0 && x && y);
        return;
    }

    for (CBLAS_INDEX i = 0; i < n; i++)
    {
        *x = *y;
        x += incx;
        y += incy;
    }
}
