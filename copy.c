//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include <assert.h>
#include "cblas.h"

//------------------------------------------------------
//
//------------------------------------------------------
void cblas_scopy(CBLAS_INDEX n, float *x, CBLAS_INDEX incx, float *y, CBLAS_INDEX incy)
{
    if (!x || !y)
    {
        return;
        assert(x && y);
    }

}

//------------------------------------------------------
//
//------------------------------------------------------
void cblas_dcopy(CBLAS_INDEX n, double *x, CBLAS_INDEX incx, double *y, CBLAS_INDEX incy)
{
    if (!x || !y)
    {
        return;
        assert(x && y);
    }

}
