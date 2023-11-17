//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"

//------------------------------------------------------
//
//------------------------------------------------------
void cblas_sscal(CBLAS_INDEX n, float alpha, float *x, CBLAS_INDEX incx)
{
    if (n <= 0 || !x || incx <= 0)
    {
        assert(n >= 0 && x && incx > 0);
        return;
    }

    if (alpha == 1.0f)
    {
        // nothing to do!!
        return;
    }
    else
    {
        for (CBLAS_INDEX i = 0; i < n; i++)
        {
            *x = alpha * *x;
            x += incx;
        }
    }
}

//------------------------------------------------------
//
//------------------------------------------------------
void cblas_dscal(CBLAS_INDEX n, double alpha, double *x, CBLAS_INDEX incx)
{
    if (n <= 0 || !x || incx <= 0)
    {
        assert(n >= 0 && x && incx > 0);
        return;
    }

    if (alpha == 1.0)
    {
        // nothing to do!!
        return;
    }
    else
    {
        for (CBLAS_INDEX i = 0; i < n; i++)
        {
            *x = alpha * *x;
            x += incx;
        }
    }
}
