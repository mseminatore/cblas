//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include <assert.h>
#include "cblas.h"

//------------------------------------------------------
//
//------------------------------------------------------
void cblas_srotg(float *a, float *b, float *c, float *s)
{
    if (!a || !b || !c || !s)
    {
        assert(a && b && c && s);
        return;
    }

    // float temp;
    // for (CBLAS_INDEX i = 0; i < n; i++)
    // {
    //     temp = c * *x + s * *y;
    //     *y = c * *y - s * *x;
    //     *x = temp;
    //     x += incx;
    //     y += incy;
    // }
}

//------------------------------------------------------
//
//------------------------------------------------------
    void cblas_drotg(double *a, double *b, double *c, double *s)
{
    if (!a || !b || !c || !s)
    {
        assert(a && b && c && s);
        return;
    }

    // double temp;
    // for (CBLAS_INDEX i = 0; i < n; i++)
    // {
    //     temp = c * *x + s * *y;
    //     *y = c * *y - s * *x;
    //     *x = temp;
    //     x += incx;
    //     y += incy;
    // }
}
