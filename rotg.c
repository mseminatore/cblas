//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"

//------------------------------------------------------
// Level-1 single-precision generate rotation
//------------------------------------------------------
void cblas_srotg(float *a, float *b, float *c, float *s)
{
#ifdef CBLAS_CHECK_INPUTS

#ifdef CBLAS_XERBLA_INPUTS
    int info = 0;
    if (!a)
        info = 1;
    else if (!b)
        info = 2;
    else if (!c)
        info = 3;
    else if (!s)
        info = 4;

    if (info) {
        XERBLA(info);
        return;
    }
#else
    if (!a || !b || !c || !s)
    {
        assert(a && b && c && s);
        return;
    }
#endif
#endif

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
// Level-1 double-precision generate rotation
//------------------------------------------------------
void cblas_drotg(double *a, double *b, double *c, double *s)
{
#ifdef CBLAS_CHECK_INPUTS

#ifdef CBLAS_XERBLA_INPUTS
    int info = 0;
    if (!a)
        info = 1;
    else if (!b)
        info = 2;
    else if (!c)
        info = 3;
    else if (!s)
        info = 4;

    if (info) {
        XERBLA(info);
        return;
    }
#else
    if (!a || !b || !c || !s)
    {
        assert(a && b && c && s);
        return;
    }
#endif
#endif

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
