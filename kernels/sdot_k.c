//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"
#include "cblas_simd.h"

//------------------------------------------------------
// single-precision vector dot product kernel
//------------------------------------------------------
void cblas_sdot_k(cblas_args_t* args)
{
    float sum = 0.0f;
    float* x = args->x;
    float* y = args->y;
    float* result = args->c;
    register CBLAS_INDEX incx = args->incx, incy = args->incy, n = args->n;

    for (CBLAS_INDEX i = 0; i < n; i++)
    {
        sum += *x * *y;
        x += incx;
        y += incy;
    }

    // set return value
    *result = sum;
}
