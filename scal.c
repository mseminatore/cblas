//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"

//------------------------------------------------------
// Level-1 single-precision vector scale
//------------------------------------------------------
void cblas_sscal(CBLAS_INDEX n, float alpha, float *x, CBLAS_INDEX incx)
{
    CBLAS_VALIDATE_SCAL(n, alpha, x, incx, );

    CBLAS_STATS_START();

    int mt_used = 0;

    if (alpha == 1.0f)
    {
        // nothing to do!!
        CBLAS_STATS_END("sscal", n, mt_used);
        return;
    }

    if (incx == 1)
    {
        for (CBLAS_INDEX i = 0; i < n; i++)
        {
            *x = alpha * *x;
            x++;
        }
    }
    else
    {
        for (CBLAS_INDEX i = 0; i < n; i++)
        {
            *x = alpha * *x;
            x += incx;
        }
    }

    CBLAS_STATS_END("sscal", n, mt_used);
}

//------------------------------------------------------
// Level-1 single-precision vector scale
//------------------------------------------------------
void cblas_dscal(CBLAS_INDEX n, double alpha, double* x, CBLAS_INDEX incx)
{
#ifdef CBLAS_CHECK_INPUTS

#ifdef CBLAS_XERBLA_INPUTS
    int info = 0;
    if (n <= 0)
        info = 1;
    else if (!x)
        info = 3;
    else if (incx <= 0)
        info = 4;

    if (info) {
        XERBLA(info);
        return;
    }
#else
    if (n <= 0 || !x || incx <= 0)
    {
        assert(n >= 0 && x && incx > 0);
        return;
    }
#endif
#endif

    CBLAS_STATS_START();

    int mt_used = 0;

    if (alpha == 1.0)
    {
        // nothing to do!!
        CBLAS_STATS_END("dscal", n, mt_used);
        return;
    }

    if (incx == 1)
    {
        for (CBLAS_INDEX i = 0; i < n; i++)
        {
            *x = alpha * *x;
            x++;
        }
    }
    else
    {
        for (CBLAS_INDEX i = 0; i < n; i++)
        {
            *x = alpha * *x;
            x += incx;
        }
    }

    CBLAS_STATS_END("dscal", n, mt_used);
}
