//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"

//------------------------------------------------------
//
//------------------------------------------------------
void cblas_ssetv(CBLAS_INDEX n, float *x, float v)
{
#ifdef CBLAS_CHECK_INPUTS

#ifdef CBLAS_XERBLA_INPUTS
    int info = 0;
    if (n <= 0)
        info = 1;
    else if (!x)
        info = 2;

    if (info) {
        XERBLA(info);
        return;
    }
#else
    if (n <= 0 || !x)
    {
        assert(n > 0 && x);
        return;
    }
#endif
#endif

	for (CBLAS_INDEX i = 0; i < n; i++)
		*x++ = v;
}

//------------------------------------------------------
//
//------------------------------------------------------
void cblas_dsetv(CBLAS_INDEX n, double *x, double v)
{
#ifdef CBLAS_CHECK_INPUTS

#ifdef CBLAS_XERBLA_INPUTS
    int info = 0;
    if (n <= 0)
        info = 1;
    else if (!x)
        info = 2;

    if (info) {
        XERBLA(info);
        return;
    }
#else
    if (n <= 0 || !x)
    {
        assert(n > 0 && x);
        return;
    }
#endif
#endif

    for (CBLAS_INDEX i = 0; i < n; i++)
		*x++ = v;
}