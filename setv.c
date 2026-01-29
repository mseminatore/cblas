//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"

//------------------------------------------------------
// single precision set vector
//------------------------------------------------------
void cblas_ssetv(CBLAS_INDEX n, float *x, float v)
{
    CBLAS_VALIDATE_VEC1(n, x, 1, );

	for (CBLAS_INDEX i = 0; i < n; i++)
		*x++ = v;
}

//------------------------------------------------------
// double precision set vector
//------------------------------------------------------
void cblas_dsetv(CBLAS_INDEX n, double *x, double v)
{
    CBLAS_VALIDATE_VEC1(n, x, 1, );

    for (CBLAS_INDEX i = 0; i < n; i++)
		*x++ = v;
}
