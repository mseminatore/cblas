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
#endif  // CBLAS_XERBLA_INPUTS
#endif  // CBLAS_CHECK_INPUTS

    CBLAS_STATS_START();

    int mt_used = 0;

    float r, roe, scale, z;
    
    roe = *b;
    if (fabsf(*a) > fabsf(*b))
        roe = *a;
    
    scale = fabsf(*a) + fabsf(*b);
    
    if (scale == 0.0f) {
        *c = 1.0f;
        *s = 0.0f;
        r = 0.0f;
        z = 0.0f;
    } else {
        r = scale * sqrtf((*a / scale) * (*a / scale) + (*b / scale) * (*b / scale));
        r = copysignf(r, roe);
        *c = *a / r;
        *s = *b / r;
        z = 1.0f;
        if (fabsf(*a) > fabsf(*b))
            z = *s;
        if (fabsf(*b) >= fabsf(*a) && *c != 0.0f)
            z = 1.0f / *c;
    }
    
    *a = r;
    *b = z;

    CBLAS_STATS_END("srotg", 1, mt_used);
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
#endif  // CBLAS_XERBLA_INPUTS
#endif  // CBLAS_CHECK_INPUTS

    CBLAS_STATS_START();

    int mt_used = 0;

    double r, roe, scale, z;
    
    roe = *b;
    if (fabs(*a) > fabs(*b))
        roe = *a;
    
    scale = fabs(*a) + fabs(*b);
    
    if (scale == 0.0) {
        *c = 1.0;
        *s = 0.0;
        r = 0.0;
        z = 0.0;
    } else {
        r = scale * sqrt((*a / scale) * (*a / scale) + (*b / scale) * (*b / scale));
        r = copysign(r, roe);
        *c = *a / r;
        *s = *b / r;
        z = 1.0;
        if (fabs(*a) > fabs(*b))
            z = *s;
        if (fabs(*b) >= fabs(*a) && *c != 0.0)
            z = 1.0 / *c;
    }
    
    *a = r;
    *b = z;

    CBLAS_STATS_END("drotg", 1, mt_used);
}
