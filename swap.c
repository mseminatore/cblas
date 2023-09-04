#include "cblas.h"

//
//
//
void cblas_sswap(CBLAS_INDEX n, float *x, CBLAS_INDEX incx, float *y, CBLAS_INDEX incy)
{
    if (!x || !y)
        return;

    float temp;
    for (CBLAS_INDEX i = 0; i < n; i++)
    {
        
    }
}

//
//
//
void cblas_dswap(CBLAS_INDEX n, double *x, CBLAS_INDEX incx, double *y, CBLAS_INDEX incy)
{
    if (!x || !y)
        return;
    
}
