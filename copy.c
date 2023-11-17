//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"

//------------------------------------------------------
// single precision copy kernel
//------------------------------------------------------
void cblas_scopy_k(cblas_args_t *args)
{
    float *x = args->x;
    float *y = args->y;

    for (CBLAS_INDEX i = 0; i < args->n; i++)
    {
        *x = *y;
        x += args->incx;
        y += args->incy;
    }
}

//------------------------------------------------------
//
//------------------------------------------------------
void cblas_level1_exec(kernel_function kernel, CBLAS_INDEX n, float* x, CBLAS_INDEX incx, const float* y, CBLAS_INDEX incy)
{
    work_queue_t queue[MAX_THREADS];
    cblas_args_t args[MAX_THREADS];

    int thread_count = cblas_get_num_threads();

    CBLAS_INDEX partition = n / thread_count;

    for (int i = 0; i < thread_count; i++)
    {
        args[i].n = partition;
        args[i].incx = incx;
        args[i].incy = incy;
        
        // compute partition starts based on partition size and increments
        args[i].x = x;
        args[i].y = y;

        queue[i].finished = 0;
        queue[i].args = &args[i];
        queue[i].kernel = kernel;
        queue[i].next = &queue[i + 1];
    }

    // mark end of task queue
    queue[thread_count - 1].next = NULL;

    cblas_execute(thread_count, queue);
}

//------------------------------------------------------
//
//------------------------------------------------------
void cblas_scopy(CBLAS_INDEX n, float *x, CBLAS_INDEX incx, const float *y, CBLAS_INDEX incy)
{
    if (n < 0 || !x || !y)
    {
        assert(n > 0 && x && y);
        return;
    }

#ifdef MT_ENABLED
    cblas_level1_exec(cblas_scopy_k, n, x, incx, y, incy);
#else
    for (CBLAS_INDEX i = 0; i < n; i++)
    {
        *x = *y;
        x += incx;
        y += incy;
    }
#endif
}

//------------------------------------------------------
//
//------------------------------------------------------
void cblas_dcopy(CBLAS_INDEX n, double *x, CBLAS_INDEX incx, const double *y, CBLAS_INDEX incy)
{
    if (n < 0 || !x || !y)
    {
        assert(n > 0 && x && y);
        return;
    }

    for (CBLAS_INDEX i = 0; i < n; i++)
    {
        *x = *y;
        x += incx;
        y += incy;
    }
}
