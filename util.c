//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------
#include <stdio.h>
#include "cblas.h"

//
int cblas_max_threads = MAX_THREADS;

//------------------------------------------------------
// return the current config
//------------------------------------------------------
const char *cblas_get_config()
{
    static char buf[256];
    sprintf(buf, "\nCBLAS 0.1 %s MAX_THREADS=%d", cpu_get_core_name(), MAX_THREADS);
    return buf;
}

//------------------------------------------------------
// return the name of the host CPU
//------------------------------------------------------
const char *cblas_get_corename()
{
    return cpu_get_core_name();
}

//------------------------------------------------------
// return number of CPUs on the host system
//------------------------------------------------------
int cblas_get_num_procs()
{
    return cpu_get_core_count();
}

//------------------------------------------------------
// leve1 1 dispatch
//------------------------------------------------------
void cblas_level1_exec(CBLAS_INDEX stride, kernel_function kernel, CBLAS_INDEX n, void *x, CBLAS_INDEX incx, void *y, CBLAS_INDEX incy)
{
    work_queue_t queue[MAX_THREADS];
    cblas_args_t args[MAX_THREADS];

    int thread_count = cblas_get_num_threads();

    for (int i = 0; i < thread_count; i++)
    {
        args[i].incx = incx;
        args[i].incy = incy;

        // compute partition starts based on remaining task size and remaining threads
        CBLAS_INDEX partition_size = (n + thread_count - i - 1) / (thread_count - i);

        args[i].n = partition_size;
        args[i].x = x;
        args[i].y = y;

        n -= partition_size;

        x = (void*)((CBLAS_INDEX)x + partition_size * incx * stride);
        y = (void*)((CBLAS_INDEX)y + partition_size * incy * stride);

        queue[i].finished = 0;
        queue[i].args = &args[i];
        queue[i].kernel = kernel;
        queue[i].next = &queue[i + 1];
    }

    // mark end of task queue
    queue[thread_count - 1].next = NULL;

    // synchronously execute task queue
    cblas_execute(thread_count, queue);
}

//------------------------------------------------------
// initialize the CBLAS library
//------------------------------------------------------
void cblas_init()
{
    cblas_set_num_threads(cpu_get_core_count());
    // printf("%s\n", cblas_get_config());
    // printf("Cores/Threads: %d/%d\n", cblas_get_num_procs(), cblas_get_num_threads());

    // TODO - detect cache sizes?
    // TODO - detect cpu features?

    // start server
    cblas_init_server();
}

//------------------------------------------------------
// set the active number of threads
//------------------------------------------------------
void cblas_set_num_threads(int threads)
{
//    printf("set threads = %d\n", threads);

    if (threads < 1)
        threads = 1;
        
    if (threads > MAX_THREADS)
        threads = MAX_THREADS;

    // TODO - need code for if we are adding more threads!!
    // if (threads > cblas_max_threads)
    //     ;

    cblas_max_threads = threads;
}

//------------------------------------------------------
// return the active number of cblas threads
//------------------------------------------------------
int cblas_get_num_threads(void)
{
    return cblas_max_threads;
}
