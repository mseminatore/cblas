//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include "cblas.h"

//
int cblas_max_threads = MAX_THREADS;    // max system supported threads
int cblas_set_threads = 1;              // set number of threads
int server_alive = 0;                   // has thread server been initialized

#ifdef _WIN32
#   define strcat(s1, s2) strcat_s((s1), sizeof(s1), (s2))
#endif

//------------------------------------------------------
// standard BLAS error handler
//
// srcname - name of the function that called xerbla
// info - position of the invalid parameter in the parameter list
// len - length of the name in bytes
//------------------------------------------------------
void xerbla(const char *srcname, int info, size_t len)
{
    printf("\nCBLAS error: parameter %d was invalid in call to %s()\n",info, srcname);
}

//------------------------------------------------------
// return the current config
//------------------------------------------------------
const char *cblas_get_config()
{
    static char buf[CBLAS_SMALL_BUF];

#ifdef _WIN32
    sprintf_s(buf, sizeof(buf), "\nCBLAS 0.1 %s MAX_THREADS=%d", cpu_get_core_name(), MAX_THREADS);
#else
    sprintf(buf, "\nCBLAS 0.1 %s MAX_THREADS=%d", cpu_get_core_name(), MAX_THREADS);
#endif
    
    return buf;
}

//------------------------------------------------------
// return bit flags defining available ISA extensions 
//------------------------------------------------------
const char *cblas_get_isa_features()
{
    static char buf[CBLAS_SMALL_BUF];

    unsigned int cpu = cpu_get_features();

    buf[0] = 0;

#ifdef __APPLE__
    if (cpu & CPU_NEON)
        strcat(buf, "NEON");

    if (cpu & CPU_NEON_FMA)
        strcat(buf, ", FMA");
#endif

#if defined(__x86_64__) || defined(_M_X64)
    if (cpu & CPU_SSE)
        strcat(buf, "SSE");
    if (cpu & CPU_AVX)
        strcat(buf, ", AVX");
    if (cpu & CPU_AVX2)
        strcat(buf, ", AVX2");
    if (cpu & CPU_x64_FMA3)
        strcat(buf, ", FMA");
#endif

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
//
//------------------------------------------------------
void cblas_print_configuration()
{
    printf("%s\n", cblas_get_config());
    printf("     CPU uArch: %s\n", cblas_get_corename());
	printf("ISA Extensions: %s\n", cblas_get_isa_features());
    printf(" Cores/Threads: %d/%d\n\n", cblas_get_num_procs(), cblas_get_num_threads());
}

//------------------------------------------------------
// leve1 1 dispatch
//------------------------------------------------------
void cblas_level1_exec(CBLAS_INDEX stride, kernel_function kernel, CBLAS_INDEX n, void *x, CBLAS_INDEX incx, void *y, CBLAS_INDEX incy)
{
    work_queue_t queue[MAX_THREADS];
    cblas_args_t args[MAX_THREADS];

    int thread_count = CLAMP(cblas_get_num_threads(), 1, MAX_THREADS);

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

        // TODO - the x/y is wrong when incx/incy is > 1
        x = (void*)((CBLAS_INDEX)x + partition_size * incx * stride);
        y = (void*)((CBLAS_INDEX)y + partition_size * incy * stride);

        queue[i].finished   = 0;
        queue[i].args       = &args[i];
        queue[i].kernel     = kernel;
        queue[i].next       = &queue[i + 1];
    }

    // mark end of task queue
    queue[thread_count - 1].next = NULL;

    // synchronously execute task queue
    cblas_execute(thread_count, queue);
}

//------------------------------------------------------
// initialize the CBLAS library
//------------------------------------------------------
void cblas_init(int threads)
{
    if (CBLAS_DEFAULT_THREADS == threads)
        threads = cpu_get_core_count();

    char *s_env_threads = getenv("CBLAS_THREADS");
    int env_threads = threads;

    if (s_env_threads)
        env_threads = atoi(s_env_threads);

    // make sure env_threads is valid;
    env_threads = MAX(env_threads, 1);

    threads = MIN(threads, env_threads);

#ifndef MT_ENABLED
    threads = 1;
#endif

    cblas_set_num_threads(threads);

    // TODO - detect cache sizes?
    // TODO - detect cpu features?

    // start thread server
#ifdef MT_ENABLED
    if (!server_alive && cblas_init_server())
        server_alive = 1;
#endif
}

//------------------------------------------------------
// return the active number of cblas threads
//------------------------------------------------------
int cblas_get_num_threads(void)
{
    return cblas_max_threads;
}
