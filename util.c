#include "cblas.h"

//
int cblas_max_threads = MAX_THREADS;

//------------------------------------------------------
//
//------------------------------------------------------
void cblas_init()
{
    // TODO - detect CPU

    // start server
    cblas_init_server();
}

//------------------------------------------------------
//
//------------------------------------------------------
void cblas_set_num_threads(int threads)
{
    if (threads < 1)
        threads = 1;
        
    if (threads > MAX_THREADS)
        threads = MAX_THREADS;

    cblas_max_threads = threads;
}

//------------------------------------------------------
//
//------------------------------------------------------
int cblas_get_num_threads(void)
{
    return cblas_max_threads;
}
