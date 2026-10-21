//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------
//
// Print the detected CPU features, core/thread counts and block sizes.
// Run in CI so every job log records what hardware it ran on.
//
#include <stdio.h>
#include "cblas.h"

int main(void)
{
    cblas_init(CBLAS_DEFAULT_THREADS);
    cblas_print_configuration();
    cblas_shutdown();

    return 0;
}
