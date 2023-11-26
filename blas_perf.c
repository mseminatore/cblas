//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------
#include <stdio.h>
#include <string.h>
#include <time.h>
//#include "test.h"
#include "cblas.h"


//------------------------------------------------------
//
//------------------------------------------------------
int main(int argc, char *argv[])
{
	cblas_init();

    printf( "%s\n", cblas_get_config());
    printf("      CPU uArch: %s\n", cblas_get_corename());
    printf("  Cores/Threads: %d/%d\n", cblas_get_num_procs(), cblas_get_num_threads());
	
	return 0;
}