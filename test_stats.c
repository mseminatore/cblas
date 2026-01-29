//------------------------------------------------------
// test_stats.c - Test performance counter functionality
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include "cblas.h"
#include "test.h"

int main(void)
{
    printf("\n=== Testing Performance Counter Functionality ===\n\n");
    
    cblas_init(CBLAS_DEFAULT_THREADS);
    cblas_print_configuration();

    // Reset stats to start fresh
    cblas_reset_stats();
    
    // Test data
    float sx[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float sy[] = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    float sz[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    int n = 5;
    
    printf("Performing test operations...\n");
    
    // Test Level-1 operations
    printf("  Level-1 operations:\n");
    float result = cblas_sdot(n, sx, 1, sy, 1);
    printf("    sdot: %f\n", result);
    
    cblas_scopy(n, sx, 1, sz, 1);
    printf("    scopy: done\n");
    
    cblas_sswap(n, sx, 1, sy, 1);
    printf("    sswap: done\n");
    
    cblas_saxpy(n, 2.0f, sx, 1, sy, 1);
    printf("    saxpy: done\n");
    
    cblas_sscal(n, 2.0f, sx, 1);
    printf("    sscal: done\n");
    
    result = cblas_sasum(n, sx, 1);
    printf("    sasum: %f\n", result);
    
    result = cblas_snrm2(n, sx, 1);
    printf("    snrm2: %f\n", result);
    
    // Test Level-2 operation
    printf("  Level-2 operations:\n");
    float A[9] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    cblas_sger(CblasRowMajor, 3, 3, 1.0f, sx, 1, sy, 1, A, 3);
    printf("    sger: done\n");
    
    // Test Level-3 operation
    printf("  Level-3 operations:\n");
    float B[9] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    float C[9] = {0.0f};
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 3, 3, 1.0f, A, 3, B, 3, 0.0f, C, 3);
    printf("    sgemm: done\n");
    
    printf("\n");
    
    // Print all statistics
    cblas_print_stats();
    
    // Verify key operations were tracked
    printf("Verifying stats tracking:\n");
    const char* ops[] = {"sdot", "scopy", "sswap", "saxpy", "sscal", "sasum", "snrm2", "sger", "sgemm"};
    int num_ops = 9;
    int all_found = 1;
    
    for (int i = 0; i < num_ops; i++) {
        const cblas_stats_t* stats = cblas_get_stats(ops[i]);
        if (stats && stats->total_calls > 0) {
            printf("  " CHECK_MARK " %s tracked\n", ops[i]);
        } else {
            printf("  " X_MARK " %s NOT tracked\n", ops[i]);
            all_found = 0;
        }
    }
    
    if (!all_found) {
        printf("\nERROR: Not all operations were tracked!\n");
        return 1;
    }
    
    // Test reset
    printf("\nTesting cblas_reset_stats()...\n");
    cblas_reset_stats();
    
    const cblas_stats_t* stats = cblas_get_stats("sdot");
    if (stats && stats->total_calls == 0) {
        printf("  " CHECK_MARK " Stats successfully reset\n");
    } else {
        printf("  " X_MARK " Stats not properly reset\n");
        return 1;
    }
    
    printf("\n=== All Performance Counter Tests Passed! ===\n\n");
    
    cblas_shutdown();
    
    return 0;
}
