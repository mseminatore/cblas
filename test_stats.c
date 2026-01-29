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
    
    // Reset stats to start fresh
    cblas_reset_stats();
    
    // Test data
    float sx[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float sy[] = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    int n = 5;
    
    // Perform some operations
    printf("Performing test operations...\n");
    
    // Call sdot a few times
    float result = cblas_sdot(n, sx, 1, sy, 1);
    printf("  sdot result: %f\n", result);
    
    result = cblas_sdot(n, sx, 1, sy, 1);
    printf("  sdot result: %f\n", result);
    
    result = cblas_sdot(n, sx, 1, sy, 1);
    printf("  sdot result: %f\n", result);
    
    // Call ddot
    double dx[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double dy[] = {5.0, 4.0, 3.0, 2.0, 1.0};
    
    double dresult = cblas_ddot(n, dx, 1, dy, 1);
    printf("  ddot result: %f\n", dresult);
    
    printf("\n");
    
    // Print statistics
    cblas_print_stats();
    
    // Test individual stat retrieval
    printf("Testing cblas_get_stats():\n");
    const cblas_stats_t* stats = cblas_get_stats("sdot");
    if (stats) {
        printf("  sdot: calls=%llu, elements=%llu, mt_activations=%llu\n",
               (unsigned long long)stats->total_calls,
               (unsigned long long)stats->total_elements,
               (unsigned long long)stats->mt_activations);
        
        if (stats->total_calls != 3) {
            printf("  ERROR: Expected 3 calls, got %llu\n", (unsigned long long)stats->total_calls);
            return 1;
        }
        if (stats->total_elements != 15) {
            printf("  ERROR: Expected 15 elements, got %llu\n", (unsigned long long)stats->total_elements);
            return 1;
        }
    } else {
        printf("  ERROR: Could not retrieve stats for sdot\n");
        return 1;
    }
    
    stats = cblas_get_stats("ddot");
    if (stats) {
        printf("  ddot: calls=%llu, elements=%llu, mt_activations=%llu\n",
               (unsigned long long)stats->total_calls,
               (unsigned long long)stats->total_elements,
               (unsigned long long)stats->mt_activations);
        
        if (stats->total_calls != 1) {
            printf("  ERROR: Expected 1 call, got %llu\n", (unsigned long long)stats->total_calls);
            return 1;
        }
    } else {
        printf("  ERROR: Could not retrieve stats for ddot\n");
        return 1;
    }
    
    // Test reset
    printf("\nTesting cblas_reset_stats()...\n");
    cblas_reset_stats();
    
    stats = cblas_get_stats("sdot");
    if (stats && stats->total_calls == 0) {
        printf("  Stats successfully reset\n");
    } else {
        printf("  ERROR: Stats not properly reset\n");
        return 1;
    }
    
    printf("\n=== All Performance Counter Tests Passed! ===\n\n");
    
    cblas_shutdown();
    
    return 0;
}
