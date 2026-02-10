//------------------------------------------------------
// Test program to demonstrate enhanced MT debug output
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include "cblas.h"

int main(void)
{
    // Initialize with 4 threads
    cblas_init(4);
    
    cblas_print_configuration();
    
    // Create large vectors to trigger multi-threading
    const size_t n = 500000;  // Much larger to ensure multiple threads work
    float *x = malloc(n * sizeof(float));
    float *y = malloc(n * sizeof(float));
    
    if (!x || !y) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }
    
    // Initialize vectors
    for (size_t i = 0; i < n; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }
    
    printf("Testing DOT product with %zu elements (MT threshold: %lu)...\n", n, (unsigned long)CBLAS_MT_DOT);
    
    // Perform dot product - should trigger multi-threading
    float result = cblas_sdot(n, x, 1, y, 1);
    
    printf("Result: %f (expected: %f)\n", result, (float)n * 2.0f);
    
    // Test COPY as well
    float *z = malloc(n * sizeof(float));
    if (z) {
        printf("\nTesting COPY with %zu elements (MT threshold: %lu)...\n", n, (unsigned long)CBLAS_MT_COPY);
        cblas_scopy(n, x, 1, z, 1);
        free(z);
    }
    
    // Cleanup
    free(x);
    free(y);
    cblas_shutdown();
    
    return 0;
}
