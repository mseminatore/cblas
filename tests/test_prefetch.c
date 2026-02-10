//------------------------------------------------------
//
// Test prefetching for Level-1 operations
//
//------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cblas.h"

#define LARGE_SIZE 200000  // Above CBLAS_PREFETCH_THRESHOLD
#define SMALL_SIZE 50000   // Below CBLAS_PREFETCH_THRESHOLD

static double get_time_sec()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void test_dot_prefetch()
{
    printf("\nTesting DOT with prefetching...\n");
    
    // Test with large vectors (should use prefetching)
    float *x_large = malloc(LARGE_SIZE * sizeof(float));
    float *y_large = malloc(LARGE_SIZE * sizeof(float));
    
    for (size_t i = 0; i < LARGE_SIZE; i++) {
        x_large[i] = 1.0f;
        y_large[i] = 2.0f;
    }
    
    double start = get_time_sec();
    float result = cblas_sdot(LARGE_SIZE, x_large, 1, y_large, 1);
    double elapsed = get_time_sec() - start;
    
    printf("  Large vector (n=%d): result=%.0f, time=%.6f sec\n", 
           LARGE_SIZE, result, elapsed);
    printf("  Expected result: %.0f\n", (float)LARGE_SIZE * 2.0f);
    
    if (result == (float)LARGE_SIZE * 2.0f) {
        printf("  ✓ Test passed\n");
    } else {
        printf("  ✗ Test failed\n");
    }
    
    free(x_large);
    free(y_large);
}

void test_axpy_prefetch()
{
    printf("\nTesting AXPY with prefetching...\n");
    
    // Test with large vectors (should use prefetching)
    float *x_large = malloc(LARGE_SIZE * sizeof(float));
    float *y_large = malloc(LARGE_SIZE * sizeof(float));
    
    for (size_t i = 0; i < LARGE_SIZE; i++) {
        x_large[i] = 1.0f;
        y_large[i] = 2.0f;
    }
    
    double start = get_time_sec();
    cblas_saxpy(LARGE_SIZE, 3.0f, x_large, 1, y_large, 1);
    double elapsed = get_time_sec() - start;
    
    printf("  Large vector (n=%d): time=%.6f sec\n", LARGE_SIZE, elapsed);
    
    // Verify result
    int correct = 1;
    for (size_t i = 0; i < LARGE_SIZE; i++) {
        if (y_large[i] != 5.0f) {  // 3*1 + 2 = 5
            correct = 0;
            break;
        }
    }
    
    if (correct) {
        printf("  ✓ Test passed\n");
    } else {
        printf("  ✗ Test failed\n");
    }
    
    free(x_large);
    free(y_large);
}

void test_copy_prefetch()
{
    printf("\nTesting COPY with prefetching...\n");
    
    // Test with large vectors (should use prefetching)
    float *x_large = malloc(LARGE_SIZE * sizeof(float));
    float *y_large = malloc(LARGE_SIZE * sizeof(float));
    
    for (size_t i = 0; i < LARGE_SIZE; i++) {
        x_large[i] = (float)i;
        y_large[i] = 0.0f;
    }
    
    double start = get_time_sec();
    cblas_scopy(LARGE_SIZE, x_large, 1, y_large, 1);
    double elapsed = get_time_sec() - start;
    
    printf("  Large vector (n=%d): time=%.6f sec\n", LARGE_SIZE, elapsed);
    
    // Verify result
    int correct = 1;
    for (size_t i = 0; i < LARGE_SIZE; i++) {
        if (y_large[i] != (float)i) {
            correct = 0;
            break;
        }
    }
    
    if (correct) {
        printf("  ✓ Test passed\n");
    } else {
        printf("  ✗ Test failed\n");
    }
    
    free(x_large);
    free(y_large);
}

int main()
{
    printf("==============================================\n");
    printf("Prefetching Test for Level-1 Operations\n");
    printf("==============================================\n");
    printf("Prefetch threshold: %d elements\n", CBLAS_PREFETCH_THRESHOLD);
    printf("Prefetch distance: %d elements\n", CBLAS_PREFETCH_DISTANCE);
    
    cblas_init(1);  // Initialize with 1 thread to test sequential code
    
    test_dot_prefetch();
    test_axpy_prefetch();
    test_copy_prefetch();
    
    cblas_shutdown();
    
    printf("\n==============================================\n");
    printf("All prefetching tests completed!\n");
    printf("==============================================\n");
    
    return 0;
}
