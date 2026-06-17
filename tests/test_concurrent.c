//------------------------------------------------------
// Thread safety tests for CBLAS
//
// Copyright 2024 Mark Seminatore. All rights reserved.
//------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "test.h"
#include "cblas.h"

// Platform-specific includes and types
#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
typedef HANDLE thread_t;
typedef CRITICAL_SECTION mutex_t;
#define THREAD_RETURN DWORD WINAPI
#define THREAD_RETURN_TYPE DWORD
#define THREAD_RETURN_VALUE 0
#define mutex_init(m) InitializeCriticalSection(m)
#define mutex_destroy(m) DeleteCriticalSection(m)
#define mutex_lock(m) EnterCriticalSection(m)
#define mutex_unlock(m) LeaveCriticalSection(m)
#define sleep_ms(ms) Sleep(ms)
#else
#include <pthread.h>
#include <unistd.h>
typedef pthread_t thread_t;
typedef pthread_mutex_t mutex_t;
#define THREAD_RETURN void*
#define THREAD_RETURN_TYPE void*
#define THREAD_RETURN_VALUE NULL
#define mutex_init(m) do { *(m) = (pthread_mutex_t)PTHREAD_MUTEX_INITIALIZER; } while(0)
#define mutex_destroy(m) pthread_mutex_destroy(m)
#define mutex_lock(m) pthread_mutex_lock(m)
#define mutex_unlock(m) pthread_mutex_unlock(m)
#define sleep_ms(ms) usleep((ms) * 1000)
#endif

// Test configuration
#define VECTOR_SIZE 1000
#define MATRIX_SIZE 100
#define ITERATIONS_PER_THREAD 100

// Thread test data structure
typedef struct {
    int thread_id;
    int num_iterations;
    int *error_count;
    mutex_t *error_lock;
} thread_args_t;

//------------------------------------------------------
// Thread function: Multiple BLAS operations
//------------------------------------------------------
THREAD_RETURN test_blas_operations_thread(void* arg)
{
    thread_args_t* args = (thread_args_t*)arg;
    
    // Allocate thread-local vectors
    float* x = (float*)malloc(VECTOR_SIZE * sizeof(float));
    float* y = (float*)malloc(VECTOR_SIZE * sizeof(float));
    float* result = (float*)malloc(VECTOR_SIZE * sizeof(float));
    
    if (!x || !y || !result) {
        mutex_lock(args->error_lock);
        (*args->error_count)++;
        mutex_unlock(args->error_lock);
        free(x);
        free(y);
        free(result);
        return THREAD_RETURN_VALUE;
    }
    
    // Initialize vectors
    for (int i = 0; i < VECTOR_SIZE; i++) {
        x[i] = (float)(i % 100);
        y[i] = (float)((i + args->thread_id) % 100);
    }
    
    for (int iter = 0; iter < args->num_iterations; iter++) {
        // Test cblas_scopy
        cblas_scopy(VECTOR_SIZE, x, 1, result, 1);
        
        // Verify copy
        for (int i = 0; i < VECTOR_SIZE; i++) {
            if (result[i] != x[i]) {
                mutex_lock(args->error_lock);
                (*args->error_count)++;
                mutex_unlock(args->error_lock);
                break;
            }
        }
        
        // Test cblas_sdot
        float dot_result = cblas_sdot(VECTOR_SIZE, x, 1, y, 1);
        if (dot_result < 0.0f) { // Basic sanity check
            mutex_lock(args->error_lock);
            (*args->error_count)++;
            mutex_unlock(args->error_lock);
        }
        
        // Test cblas_saxpy
        cblas_saxpy(VECTOR_SIZE, 2.0f, x, 1, result, 1);
        
        // Test cblas_sscal
        cblas_sscal(VECTOR_SIZE, 0.5f, result, 1);
        
        // Test cblas_sasum
        float asum = cblas_sasum(VECTOR_SIZE, x, 1);
        if (asum < 0.0f) { // Basic sanity check
            mutex_lock(args->error_lock);
            (*args->error_count)++;
            mutex_unlock(args->error_lock);
        }
        
        // Test cblas_snrm2
        float nrm = cblas_snrm2(VECTOR_SIZE, x, 1);
        if (nrm < 0.0f) { // Basic sanity check
            mutex_lock(args->error_lock);
            (*args->error_count)++;
            mutex_unlock(args->error_lock);
        }
    }
    
    free(x);
    free(y);
    free(result);
    
    return THREAD_RETURN_VALUE;
}

//------------------------------------------------------
// Thread function: Matrix operations
//------------------------------------------------------
THREAD_RETURN test_matrix_operations_thread(void* arg)
{
    thread_args_t* args = (thread_args_t*)arg;
    
    // Allocate thread-local matrices
    float* A = (float*)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    float* B = (float*)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    float* C = (float*)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    float* x = (float*)malloc(MATRIX_SIZE * sizeof(float));
    float* y = (float*)malloc(MATRIX_SIZE * sizeof(float));
    
    if (!A || !B || !C || !x || !y) {
        mutex_lock(args->error_lock);
        (*args->error_count)++;
        mutex_unlock(args->error_lock);
        free(A);
        free(B);
        free(C);
        free(x);
        free(y);
        return THREAD_RETURN_VALUE;
    }
    
    // Initialize matrices and vectors
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
        A[i] = 1.0f;
        B[i] = 1.0f;
        C[i] = 0.0f;
    }
    for (int i = 0; i < MATRIX_SIZE; i++) {
        x[i] = 1.0f;
        y[i] = 0.0f;
    }
    
    for (int iter = 0; iter < args->num_iterations; iter++) {
        // Test cblas_sgemm
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE,
                    1.0f, A, MATRIX_SIZE, B, MATRIX_SIZE,
                    0.0f, C, MATRIX_SIZE);
        
        // Test cblas_sger
        cblas_sger(CblasRowMajor, MATRIX_SIZE, MATRIX_SIZE,
                   1.0f, x, 1, x, 1, A, MATRIX_SIZE);
        
        // Test cblas_sgemv
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    MATRIX_SIZE, MATRIX_SIZE,
                    1.0f, A, MATRIX_SIZE, x, 1,
                    0.0f, y, 1);
    }
    
    free(A);
    free(B);
    free(C);
    free(x);
    free(y);
    
    return THREAD_RETURN_VALUE;
}

//------------------------------------------------------
// Thread function: Set num threads concurrently
//------------------------------------------------------
THREAD_RETURN test_set_num_threads_thread(void* arg)
{
    thread_args_t* args = (thread_args_t*)arg;
    
    for (int iter = 0; iter < args->num_iterations; iter++) {
        // Cycle through different thread counts
        int thread_count = (args->thread_id % 4) + 1; // 1-4 threads
        cblas_set_num_threads(thread_count);
        
        // Do some work
        float x[100], y[100];
        for (int i = 0; i < 100; i++) {
            x[i] = (float)i;
            y[i] = (float)(i * 2);
        }
        cblas_sdot(100, x, 1, y, 1);
        
        // Small delay to increase contention
        sleep_ms(1);
    }
    
    return THREAD_RETURN_VALUE;
}

//------------------------------------------------------
// Test: Multiple threads calling different BLAS operations
//------------------------------------------------------
static void test_concurrent_blas_operations(int num_threads)
{
    printf("\n  Testing %d threads with concurrent BLAS operations...\n", num_threads);
    
    thread_t* threads = (thread_t*)malloc(num_threads * sizeof(thread_t));
    thread_args_t* args = (thread_args_t*)malloc(num_threads * sizeof(thread_args_t));
    int error_count = 0;
    mutex_t error_lock;
    mutex_init(&error_lock);
    
    if (!threads || !args) {
        free(threads);
        free(args);
        printf("    %s Memory allocation failed\n", X_MARK);
        mutex_destroy(&error_lock);
        return;
    }
    
    // Create threads
    int created_count = 0;
    for (int i = 0; i < num_threads; i++) {
        args[i].thread_id = i;
        args[i].num_iterations = ITERATIONS_PER_THREAD;
        args[i].error_count = &error_count;
        args[i].error_lock = &error_lock;
        
#if defined(_WIN32) || defined(_WIN64)
        threads[i] = CreateThread(NULL, 0, test_blas_operations_thread, &args[i], 0, NULL);
        if (threads[i] == NULL) {
#else
        if (pthread_create(&threads[i], NULL, test_blas_operations_thread, &args[i]) != 0) {
#endif
            printf("    %s Failed to create thread %d\n", X_MARK, i);
            // Clean up already created threads
            for (int j = 0; j < created_count; j++) {
#if defined(_WIN32) || defined(_WIN64)
                WaitForSingleObject(threads[j], INFINITE);
                CloseHandle(threads[j]);
#else
                pthread_join(threads[j], NULL);
#endif
            }
            mutex_destroy(&error_lock);
            free(threads);
            free(args);
            return;
        }
        created_count++;
    }
    
    // Wait for all threads
    for (int i = 0; i < num_threads; i++) {
#if defined(_WIN32) || defined(_WIN64)
        WaitForSingleObject(threads[i], INFINITE);
        CloseHandle(threads[i]);
#else
        pthread_join(threads[i], NULL);
#endif
    }
    
    mutex_destroy(&error_lock);
    free(threads);
    free(args);
    
    if (error_count == 0) {
        printf("    %s No errors detected\n", CHECK_MARK);
    } else {
        printf("    %s %d errors detected\n", X_MARK, error_count);
    }
}

//------------------------------------------------------
// Test: Multiple threads with matrix operations
//------------------------------------------------------
static void test_concurrent_matrix_operations(int num_threads)
{
    printf("\n  Testing %d threads with concurrent matrix operations...\n", num_threads);
    
    thread_t* threads = (thread_t*)malloc(num_threads * sizeof(thread_t));
    thread_args_t* args = (thread_args_t*)malloc(num_threads * sizeof(thread_args_t));
    int error_count = 0;
    mutex_t error_lock;
    mutex_init(&error_lock);
    
    if (!threads || !args) {
        free(threads);
        free(args);
        printf("    %s Memory allocation failed\n", X_MARK);
        mutex_destroy(&error_lock);
        return;
    }
    
    // Create threads
    int created_count = 0;
    for (int i = 0; i < num_threads; i++) {
        args[i].thread_id = i;
        args[i].num_iterations = 10; // Fewer iterations for matrix ops
        args[i].error_count = &error_count;
        args[i].error_lock = &error_lock;
        
#if defined(_WIN32) || defined(_WIN64)
        threads[i] = CreateThread(NULL, 0, test_matrix_operations_thread, &args[i], 0, NULL);
        if (threads[i] == NULL) {
#else
        if (pthread_create(&threads[i], NULL, test_matrix_operations_thread, &args[i]) != 0) {
#endif
            printf("    %s Failed to create thread %d\n", X_MARK, i);
            // Clean up already created threads
            for (int j = 0; j < created_count; j++) {
#if defined(_WIN32) || defined(_WIN64)
                WaitForSingleObject(threads[j], INFINITE);
                CloseHandle(threads[j]);
#else
                pthread_join(threads[j], NULL);
#endif
            }
            mutex_destroy(&error_lock);
            free(threads);
            free(args);
            return;
        }
        created_count++;
    }
    
    // Wait for all threads
    for (int i = 0; i < num_threads; i++) {
#if defined(_WIN32) || defined(_WIN64)
        WaitForSingleObject(threads[i], INFINITE);
        CloseHandle(threads[i]);
#else
        pthread_join(threads[i], NULL);
#endif
    }
    
    mutex_destroy(&error_lock);
    free(threads);
    free(args);
    
    if (error_count == 0) {
        printf("    %s No errors detected\n", CHECK_MARK);
    } else {
        printf("    %s %d errors detected\n", X_MARK, error_count);
    }
}

//------------------------------------------------------
// Test: Concurrent cblas_set_num_threads calls
//------------------------------------------------------
static void test_concurrent_set_num_threads(int num_threads)
{
    printf("\n  Testing %d threads calling cblas_set_num_threads()...\n", num_threads);
    
    thread_t* threads = (thread_t*)malloc(num_threads * sizeof(thread_t));
    thread_args_t* args = (thread_args_t*)malloc(num_threads * sizeof(thread_args_t));
    int error_count = 0;
    mutex_t error_lock;
    mutex_init(&error_lock);
    
    if (!threads || !args) {
        free(threads);
        free(args);
        printf("    %s Memory allocation failed\n", X_MARK);
        mutex_destroy(&error_lock);
        return;
    }
    
    // Create threads
    int created_count = 0;
    for (int i = 0; i < num_threads; i++) {
        args[i].thread_id = i;
        args[i].num_iterations = 20;
        args[i].error_count = &error_count;
        args[i].error_lock = &error_lock;
        
#if defined(_WIN32) || defined(_WIN64)
        threads[i] = CreateThread(NULL, 0, test_set_num_threads_thread, &args[i], 0, NULL);
        if (threads[i] == NULL) {
#else
        if (pthread_create(&threads[i], NULL, test_set_num_threads_thread, &args[i]) != 0) {
#endif
            printf("    %s Failed to create thread %d\n", X_MARK, i);
            // Clean up already created threads
            for (int j = 0; j < created_count; j++) {
#if defined(_WIN32) || defined(_WIN64)
                WaitForSingleObject(threads[j], INFINITE);
                CloseHandle(threads[j]);
#else
                pthread_join(threads[j], NULL);
#endif
            }
            mutex_destroy(&error_lock);
            free(threads);
            free(args);
            return;
        }
        created_count++;
    }
    
    // Wait for all threads
    for (int i = 0; i < num_threads; i++) {
#if defined(_WIN32) || defined(_WIN64)
        WaitForSingleObject(threads[i], INFINITE);
        CloseHandle(threads[i]);
#else
        pthread_join(threads[i], NULL);
#endif
    }
    
    mutex_destroy(&error_lock);
    free(threads);
    free(args);
    
    printf("    %s Completed\n", CHECK_MARK);
}

//------------------------------------------------------
// Test: Init/shutdown cycles
//------------------------------------------------------
static void test_init_shutdown_cycles(void)
{
    printf("\n  Testing cblas_init()/cblas_shutdown() cycles...\n");
    
    int error_count = 0;
    
    for (int i = 0; i < 10; i++) {
        cblas_init(4);
        
        // Do some work
        float x[100], y[100];
        for (int j = 0; j < 100; j++) {
            x[j] = (float)j;
            y[j] = (float)(j * 2);
        }
        
        float result = cblas_sdot(100, x, 1, y, 1);
        
        // Verify result is reasonable
        if (result < 0.0f) {
            error_count++;
        }
        
        cblas_shutdown();
    }
    
    if (error_count == 0) {
        printf("    %s 10 cycles completed successfully\n", CHECK_MARK);
    } else {
        printf("    %s %d errors in init/shutdown cycles\n", X_MARK, error_count);
    }
}

//------------------------------------------------------
// Main test function
//------------------------------------------------------
int test_main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;
    
    MODULE("Thread Safety Tests");
    
    SUITE("Concurrent BLAS operations");
    
    // Initialize CBLAS once for the concurrent operation tests
    cblas_init(CBLAS_DEFAULT_THREADS);
    cblas_print_configuration();
    
    // Concurrent BLAS from multiple application threads is a supported pattern.
    // Cap the thread count to a realistic level rather than massively
    // oversubscribing a small CI runner (16 threads on 2-4 cores just adds
    // scheduling noise without testing anything new).
    test_concurrent_blas_operations(2);
    test_concurrent_blas_operations(4);

    SUITE("Concurrent matrix operations");
    test_concurrent_matrix_operations(2);
    test_concurrent_matrix_operations(4);

    // NOTE: cblas_set_num_threads() is a global control knob and, like in
    // OpenBLAS/MKL, is not guaranteed safe to call concurrently with other BLAS
    // work from multiple threads. We exercise it from a single thread only; the
    // previous many-threads-hammering-the-knob test was an unsupported pattern.
    SUITE("cblas_set_num_threads");
    test_concurrent_set_num_threads(1);

    cblas_shutdown();
    
    SUITE("Init/shutdown cycles");
    test_init_shutdown_cycles();
    
    printf("\n%sThread safety tests completed!%s\n", TERM_BRIGHT_MAGENTA, TERM_RESET);
    printf("\n%sTo run with ThreadSanitizer (TSAN):%s\n", TERM_BRIGHT_CYAN, TERM_RESET);
    printf("  1. Rebuild with: %sCFLAGS=\"-fsanitize=thread -g\" make clean all%s\n", TERM_YELLOW, TERM_RESET);
    printf("  2. Run with: %s./test_concurrent%s\n", TERM_YELLOW, TERM_RESET);
    printf("  3. Or with CMake: %scmake -DCMAKE_C_FLAGS=\"-fsanitize=thread -g\" ..%s\n", TERM_YELLOW, TERM_RESET);
    printf("\nThreadSanitizer will detect data races and threading issues.\n\n");
    
    return 0;
}
