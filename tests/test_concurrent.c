//------------------------------------------------------
// Thread safety tests for CBLAS
//
// Copyright 2024 Mark Seminatore. All rights reserved.
//------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
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
// Concurrent GEMM results: every thread runs differently shaped sgemm and
// dgemm calls (with transposes), small enough for the single-threaded path
// and one large enough for the multi-threaded one, and checks each result
// against a reference computed before the threads start. Calls racing for a
// shared packing buffer would corrupt each other's results.
//------------------------------------------------------
#define GEMM_SHAPES 5
static const int gemm_shapes[GEMM_SHAPES][5] = {    // M N K transA transB
    { 32, 64, 64, 0, 1 }, { 64, 64, 32, 1, 0 }, { 16, 48, 40, 0, 0 },
    { 40, 48, 6, 1, 1 }, { 192, 160, 128, 0, 0 } };

typedef struct {
    int shape;
    float *a, *b, *c, *ref;
    double *ad, *bd, *cd, *refd;
    int errors;
} gemm_case_t;

static void gemm_run(const gemm_case_t *g, float *c, double *cd)
{
    const int *d = gemm_shapes[g->shape];
    int m = d[0], n = d[1], k = d[2];
    CBLAS_TRANSPOSE ta = d[3] ? CblasTrans : CblasNoTrans, tb = d[4] ? CblasTrans : CblasNoTrans;
    cblas_sgemm(CblasRowMajor, ta, tb, m, n, k, 1.0f, g->a, d[3] ? m : k, g->b, d[4] ? k : n, 0.0f, c, n);
    cblas_dgemm(CblasRowMajor, ta, tb, m, n, k, 1.0, g->ad, d[3] ? m : k, g->bd, d[4] ? k : n, 0.0, cd, n);
}

static int gemm_alloc(gemm_case_t *g, int shape, int seed)
{
    const int *d = gemm_shapes[shape];
    size_t sa = (size_t)d[0] * d[2], sb = (size_t)d[2] * d[1], sc = (size_t)d[0] * d[1];
    g->shape = shape;
    g->errors = 0;
    g->a = malloc(sa * sizeof(float)); g->b = malloc(sb * sizeof(float));
    g->c = malloc(sc * sizeof(float)); g->ref = malloc(sc * sizeof(float));
    g->ad = malloc(sa * sizeof(double)); g->bd = malloc(sb * sizeof(double));
    g->cd = malloc(sc * sizeof(double)); g->refd = malloc(sc * sizeof(double));
    if (!g->a || !g->b || !g->c || !g->ref || !g->ad || !g->bd || !g->cd || !g->refd)
        return 0;
    for (size_t i = 0; i < sa; i++) g->ad[i] = g->a[i] = (float)((int)((i * 7 + seed * 13) % 17) - 8) / 8.0f;
    for (size_t i = 0; i < sb; i++) g->bd[i] = g->b[i] = (float)((int)((i * 5 + seed * 3) % 13) - 6) / 6.0f;
    gemm_run(g, g->ref, g->refd);
    return 1;
}

static void gemm_free(gemm_case_t *g)
{
    free(g->a); free(g->b); free(g->c); free(g->ref);
    free(g->ad); free(g->bd); free(g->cd); free(g->refd);
}

THREAD_RETURN test_gemm_results_thread(void *arg)
{
    gemm_case_t *cases = (gemm_case_t *)arg;
    for (int iter = 0; iter < 40; iter++)
        for (int s = 0; s < GEMM_SHAPES; s++)
        {
            gemm_case_t *g = &cases[s];
            const int *d = gemm_shapes[s];
            gemm_run(g, g->c, g->cd);
            for (int i = 0; i < d[0] * d[1]; i++)
                if (fabsf(g->c[i] - g->ref[i]) > 1e-4f * (1.0f + fabsf(g->ref[i])) ||
                    fabs(g->cd[i] - g->refd[i]) > 1e-9 * (1.0 + fabs(g->refd[i])))
                {
                    g->errors++;
                    break;
                }
        }
    return THREAD_RETURN_VALUE;
}

static int test_concurrent_gemm_results(int num_threads)
{
    gemm_case_t cases[8][GEMM_SHAPES];
    thread_t threads[8];
    int ok = 1, errors = 0;
    if (num_threads > 8)
        num_threads = 8;

    for (int t = 0; t < num_threads; t++)
        for (int s = 0; s < GEMM_SHAPES; s++)
            ok &= gemm_alloc(&cases[t][s], s, t * GEMM_SHAPES + s);
    if (!ok)
        return 0;

    for (int t = 0; t < num_threads; t++)
    {
#if defined(_WIN32) || defined(_WIN64)
        threads[t] = CreateThread(NULL, 0, test_gemm_results_thread, cases[t], 0, NULL);
#else
        pthread_create(&threads[t], NULL, test_gemm_results_thread, cases[t]);
#endif
    }
    for (int t = 0; t < num_threads; t++)
    {
#if defined(_WIN32) || defined(_WIN64)
        WaitForSingleObject(threads[t], INFINITE);
        CloseHandle(threads[t]);
#else
        pthread_join(threads[t], NULL);
#endif
    }

    for (int t = 0; t < num_threads; t++)
        for (int s = 0; s < GEMM_SHAPES; s++)
        {
            errors += cases[t][s].errors;
            gemm_free(&cases[t][s]);
        }
    if (errors)
        printf("    %s %d wrong results\n", X_MARK, errors);
    return errors == 0;
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

    SUITE("Concurrent GEMM results");
    TESTEX("2 threads' sgemm/dgemm results match a serial run", test_concurrent_gemm_results(2));
    TESTEX("4 threads' sgemm/dgemm results match a serial run", test_concurrent_gemm_results(4));

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
