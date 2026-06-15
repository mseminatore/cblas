// Standalone DGEMM benchmark that links against OpenBLAS.
// Mirrors openblas_gemm_perf.c (sgemm) but double precision, for a direct
// comparison against CBLAS dgemm. Run with OPENBLAS_NUM_THREADS=4 to match
// the 4-P-core target on Apple Silicon.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cblas.h>

#define MAX_SIZE 4096

static double now_seconds(void)
{
#ifdef _WIN32
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
}

static void test_gemm(void)
{
    double *a = (double*)malloc((size_t)MAX_SIZE * MAX_SIZE * sizeof(double));
    double *b = (double*)malloc((size_t)MAX_SIZE * MAX_SIZE * sizeof(double));
    double *c = (double*)malloc((size_t)MAX_SIZE * MAX_SIZE * sizeof(double));

    if (!a || !b || !c) {
        fprintf(stderr, "allocation failed\n");
        free(a); free(b); free(c);
        return;
    }

    memset(a, 0, (size_t)MAX_SIZE * MAX_SIZE * sizeof(double));
    memset(b, 0, (size_t)MAX_SIZE * MAX_SIZE * sizeof(double));
    memset(c, 0, (size_t)MAX_SIZE * MAX_SIZE * sizeof(double));

    printf("Testing with contiguous layout (lda=n):\n\n");
    for (int i = 128; i <= MAX_SIZE; i <<= 1) {
        int m = i, n = i, k = i;

        printf("Testing size %d...", i);
        fflush(stdout);

        double *ac = (double*)malloc((size_t)i * i * sizeof(double));
        double *bc = (double*)malloc((size_t)i * i * sizeof(double));
        double *cc = (double*)malloc((size_t)i * i * sizeof(double));
        if (!ac || !bc || !cc) { printf("  alloc failed\n"); free(ac); free(bc); free(cc); continue; }
        memset(ac, 0, (size_t)i * i * sizeof(double));
        memset(bc, 0, (size_t)i * i * sizeof(double));
        memset(cc, 0, (size_t)i * i * sizeof(double));

        int iters = (i <= 256) ? 50 : (i <= 2048 ? 10 : 4);

        double t1 = now_seconds();
        for (int iter = 0; iter < iters; iter++) {
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        m, n, k, 1.0, ac, i, bc, i, 1.0, cc, i);
        }
        double dt = (now_seconds() - t1) / iters;

        printf(" %5.2f GFlops in %5.3fs\n",
               (double)2 * m * n * k / 1e9 / dt, dt);

        free(ac); free(bc); free(cc);
    }

    free(a); free(b); free(c);
}

int main(void)
{
    printf("OpenBLAS DGEMM benchmark\n\n");
    test_gemm();
    return 0;
}
