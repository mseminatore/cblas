// Standalone SGEMM benchmark that links against OpenBLAS.
// Same matrix sizes and iteration counts as gemm_perf.c for direct comparison.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cblas.h>

#define MAX_SIZE 8192

static double now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void test_gemm(void)
{
    float *a = (float*)malloc((size_t)MAX_SIZE * MAX_SIZE * sizeof(float));
    float *b = (float*)malloc((size_t)MAX_SIZE * MAX_SIZE * sizeof(float));
    float *c = (float*)malloc((size_t)MAX_SIZE * MAX_SIZE * sizeof(float));

    if (!a || !b || !c) {
        fprintf(stderr, "allocation failed\n");
        free(a); free(b); free(c);
        return;
    }

    memset(a, 0, (size_t)MAX_SIZE * MAX_SIZE * sizeof(float));
    memset(b, 0, (size_t)MAX_SIZE * MAX_SIZE * sizeof(float));
    memset(c, 0, (size_t)MAX_SIZE * MAX_SIZE * sizeof(float));

    printf("Testing performance of OpenBLAS cblas_sgemm()\n\n");

    for (int i = 4; i <= MAX_SIZE; i <<= 1) {
        int m = i, n = i, k = i;

        printf("Testing size %d...", i);
        fflush(stdout);

        int iters = 1;
        if (i <= 512) iters = 10;
        if (i <= 128) iters = 100;
        if (i <= 32)  iters = 1000;

        double t1 = now_seconds();
        for (int iter = 0; iter < iters; iter++) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        m, n, k, 1.0f, a, MAX_SIZE, b, MAX_SIZE, 1.0f, c, MAX_SIZE);
        }
        double dt = (now_seconds() - t1) / iters;

        printf(" %5.2f GFlops in %5.3fs\n",
               (double)2 * m * n * k / 1e9 / dt, dt);
    }

    printf("\nTesting with contiguous layout (lda=n):\n\n");
    for (int i = 128; i <= MAX_SIZE; i <<= 1) {
        int m = i, n = i, k = i;

        printf("Testing size %d...", i);
        fflush(stdout);

        float *ac = (float*)malloc((size_t)i * i * sizeof(float));
        float *bc = (float*)malloc((size_t)i * i * sizeof(float));
        float *cc = (float*)malloc((size_t)i * i * sizeof(float));
        if (!ac || !bc || !cc) { printf("  alloc failed\n"); free(ac); free(bc); free(cc); continue; }
        memset(ac, 0, (size_t)i * i * sizeof(float));
        memset(bc, 0, (size_t)i * i * sizeof(float));
        memset(cc, 0, (size_t)i * i * sizeof(float));

        int iters = (i <= 256) ? 50 : 10;

        double t1 = now_seconds();
        for (int iter = 0; iter < iters; iter++) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        m, n, k, 1.0f, ac, i, bc, i, 1.0f, cc, i);
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
    printf("OpenBLAS SGEMM benchmark\n\n");
    test_gemm();
    return 0;
}
