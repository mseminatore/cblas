// Controllable DGEMM benchmark + correctness check for CBLAS.
// Usage: dgemm_bench [size] [threads] [lda_pad]
//   size    : square matrix dimension (default 1024)
//   threads : thread count passed to cblas_init (default -1 = auto = P-cores)
//   lda_pad : leading dimension for padded layout (0 = contiguous, default 0)
// Wall-clock GFLOPS (CLOCK_MONOTONIC) for warm random matrices, plus a
// max-abs-diff correctness check against a naive triple loop for size <= 256.
// Mirrors sgemm_bench.c so sgemm/dgemm numbers are directly comparable.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "cblas.h"

// Wall-clock seconds; QueryPerformanceCounter on Windows, CLOCK_MONOTONIC
// elsewhere. GFLOPS must be measured against elapsed wall time (not summed
// CPU time) or multi-threaded throughput is underreported by ~the thread count.
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

static void fill_rand(double *p, size_t n)
{
    for (size_t i = 0; i < n; i++)
        p[i] = (double)((rand() % 2000) - 1000) / 1000.0;
}

static void check_correctness(int n)
{
    double *a = malloc((size_t)n*n*sizeof(double));
    double *b = malloc((size_t)n*n*sizeof(double));
    double *c = calloc((size_t)n*n, sizeof(double));
    double *ref = calloc((size_t)n*n, sizeof(double));
    fill_rand(a, (size_t)n*n);
    fill_rand(b, (size_t)n*n);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, n, n, 1.0, a, n, b, n, 0.0, c, n);

    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++) {
            double aik = a[i*n+k];
            for (int j = 0; j < n; j++)
                ref[i*n+j] += aik * b[k*n+j];
        }

    double maxd = 0.0, maxrel = 0.0;
    for (size_t i = 0; i < (size_t)n*n; i++) {
        double d = fabs(c[i] - ref[i]);
        if (d > maxd) maxd = d;
        double r = d / (fabs(ref[i]) + 1e-12);
        if (r > maxrel) maxrel = r;
    }
    printf("correctness n=%d: max_abs_diff=%.3e max_rel_diff=%.3e %s\n",
           n, maxd, maxrel, (maxd < 1e-9) ? "PASS" : "FAIL");
    free(a); free(b); free(c); free(ref);
}

int main(int argc, char **argv)
{
    int n   = (argc > 1) ? atoi(argv[1]) : 1024;
    int thr = (argc > 2) ? atoi(argv[2]) : CBLAS_DEFAULT_THREADS;
    int lda = (argc > 3) ? atoi(argv[3]) : 0;   // 0 => contiguous (lda=n)
    if (lda == 0) lda = n;

    cblas_init(thr);

    size_t elems = (size_t)lda * lda;
    double *a = malloc(elems*sizeof(double));
    double *b = malloc(elems*sizeof(double));
    double *c = calloc(elems, sizeof(double));
    if (!a || !b || !c) { fprintf(stderr, "alloc failed\n"); return 1; }
    fill_rand(a, elems);
    fill_rand(b, elems);

    // warmup
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, n, n, 1.0, a, lda, b, lda, 1.0, c, lda);

    int iters = (n <= 512) ? 20 : (n <= 2048 ? 5 : 2);
    double t0 = now_seconds();
    for (int it = 0; it < iters; it++)
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    n, n, n, 1.0, a, lda, b, lda, 1.0, c, lda);
    double dt = (now_seconds() - t0) / iters;
    double gflops = 2.0 * n * n * n / 1e9 / dt;

    printf("n=%-5d threads=%-3d lda=%-5d  %7.2f GFLOPS  (%.4fs/iter)\n",
           n, thr, lda, gflops, dt);

    if (n <= 256) check_correctness(n);

    cblas_shutdown();
    return 0;
}
