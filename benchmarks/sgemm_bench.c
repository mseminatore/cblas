// Controllable SGEMM benchmark + correctness check for CBLAS.
// Usage: sgemm_bench [size] [threads] [lda_pad]
//   size    : square matrix dimension (default 1024)
//   threads : thread count passed to cblas_init (default -1 = auto)
//   lda_pad : leading dimension for padded layout (0 = contiguous, default 0)
// Prints GFLOPS for warm random matrices and a correctness max-abs-diff
// against a naive triple-loop reference (only for size <= 256).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cblas.h"

// WALL-CLOCK time. The library's cbu_timer uses CLOCK_PROCESS_CPUTIME_ID
// (sum of CPU time across all threads), which underreports multi-threaded
// throughput by ~the thread count. Use CLOCK_MONOTONIC like OpenBLAS's bench.
static double now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void fill_rand(float *p, size_t n)
{
    for (size_t i = 0; i < n; i++)
        p[i] = (float)((rand() % 2000) - 1000) / 1000.0f;
}

static void check_correctness(int n)
{
    float *a = malloc((size_t)n*n*sizeof(float));
    float *b = malloc((size_t)n*n*sizeof(float));
    float *c = calloc((size_t)n*n, sizeof(float));
    float *ref = calloc((size_t)n*n, sizeof(float));
    fill_rand(a, (size_t)n*n);
    fill_rand(b, (size_t)n*n);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, n, n, 1.0f, a, n, b, n, 0.0f, c, n);

    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++) {
            float aik = a[i*n+k];
            for (int j = 0; j < n; j++)
                ref[i*n+j] += aik * b[k*n+j];
        }

    double maxd = 0.0, maxrel = 0.0;
    for (size_t i = 0; i < (size_t)n*n; i++) {
        double d = fabs((double)c[i] - ref[i]);
        if (d > maxd) maxd = d;
        double r = d / (fabs((double)ref[i]) + 1e-6);
        if (r > maxrel) maxrel = r;
    }
    // Gate on absolute error: float32 accumulation of n O(1) terms has eps ~ n*1e-7.
    // (Relative error is reported for info but is meaningless on near-zero entries.)
    printf("correctness n=%d: max_abs_diff=%.3e max_rel_diff=%.3e %s\n",
           n, maxd, maxrel, (maxd < 1e-2) ? "PASS" : "FAIL");
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
    float *a = malloc(elems*sizeof(float));
    float *b = malloc(elems*sizeof(float));
    float *c = calloc(elems, sizeof(float));
    if (!a || !b || !c) { fprintf(stderr, "alloc failed\n"); return 1; }
    fill_rand(a, elems);
    fill_rand(b, elems);

    // warmup
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, n, n, 1.0f, a, lda, b, lda, 1.0f, c, lda);

    int iters = (n <= 512) ? 20 : (n <= 2048 ? 5 : 2);
    now_seconds(); // start baseline
    double t0 = now_seconds();
    for (int it = 0; it < iters; it++)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    n, n, n, 1.0f, a, lda, b, lda, 1.0f, c, lda);
    double dt = (now_seconds() - t0) / iters;
    double gflops = 2.0 * n * n * n / 1e9 / dt;

    printf("n=%-5d threads=%-3d lda=%-5d  %7.2f GFLOPS  (%.4fs/iter)\n",
           n, thr, lda, gflops, dt);

    if (n <= 256) check_correctness(n);

    cblas_shutdown();
    return 0;
}
