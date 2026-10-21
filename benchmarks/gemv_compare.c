//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------
//
// Side-by-side SGEMV benchmark. Build once against CBLAS and once against
// OpenBLAS (-DUSE_OPENBLAS) and compare the output.
//
// usage: gemv_compare [threads]
//
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef USE_OPENBLAS
#   include <cblas.h>
#   define LIB_NAME "OpenBLAS"
#else
#   include "cblas.h"
#   define LIB_NAME "CBLAS"
#endif

#ifdef _WIN32
#   include <windows.h>
static double now_sec(void)
{
    LARGE_INTEGER f, t;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)f.QuadPart;
}
#else
#   include <time.h>
static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
#endif

typedef struct { int m, n; } shape_t;

// Square sizes plus shapes typical of small dense NN layers (ann)
static const shape_t shapes[] = {
    {   10,  128 }, {  128,  784 }, {  784,  128 }, {  64,   64 },
    {  256,  256 }, {  512,  512 }, { 1024, 1024 }, { 2048, 2048 }, { 4096, 4096 },
};

//------------------------------------------------------
// run one shape/transpose combination, return best time
//------------------------------------------------------
static double bench(int trans, int m, int n, float *a, float *x, float *y, double *checksum)
{
    // keep total work per measurement roughly constant (~200 MFLOP)
    long long work = 2LL * m * n;
    int reps = (int)(2e8 / work);
    if (reps < 1) reps = 1;

    int ylen = trans ? n : m;
    int xlen = trans ? m : n;
    (void)xlen;

    double best = 1e30;
    for (int trial = 0; trial < 7; trial++) {
        double t0 = now_sec();
        for (int r = 0; r < reps; r++) {
            cblas_sgemv(CblasRowMajor, trans ? CblasTrans : CblasNoTrans, m, n,
                        1.0f, a, n, x, 1, 0.5f, y, 1);
        }
        double dt = (now_sec() - t0) / reps;
        if (dt < best) best = dt;
    }

    double s = 0;
    for (int i = 0; i < ylen; i++) s += y[i];
    *checksum = s;
    return best;
}

int main(int argc, char *argv[])
{
    int threads = (argc > 1) ? atoi(argv[1]) : 1;

#ifdef USE_OPENBLAS
    openblas_set_num_threads(threads);
#else
    cblas_init(threads);
#endif

    const int maxdim = 4096;
    float *a = (float *)malloc(sizeof(float) * maxdim * maxdim);
    float *x = (float *)malloc(sizeof(float) * maxdim);
    float *y = (float *)malloc(sizeof(float) * maxdim);

    for (int i = 0; i < maxdim * maxdim; i++) a[i] = (float)((i * 7) % 13) / 13.0f - 0.5f;
    for (int i = 0; i < maxdim; i++) x[i] = (float)((i * 3) % 11) / 11.0f - 0.5f;

    printf("%s sgemv, threads=%d\n", LIB_NAME, threads);
    printf("%6s %6s %6s %10s %10s %10s\n", "trans", "m", "n", "us", "GFlops", "GB/s");

    for (int t = 0; t < 2; t++) {
        for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++) {
            int m = shapes[s].m, n = shapes[s].n;
            for (int i = 0; i < maxdim; i++) y[i] = 0.0f;
            double cs;
            double dt = bench(t, m, n, a, x, y, &cs);
            double gflops = 2.0 * m * n / dt / 1e9;
            double gbs = (double)m * n * sizeof(float) / dt / 1e9;
            printf("%6s %6d %6d %10.2f %10.2f %10.2f\n", t ? "T" : "N", m, n, dt * 1e6, gflops, gbs);
        }
    }

#ifndef USE_OPENBLAS
    cblas_shutdown();
#endif
    free(a); free(x); free(y);
    return 0;
}
