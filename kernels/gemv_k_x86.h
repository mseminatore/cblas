//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------
//
// Shared x86 GEMV kernel template. Included by gemv_k_avx.c and
// gemv_k_fma.c, which differ only in how a multiply-add is issued.
//
// Before including, define:
//   GEMV_SUFFIX            kernel name suffix (avx, fma)
//   GEMV_MADD_PS(a, b, c)  c + a*b for __m256
//   GEMV_MADD_PD(a, b, c)  c + a*b for __m256d
//
// Kernels operate on a row-major m x n matrix A (leading dimension lda):
//   [sd]gemv_k_*    y[0:m] = alpha * A   * x + beta * y   (dot form)
//   [sd]gemv_t_k_*  y[0:n] = alpha * A^T * x + beta * y   (axpy form)
//
// Both are memory bound, so the goal is to touch each element of A exactly
// once with as few extra loads/stores as possible:
//  - dot form processes 8 (then 4) rows per pass so each load of x feeds
//    several rows
//  - axpy form processes 8 (then 4) rows per pass so each y load/store is
//    shared by several rows, and blocks y so the active slice stays in L1
//------------------------------------------------------

#define GEMV_CAT2(a, b) a##_##b
#define GEMV_CAT(a, b)  GEMV_CAT2(a, b)
#define GEMV_FN(name)   GEMV_CAT(name, GEMV_SUFFIX)

// number of y elements per axpy-form block (fits comfortably in L1)
#define GEMV_T_BLOCK_S  2048
#define GEMV_T_BLOCK_D  1024

//------------------------------------------------------
// reduce 4 vectors to a single __m128 of their horizontal sums
//------------------------------------------------------
static inline __m128 GEMV_FN(hsum4_ps)(__m256 s0, __m256 s1, __m256 s2, __m256 s3)
{
    __m256 s01 = _mm256_hadd_ps(s0, s1);
    __m256 s23 = _mm256_hadd_ps(s2, s3);
    __m256 s = _mm256_hadd_ps(s01, s23);
    return _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
}

static inline float GEMV_FN(hsum_ps)(__m256 v)
{
    __m128 r = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    r = _mm_hadd_ps(r, r);
    r = _mm_hadd_ps(r, r);
    return _mm_cvtss_f32(r);
}

static inline __m256d GEMV_FN(hsum4_pd)(__m256d s0, __m256d s1, __m256d s2, __m256d s3)
{
    __m256d s01 = _mm256_hadd_pd(s0, s1);   // [s0a s1a s0b s1b]
    __m256d s23 = _mm256_hadd_pd(s2, s3);   // [s2a s3a s2b s3b]
    __m256d lo = _mm256_permute2f128_pd(s01, s23, 0x20);
    __m256d hi = _mm256_permute2f128_pd(s01, s23, 0x31);
    return _mm256_add_pd(lo, hi);
}

static inline double GEMV_FN(hsum_pd)(__m256d v)
{
    __m128d r = _mm_add_pd(_mm256_castpd256_pd128(v), _mm256_extractf128_pd(v, 1));
    r = _mm_hadd_pd(r, r);
    return _mm_cvtsd_f64(r);
}

//------------------------------------------------------
// y = beta * y, with beta == 0 clearing y (BLAS semantics: y need not be set)
//------------------------------------------------------
static void GEMV_FN(sgemv_scale_y)(float* y, CBLAS_INDEX n, CBLAS_INDEX incy, float beta)
{
    if (beta == 1.0f)
        return;

    CBLAS_INDEX j = 0;
    if (incy == 1)
    {
        // beta == 0 must not read y (it may hold NaN/Inf)
        __m256 vb = _mm256_set1_ps(beta);
        for (; j + 8 <= n; j += 8)
            _mm256_storeu_ps(y + j, beta == 0.0f ? _mm256_setzero_ps() : _mm256_mul_ps(vb, _mm256_loadu_ps(y + j)));
    }

    for (; j < n; j++)
        y[j * incy] = (beta == 0.0f) ? 0.0f : beta * y[j * incy];
}

static void GEMV_FN(dgemv_scale_y)(double* y, CBLAS_INDEX n, CBLAS_INDEX incy, double beta)
{
    if (beta == 1.0)
        return;

    CBLAS_INDEX j = 0;
    if (incy == 1)
    {
        // beta == 0 must not read y (it may hold NaN/Inf)
        __m256d vb = _mm256_set1_pd(beta);
        for (; j + 4 <= n; j += 4)
            _mm256_storeu_pd(y + j, beta == 0.0 ? _mm256_setzero_pd() : _mm256_mul_pd(vb, _mm256_loadu_pd(y + j)));
    }

    for (; j < n; j++)
        y[j * incy] = (beta == 0.0) ? 0.0 : beta * y[j * incy];
}

//------------------------------------------------------
// Single-precision GEMV kernel, dot form: y = alpha*A*x + beta*y
//------------------------------------------------------
void GEMV_FN(sgemv_k)(cblas_args_t* args)
{
    float* a = (float*)args->a;
    float* x = (float*)args->x;
    float* y = (float*)args->y;
    CBLAS_INDEX m = args->m;
    CBLAS_INDEX n = args->n;
    CBLAS_INDEX lda = args->lda;
    CBLAS_INDEX incx = args->incx;
    CBLAS_INDEX incy = args->incy;
    float alpha = *(float*)args->alpha;
    float beta = *(float*)args->beta;

    if (incx != 1)
    {
        // General strided x - scalar
        for (CBLAS_INDEX row = 0; row < m; row++)
        {
            float sum = 0.0f;
            for (CBLAS_INDEX col = 0; col < n; col++)
                sum += a[row * lda + col] * x[col * incx];

            float* yp = &y[row * incy];
            *yp = (beta == 0.0f ? 0.0f : beta * *yp) + alpha * sum;
        }
        return;
    }

    CBLAS_INDEX n16 = n & ~(CBLAS_INDEX)15;
    CBLAS_INDEX n8 = n & ~(CBLAS_INDEX)7;
    CBLAS_INDEX row = 0;

    // 8 rows per pass: each x vector load is shared by 8 rows, which keeps
    // the load ports busy with A rather than x
    for (; row + 8 <= m; row += 8)
    {
        const float* ar[8];
        for (int r = 0; r < 8; r++)
            ar[r] = a + (row + r) * lda;

        __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
        __m256 s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
        __m256 s4 = _mm256_setzero_ps(), s5 = _mm256_setzero_ps();
        __m256 s6 = _mm256_setzero_ps(), s7 = _mm256_setzero_ps();

        CBLAS_INDEX col = 0;
        for (; col < n8; col += 8)
        {
            __m256 x0 = _mm256_loadu_ps(x + col);

            s0 = GEMV_MADD_PS(_mm256_loadu_ps(ar[0] + col), x0, s0);
            s1 = GEMV_MADD_PS(_mm256_loadu_ps(ar[1] + col), x0, s1);
            s2 = GEMV_MADD_PS(_mm256_loadu_ps(ar[2] + col), x0, s2);
            s3 = GEMV_MADD_PS(_mm256_loadu_ps(ar[3] + col), x0, s3);
            s4 = GEMV_MADD_PS(_mm256_loadu_ps(ar[4] + col), x0, s4);
            s5 = GEMV_MADD_PS(_mm256_loadu_ps(ar[5] + col), x0, s5);
            s6 = GEMV_MADD_PS(_mm256_loadu_ps(ar[6] + col), x0, s6);
            s7 = GEMV_MADD_PS(_mm256_loadu_ps(ar[7] + col), x0, s7);
        }

        float sum[8];
        _mm_storeu_ps(sum, GEMV_FN(hsum4_ps)(s0, s1, s2, s3));
        _mm_storeu_ps(sum + 4, GEMV_FN(hsum4_ps)(s4, s5, s6, s7));

        for (; col < n; col++)
        {
            float xv = x[col];
            for (int r = 0; r < 8; r++)
                sum[r] += ar[r][col] * xv;
        }

        for (int r = 0; r < 8; r++)
        {
            float* yp = &y[(row + r) * incy];
            *yp = (beta == 0.0f ? 0.0f : beta * *yp) + alpha * sum[r];
        }
    }

    // 4 rows per pass: each x vector load is shared by 4 rows
    for (; row + 4 <= m; row += 4)
    {
        const float* a0 = a + row * lda;
        const float* a1 = a0 + lda;
        const float* a2 = a1 + lda;
        const float* a3 = a2 + lda;

        __m256 s00 = _mm256_setzero_ps(), s01 = _mm256_setzero_ps();
        __m256 s10 = _mm256_setzero_ps(), s11 = _mm256_setzero_ps();
        __m256 s20 = _mm256_setzero_ps(), s21 = _mm256_setzero_ps();
        __m256 s30 = _mm256_setzero_ps(), s31 = _mm256_setzero_ps();

        CBLAS_INDEX col = 0;
        for (; col < n16; col += 16)
        {
            __m256 x0 = _mm256_loadu_ps(x + col);
            __m256 x1 = _mm256_loadu_ps(x + col + 8);

            s00 = GEMV_MADD_PS(_mm256_loadu_ps(a0 + col), x0, s00);
            s01 = GEMV_MADD_PS(_mm256_loadu_ps(a0 + col + 8), x1, s01);
            s10 = GEMV_MADD_PS(_mm256_loadu_ps(a1 + col), x0, s10);
            s11 = GEMV_MADD_PS(_mm256_loadu_ps(a1 + col + 8), x1, s11);
            s20 = GEMV_MADD_PS(_mm256_loadu_ps(a2 + col), x0, s20);
            s21 = GEMV_MADD_PS(_mm256_loadu_ps(a2 + col + 8), x1, s21);
            s30 = GEMV_MADD_PS(_mm256_loadu_ps(a3 + col), x0, s30);
            s31 = GEMV_MADD_PS(_mm256_loadu_ps(a3 + col + 8), x1, s31);
        }

        for (; col < n8; col += 8)
        {
            __m256 x0 = _mm256_loadu_ps(x + col);

            s00 = GEMV_MADD_PS(_mm256_loadu_ps(a0 + col), x0, s00);
            s10 = GEMV_MADD_PS(_mm256_loadu_ps(a1 + col), x0, s10);
            s20 = GEMV_MADD_PS(_mm256_loadu_ps(a2 + col), x0, s20);
            s30 = GEMV_MADD_PS(_mm256_loadu_ps(a3 + col), x0, s30);
        }

        float sum[4];
        _mm_storeu_ps(sum, GEMV_FN(hsum4_ps)(_mm256_add_ps(s00, s01), _mm256_add_ps(s10, s11),
                                             _mm256_add_ps(s20, s21), _mm256_add_ps(s30, s31)));

        for (; col < n; col++)
        {
            float xv = x[col];
            sum[0] += a0[col] * xv;
            sum[1] += a1[col] * xv;
            sum[2] += a2[col] * xv;
            sum[3] += a3[col] * xv;
        }

        for (int r = 0; r < 4; r++)
        {
            float* yp = &y[(row + r) * incy];
            *yp = (beta == 0.0f ? 0.0f : beta * *yp) + alpha * sum[r];
        }
    }

    // remaining rows, one at a time
    for (; row < m; row++)
    {
        const float* a0 = a + row * lda;
        __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();

        CBLAS_INDEX col = 0;
        for (; col < n16; col += 16)
        {
            s0 = GEMV_MADD_PS(_mm256_loadu_ps(a0 + col), _mm256_loadu_ps(x + col), s0);
            s1 = GEMV_MADD_PS(_mm256_loadu_ps(a0 + col + 8), _mm256_loadu_ps(x + col + 8), s1);
        }
        for (; col < n8; col += 8)
            s0 = GEMV_MADD_PS(_mm256_loadu_ps(a0 + col), _mm256_loadu_ps(x + col), s0);

        float sum = GEMV_FN(hsum_ps)(_mm256_add_ps(s0, s1));
        for (; col < n; col++)
            sum += a0[col] * x[col];

        float* yp = &y[row * incy];
        *yp = (beta == 0.0f ? 0.0f : beta * *yp) + alpha * sum;
    }
}

//------------------------------------------------------
// Single-precision GEMV kernel, axpy form: y = alpha*A^T*x + beta*y
//------------------------------------------------------
void GEMV_FN(sgemv_t_k)(cblas_args_t* args)
{
    float* a = (float*)args->a;
    float* x = (float*)args->x;
    float* y = (float*)args->y;
    CBLAS_INDEX m = args->m;
    CBLAS_INDEX n = args->n;
    CBLAS_INDEX lda = args->lda;
    CBLAS_INDEX incx = args->incx;
    CBLAS_INDEX incy = args->incy;
    float alpha = *(float*)args->alpha;
    float beta = *(float*)args->beta;

    GEMV_FN(sgemv_scale_y)(y, n, incy, beta);

    if (alpha == 0.0f)
        return;

    if (incy != 1)
    {
        // General strided y - scalar, still walking A row by row
        for (CBLAS_INDEX row = 0; row < m; row++)
        {
            float t = alpha * x[row * incx];
            const float* ar = a + row * lda;
            for (CBLAS_INDEX col = 0; col < n; col++)
                y[col * incy] += t * ar[col];
        }
        return;
    }

    for (CBLAS_INDEX jb = 0; jb < n; jb += GEMV_T_BLOCK_S)
    {
        CBLAS_INDEX nb = MIN(GEMV_T_BLOCK_S, n - jb);
        CBLAS_INDEX nb16 = nb & ~(CBLAS_INDEX)15;
        CBLAS_INDEX nb8 = nb & ~(CBLAS_INDEX)7;
        float* yb = y + jb;
        CBLAS_INDEX row = 0;

        // 8 rows per pass: each y load/store is shared by 8 rows. Rows 4-7
        // accumulate separately to halve the dependency chain on y.
        for (; row + 8 <= m; row += 8)
        {
            const float* ar[8];
            float ts[8];
            __m256 t[8];
            for (int r = 0; r < 8; r++)
            {
                ar[r] = a + (row + r) * lda + jb;
                ts[r] = alpha * x[(row + r) * incx];
                t[r] = _mm256_set1_ps(ts[r]);
            }

            CBLAS_INDEX col = 0;
            for (; col < nb8; col += 8)
            {
                __m256 y0 = _mm256_loadu_ps(yb + col);
                __m256 p = _mm256_mul_ps(_mm256_loadu_ps(ar[4] + col), t[4]);

                y0 = GEMV_MADD_PS(_mm256_loadu_ps(ar[0] + col), t[0], y0);
                p = GEMV_MADD_PS(_mm256_loadu_ps(ar[5] + col), t[5], p);
                y0 = GEMV_MADD_PS(_mm256_loadu_ps(ar[1] + col), t[1], y0);
                p = GEMV_MADD_PS(_mm256_loadu_ps(ar[6] + col), t[6], p);
                y0 = GEMV_MADD_PS(_mm256_loadu_ps(ar[2] + col), t[2], y0);
                p = GEMV_MADD_PS(_mm256_loadu_ps(ar[7] + col), t[7], p);
                y0 = GEMV_MADD_PS(_mm256_loadu_ps(ar[3] + col), t[3], y0);

                _mm256_storeu_ps(yb + col, _mm256_add_ps(y0, p));
            }

            for (; col < nb; col++)
            {
                float acc = yb[col];
                for (int r = 0; r < 8; r++)
                    acc += ts[r] * ar[r][col];
                yb[col] = acc;
            }
        }

        // 4 rows per pass: each y load/store is shared by 4 rows
        for (; row + 4 <= m; row += 4)
        {
            const float* a0 = a + row * lda + jb;
            const float* a1 = a0 + lda;
            const float* a2 = a1 + lda;
            const float* a3 = a2 + lda;

            float t0s = alpha * x[(row + 0) * incx];
            float t1s = alpha * x[(row + 1) * incx];
            float t2s = alpha * x[(row + 2) * incx];
            float t3s = alpha * x[(row + 3) * incx];
            __m256 t0 = _mm256_set1_ps(t0s);
            __m256 t1 = _mm256_set1_ps(t1s);
            __m256 t2 = _mm256_set1_ps(t2s);
            __m256 t3 = _mm256_set1_ps(t3s);

            CBLAS_INDEX col = 0;
            for (; col < nb16; col += 16)
            {
                __m256 y0 = _mm256_loadu_ps(yb + col);
                __m256 y1 = _mm256_loadu_ps(yb + col + 8);

                y0 = GEMV_MADD_PS(_mm256_loadu_ps(a0 + col), t0, y0);
                y1 = GEMV_MADD_PS(_mm256_loadu_ps(a0 + col + 8), t0, y1);
                y0 = GEMV_MADD_PS(_mm256_loadu_ps(a1 + col), t1, y0);
                y1 = GEMV_MADD_PS(_mm256_loadu_ps(a1 + col + 8), t1, y1);
                y0 = GEMV_MADD_PS(_mm256_loadu_ps(a2 + col), t2, y0);
                y1 = GEMV_MADD_PS(_mm256_loadu_ps(a2 + col + 8), t2, y1);
                y0 = GEMV_MADD_PS(_mm256_loadu_ps(a3 + col), t3, y0);
                y1 = GEMV_MADD_PS(_mm256_loadu_ps(a3 + col + 8), t3, y1);

                _mm256_storeu_ps(yb + col, y0);
                _mm256_storeu_ps(yb + col + 8, y1);
            }

            for (; col < nb8; col += 8)
            {
                __m256 y0 = _mm256_loadu_ps(yb + col);
                y0 = GEMV_MADD_PS(_mm256_loadu_ps(a0 + col), t0, y0);
                y0 = GEMV_MADD_PS(_mm256_loadu_ps(a1 + col), t1, y0);
                y0 = GEMV_MADD_PS(_mm256_loadu_ps(a2 + col), t2, y0);
                y0 = GEMV_MADD_PS(_mm256_loadu_ps(a3 + col), t3, y0);
                _mm256_storeu_ps(yb + col, y0);
            }

            for (; col < nb; col++)
                yb[col] += t0s * a0[col] + t1s * a1[col] + t2s * a2[col] + t3s * a3[col];
        }

        // remaining rows, one at a time
        for (; row < m; row++)
        {
            const float* a0 = a + row * lda + jb;
            float t0s = alpha * x[row * incx];
            __m256 t0 = _mm256_set1_ps(t0s);

            CBLAS_INDEX col = 0;
            for (; col < nb8; col += 8)
                _mm256_storeu_ps(yb + col, GEMV_MADD_PS(_mm256_loadu_ps(a0 + col), t0, _mm256_loadu_ps(yb + col)));

            for (; col < nb; col++)
                yb[col] += t0s * a0[col];
        }
    }
}

//------------------------------------------------------
// Double-precision GEMV kernel, dot form: y = alpha*A*x + beta*y
//------------------------------------------------------
void GEMV_FN(dgemv_k)(cblas_args_t* args)
{
    double* a = (double*)args->a;
    double* x = (double*)args->x;
    double* y = (double*)args->y;
    CBLAS_INDEX m = args->m;
    CBLAS_INDEX n = args->n;
    CBLAS_INDEX lda = args->lda;
    CBLAS_INDEX incx = args->incx;
    CBLAS_INDEX incy = args->incy;
    double alpha = *(double*)args->alpha;
    double beta = *(double*)args->beta;

    if (incx != 1)
    {
        // General strided x - scalar
        for (CBLAS_INDEX row = 0; row < m; row++)
        {
            double sum = 0.0;
            for (CBLAS_INDEX col = 0; col < n; col++)
                sum += a[row * lda + col] * x[col * incx];

            double* yp = &y[row * incy];
            *yp = (beta == 0.0 ? 0.0 : beta * *yp) + alpha * sum;
        }
        return;
    }

    CBLAS_INDEX n8 = n & ~(CBLAS_INDEX)7;
    CBLAS_INDEX n4 = n & ~(CBLAS_INDEX)3;
    CBLAS_INDEX row = 0;

    // 8 rows per pass: each x vector load is shared by 8 rows
    for (; row + 8 <= m; row += 8)
    {
        const double* ar[8];
        for (int r = 0; r < 8; r++)
            ar[r] = a + (row + r) * lda;

        __m256d s0 = _mm256_setzero_pd(), s1 = _mm256_setzero_pd();
        __m256d s2 = _mm256_setzero_pd(), s3 = _mm256_setzero_pd();
        __m256d s4 = _mm256_setzero_pd(), s5 = _mm256_setzero_pd();
        __m256d s6 = _mm256_setzero_pd(), s7 = _mm256_setzero_pd();

        CBLAS_INDEX col = 0;
        for (; col < n4; col += 4)
        {
            __m256d x0 = _mm256_loadu_pd(x + col);

            s0 = GEMV_MADD_PD(_mm256_loadu_pd(ar[0] + col), x0, s0);
            s1 = GEMV_MADD_PD(_mm256_loadu_pd(ar[1] + col), x0, s1);
            s2 = GEMV_MADD_PD(_mm256_loadu_pd(ar[2] + col), x0, s2);
            s3 = GEMV_MADD_PD(_mm256_loadu_pd(ar[3] + col), x0, s3);
            s4 = GEMV_MADD_PD(_mm256_loadu_pd(ar[4] + col), x0, s4);
            s5 = GEMV_MADD_PD(_mm256_loadu_pd(ar[5] + col), x0, s5);
            s6 = GEMV_MADD_PD(_mm256_loadu_pd(ar[6] + col), x0, s6);
            s7 = GEMV_MADD_PD(_mm256_loadu_pd(ar[7] + col), x0, s7);
        }

        double sum[8];
        _mm256_storeu_pd(sum, GEMV_FN(hsum4_pd)(s0, s1, s2, s3));
        _mm256_storeu_pd(sum + 4, GEMV_FN(hsum4_pd)(s4, s5, s6, s7));

        for (; col < n; col++)
        {
            double xv = x[col];
            for (int r = 0; r < 8; r++)
                sum[r] += ar[r][col] * xv;
        }

        for (int r = 0; r < 8; r++)
        {
            double* yp = &y[(row + r) * incy];
            *yp = (beta == 0.0 ? 0.0 : beta * *yp) + alpha * sum[r];
        }
    }

    // 4 rows per pass: each x vector load is shared by 4 rows
    for (; row + 4 <= m; row += 4)
    {
        const double* a0 = a + row * lda;
        const double* a1 = a0 + lda;
        const double* a2 = a1 + lda;
        const double* a3 = a2 + lda;

        __m256d s00 = _mm256_setzero_pd(), s01 = _mm256_setzero_pd();
        __m256d s10 = _mm256_setzero_pd(), s11 = _mm256_setzero_pd();
        __m256d s20 = _mm256_setzero_pd(), s21 = _mm256_setzero_pd();
        __m256d s30 = _mm256_setzero_pd(), s31 = _mm256_setzero_pd();

        CBLAS_INDEX col = 0;
        for (; col < n8; col += 8)
        {
            __m256d x0 = _mm256_loadu_pd(x + col);
            __m256d x1 = _mm256_loadu_pd(x + col + 4);

            s00 = GEMV_MADD_PD(_mm256_loadu_pd(a0 + col), x0, s00);
            s01 = GEMV_MADD_PD(_mm256_loadu_pd(a0 + col + 4), x1, s01);
            s10 = GEMV_MADD_PD(_mm256_loadu_pd(a1 + col), x0, s10);
            s11 = GEMV_MADD_PD(_mm256_loadu_pd(a1 + col + 4), x1, s11);
            s20 = GEMV_MADD_PD(_mm256_loadu_pd(a2 + col), x0, s20);
            s21 = GEMV_MADD_PD(_mm256_loadu_pd(a2 + col + 4), x1, s21);
            s30 = GEMV_MADD_PD(_mm256_loadu_pd(a3 + col), x0, s30);
            s31 = GEMV_MADD_PD(_mm256_loadu_pd(a3 + col + 4), x1, s31);
        }

        for (; col < n4; col += 4)
        {
            __m256d x0 = _mm256_loadu_pd(x + col);

            s00 = GEMV_MADD_PD(_mm256_loadu_pd(a0 + col), x0, s00);
            s10 = GEMV_MADD_PD(_mm256_loadu_pd(a1 + col), x0, s10);
            s20 = GEMV_MADD_PD(_mm256_loadu_pd(a2 + col), x0, s20);
            s30 = GEMV_MADD_PD(_mm256_loadu_pd(a3 + col), x0, s30);
        }

        double sum[4];
        _mm256_storeu_pd(sum, GEMV_FN(hsum4_pd)(_mm256_add_pd(s00, s01), _mm256_add_pd(s10, s11),
                                                _mm256_add_pd(s20, s21), _mm256_add_pd(s30, s31)));

        for (; col < n; col++)
        {
            double xv = x[col];
            sum[0] += a0[col] * xv;
            sum[1] += a1[col] * xv;
            sum[2] += a2[col] * xv;
            sum[3] += a3[col] * xv;
        }

        for (int r = 0; r < 4; r++)
        {
            double* yp = &y[(row + r) * incy];
            *yp = (beta == 0.0 ? 0.0 : beta * *yp) + alpha * sum[r];
        }
    }

    // remaining rows, one at a time
    for (; row < m; row++)
    {
        const double* a0 = a + row * lda;
        __m256d s0 = _mm256_setzero_pd(), s1 = _mm256_setzero_pd();

        CBLAS_INDEX col = 0;
        for (; col < n8; col += 8)
        {
            s0 = GEMV_MADD_PD(_mm256_loadu_pd(a0 + col), _mm256_loadu_pd(x + col), s0);
            s1 = GEMV_MADD_PD(_mm256_loadu_pd(a0 + col + 4), _mm256_loadu_pd(x + col + 4), s1);
        }
        for (; col < n4; col += 4)
            s0 = GEMV_MADD_PD(_mm256_loadu_pd(a0 + col), _mm256_loadu_pd(x + col), s0);

        double sum = GEMV_FN(hsum_pd)(_mm256_add_pd(s0, s1));
        for (; col < n; col++)
            sum += a0[col] * x[col];

        double* yp = &y[row * incy];
        *yp = (beta == 0.0 ? 0.0 : beta * *yp) + alpha * sum;
    }
}

//------------------------------------------------------
// Double-precision GEMV kernel, axpy form: y = alpha*A^T*x + beta*y
//------------------------------------------------------
void GEMV_FN(dgemv_t_k)(cblas_args_t* args)
{
    double* a = (double*)args->a;
    double* x = (double*)args->x;
    double* y = (double*)args->y;
    CBLAS_INDEX m = args->m;
    CBLAS_INDEX n = args->n;
    CBLAS_INDEX lda = args->lda;
    CBLAS_INDEX incx = args->incx;
    CBLAS_INDEX incy = args->incy;
    double alpha = *(double*)args->alpha;
    double beta = *(double*)args->beta;

    GEMV_FN(dgemv_scale_y)(y, n, incy, beta);

    if (alpha == 0.0)
        return;

    if (incy != 1)
    {
        // General strided y - scalar, still walking A row by row
        for (CBLAS_INDEX row = 0; row < m; row++)
        {
            double t = alpha * x[row * incx];
            const double* ar = a + row * lda;
            for (CBLAS_INDEX col = 0; col < n; col++)
                y[col * incy] += t * ar[col];
        }
        return;
    }

    for (CBLAS_INDEX jb = 0; jb < n; jb += GEMV_T_BLOCK_D)
    {
        CBLAS_INDEX nb = MIN(GEMV_T_BLOCK_D, n - jb);
        CBLAS_INDEX nb8 = nb & ~(CBLAS_INDEX)7;
        CBLAS_INDEX nb4 = nb & ~(CBLAS_INDEX)3;
        double* yb = y + jb;
        CBLAS_INDEX row = 0;

        // 8 rows per pass: each y load/store is shared by 8 rows. Rows 4-7
        // accumulate separately to halve the dependency chain on y.
        for (; row + 8 <= m; row += 8)
        {
            const double* ar[8];
            double ts[8];
            __m256d t[8];
            for (int r = 0; r < 8; r++)
            {
                ar[r] = a + (row + r) * lda + jb;
                ts[r] = alpha * x[(row + r) * incx];
                t[r] = _mm256_set1_pd(ts[r]);
            }

            CBLAS_INDEX col = 0;
            for (; col < nb4; col += 4)
            {
                __m256d y0 = _mm256_loadu_pd(yb + col);
                __m256d p = _mm256_mul_pd(_mm256_loadu_pd(ar[4] + col), t[4]);

                y0 = GEMV_MADD_PD(_mm256_loadu_pd(ar[0] + col), t[0], y0);
                p = GEMV_MADD_PD(_mm256_loadu_pd(ar[5] + col), t[5], p);
                y0 = GEMV_MADD_PD(_mm256_loadu_pd(ar[1] + col), t[1], y0);
                p = GEMV_MADD_PD(_mm256_loadu_pd(ar[6] + col), t[6], p);
                y0 = GEMV_MADD_PD(_mm256_loadu_pd(ar[2] + col), t[2], y0);
                p = GEMV_MADD_PD(_mm256_loadu_pd(ar[7] + col), t[7], p);
                y0 = GEMV_MADD_PD(_mm256_loadu_pd(ar[3] + col), t[3], y0);

                _mm256_storeu_pd(yb + col, _mm256_add_pd(y0, p));
            }

            for (; col < nb; col++)
            {
                double acc = yb[col];
                for (int r = 0; r < 8; r++)
                    acc += ts[r] * ar[r][col];
                yb[col] = acc;
            }
        }

        // 4 rows per pass: each y load/store is shared by 4 rows
        for (; row + 4 <= m; row += 4)
        {
            const double* a0 = a + row * lda + jb;
            const double* a1 = a0 + lda;
            const double* a2 = a1 + lda;
            const double* a3 = a2 + lda;

            double t0s = alpha * x[(row + 0) * incx];
            double t1s = alpha * x[(row + 1) * incx];
            double t2s = alpha * x[(row + 2) * incx];
            double t3s = alpha * x[(row + 3) * incx];
            __m256d t0 = _mm256_set1_pd(t0s);
            __m256d t1 = _mm256_set1_pd(t1s);
            __m256d t2 = _mm256_set1_pd(t2s);
            __m256d t3 = _mm256_set1_pd(t3s);

            CBLAS_INDEX col = 0;
            for (; col < nb8; col += 8)
            {
                __m256d y0 = _mm256_loadu_pd(yb + col);
                __m256d y1 = _mm256_loadu_pd(yb + col + 4);

                y0 = GEMV_MADD_PD(_mm256_loadu_pd(a0 + col), t0, y0);
                y1 = GEMV_MADD_PD(_mm256_loadu_pd(a0 + col + 4), t0, y1);
                y0 = GEMV_MADD_PD(_mm256_loadu_pd(a1 + col), t1, y0);
                y1 = GEMV_MADD_PD(_mm256_loadu_pd(a1 + col + 4), t1, y1);
                y0 = GEMV_MADD_PD(_mm256_loadu_pd(a2 + col), t2, y0);
                y1 = GEMV_MADD_PD(_mm256_loadu_pd(a2 + col + 4), t2, y1);
                y0 = GEMV_MADD_PD(_mm256_loadu_pd(a3 + col), t3, y0);
                y1 = GEMV_MADD_PD(_mm256_loadu_pd(a3 + col + 4), t3, y1);

                _mm256_storeu_pd(yb + col, y0);
                _mm256_storeu_pd(yb + col + 4, y1);
            }

            for (; col < nb4; col += 4)
            {
                __m256d y0 = _mm256_loadu_pd(yb + col);
                y0 = GEMV_MADD_PD(_mm256_loadu_pd(a0 + col), t0, y0);
                y0 = GEMV_MADD_PD(_mm256_loadu_pd(a1 + col), t1, y0);
                y0 = GEMV_MADD_PD(_mm256_loadu_pd(a2 + col), t2, y0);
                y0 = GEMV_MADD_PD(_mm256_loadu_pd(a3 + col), t3, y0);
                _mm256_storeu_pd(yb + col, y0);
            }

            for (; col < nb; col++)
                yb[col] += t0s * a0[col] + t1s * a1[col] + t2s * a2[col] + t3s * a3[col];
        }

        // remaining rows, one at a time
        for (; row < m; row++)
        {
            const double* a0 = a + row * lda + jb;
            double t0s = alpha * x[row * incx];
            __m256d t0 = _mm256_set1_pd(t0s);

            CBLAS_INDEX col = 0;
            for (; col < nb4; col += 4)
                _mm256_storeu_pd(yb + col, GEMV_MADD_PD(_mm256_loadu_pd(a0 + col), t0, _mm256_loadu_pd(yb + col)));

            for (; col < nb; col++)
                yb[col] += t0s * a0[col];
        }
    }
}
