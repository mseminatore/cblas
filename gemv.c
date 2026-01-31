//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"
#include "cblas_simd.h"

#ifdef _WIN32
#	include <malloc.h>
#else
#	include <alloca.h>
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)

//------------------------------------------------------
// Optimized single-precision row-wise dot product kernel (incx==1)
// Computes one row of matrix-vector product using SIMD and multi-accumulator unrolling
//------------------------------------------------------
static void sgemv_row_dot_noinc(float* a_row, float* x, CBLAS_INDEX n, float* result)
{
    CBLAS_INDEX i = 0;
    float sum = 0.0f;
    
#if defined(__AVX2__) && defined(USE_SIMD)
    // AVX2 path: Use 4 independent accumulators to hide latency
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();
    
    // Process 32 floats per iteration (4 accumulators × 8 floats)
    CBLAS_INDEX unroll_end = (n / 32) * 32;
    
#if defined(CBLAS_PREFETCH)
    const CBLAS_INDEX prefetch_distance = CBLAS_PREFETCH_DISTANCE;
#endif
    
    for (; i < unroll_end; i += 32)
    {
#if defined(CBLAS_PREFETCH)
        // Prefetch data ahead to hide memory latency
        if (i + prefetch_distance < n) {
            CBLAS_PREFETCH(a_row + i + prefetch_distance, 0, 3);
            CBLAS_PREFETCH(x + i + prefetch_distance, 0, 3);
        }
#endif
        
        __m256 a0 = _mm256_loadu_ps(a_row + i);
        __m256 x0 = _mm256_loadu_ps(x + i);
        __m256 a1 = _mm256_loadu_ps(a_row + i + 8);
        __m256 x1 = _mm256_loadu_ps(x + i + 8);
        __m256 a2 = _mm256_loadu_ps(a_row + i + 16);
        __m256 x2 = _mm256_loadu_ps(x + i + 16);
        __m256 a3 = _mm256_loadu_ps(a_row + i + 24);
        __m256 x3 = _mm256_loadu_ps(x + i + 24);
        
#if defined(USE_INTEL_FMA)
        sum0 = _mm256_fmadd_ps(a0, x0, sum0);
        sum1 = _mm256_fmadd_ps(a1, x1, sum1);
        sum2 = _mm256_fmadd_ps(a2, x2, sum2);
        sum3 = _mm256_fmadd_ps(a3, x3, sum3);
#else
        sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(a0, x0));
        sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(a1, x1));
        sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(a2, x2));
        sum3 = _mm256_add_ps(sum3, _mm256_mul_ps(a3, x3));
#endif
    }
    
    // Combine the 4 accumulators
    __m256 sum_avx = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
    
    // Handle remaining blocks of 8
    for (; i + 8 <= n; i += 8)
    {
        __m256 a_vec = _mm256_loadu_ps(a_row + i);
        __m256 x_vec = _mm256_loadu_ps(x + i);
        
#if defined(USE_INTEL_FMA)
        sum_avx = _mm256_fmadd_ps(a_vec, x_vec, sum_avx);
#else
        sum_avx = _mm256_add_ps(sum_avx, _mm256_mul_ps(a_vec, x_vec));
#endif
    }
    
    // Horizontal sum of AVX vector
    float sum_array[8];
    _mm256_storeu_ps(sum_array, sum_avx);
    sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3] +
          sum_array[4] + sum_array[5] + sum_array[6] + sum_array[7];
#endif
    
    // Handle remaining elements
    for (; i < n; i++)
    {
        sum += a_row[i] * x[i];
    }
    
    *result = sum;
}

//------------------------------------------------------
// Optimized double-precision row-wise dot product kernel (incx==1)
//------------------------------------------------------
static void dgemv_row_dot_noinc(double* a_row, double* x, CBLAS_INDEX n, double* result)
{
    CBLAS_INDEX i = 0;
    double sum = 0.0;
    
#if defined(__AVX2__) && defined(USE_SIMD)
    // AVX2 path: Use 4 independent accumulators
    __m256d sum0 = _mm256_setzero_pd();
    __m256d sum1 = _mm256_setzero_pd();
    __m256d sum2 = _mm256_setzero_pd();
    __m256d sum3 = _mm256_setzero_pd();
    
    // Process 16 doubles per iteration (4 accumulators × 4 doubles)
    CBLAS_INDEX unroll_end = (n / 16) * 16;
    
#if defined(CBLAS_PREFETCH)
    const CBLAS_INDEX prefetch_distance = CBLAS_PREFETCH_DISTANCE;
#endif
    
    for (; i < unroll_end; i += 16)
    {
#if defined(CBLAS_PREFETCH)
        if (i + prefetch_distance < n) {
            CBLAS_PREFETCH(a_row + i + prefetch_distance, 0, 3);
            CBLAS_PREFETCH(x + i + prefetch_distance, 0, 3);
        }
#endif
        
        __m256d a0 = _mm256_loadu_pd(a_row + i);
        __m256d x0 = _mm256_loadu_pd(x + i);
        __m256d a1 = _mm256_loadu_pd(a_row + i + 4);
        __m256d x1 = _mm256_loadu_pd(x + i + 4);
        __m256d a2 = _mm256_loadu_pd(a_row + i + 8);
        __m256d x2 = _mm256_loadu_pd(x + i + 8);
        __m256d a3 = _mm256_loadu_pd(a_row + i + 12);
        __m256d x3 = _mm256_loadu_pd(x + i + 12);
        
#if defined(USE_INTEL_FMA)
        sum0 = _mm256_fmadd_pd(a0, x0, sum0);
        sum1 = _mm256_fmadd_pd(a1, x1, sum1);
        sum2 = _mm256_fmadd_pd(a2, x2, sum2);
        sum3 = _mm256_fmadd_pd(a3, x3, sum3);
#else
        sum0 = _mm256_add_pd(sum0, _mm256_mul_pd(a0, x0));
        sum1 = _mm256_add_pd(sum1, _mm256_mul_pd(a1, x1));
        sum2 = _mm256_add_pd(sum2, _mm256_mul_pd(a2, x2));
        sum3 = _mm256_add_pd(sum3, _mm256_mul_pd(a3, x3));
#endif
    }
    
    // Combine the 4 accumulators
    __m256d sum_avx = _mm256_add_pd(_mm256_add_pd(sum0, sum1), _mm256_add_pd(sum2, sum3));
    
    // Handle remaining blocks of 4
    for (; i + 4 <= n; i += 4)
    {
        __m256d a_vec = _mm256_loadu_pd(a_row + i);
        __m256d x_vec = _mm256_loadu_pd(x + i);
        
#if defined(USE_INTEL_FMA)
        sum_avx = _mm256_fmadd_pd(a_vec, x_vec, sum_avx);
#else
        sum_avx = _mm256_add_pd(sum_avx, _mm256_mul_pd(a_vec, x_vec));
#endif
    }
    
    // Horizontal sum
    double sum_array[4];
    _mm256_storeu_pd(sum_array, sum_avx);
    sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];
#endif
    
    // Handle remaining elements
    for (; i < n; i++)
    {
        sum += a_row[i] * x[i];
    }
    
    *result = sum;
}

#endif // x86_64

//------------------------------------------------------
// GEMV kernel for multi-threading - single precision (NoTrans case)
//------------------------------------------------------
void sgemv_k(cblas_args_t* args)
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

	float sum;
	
	// Optimized path for unit strides
	if (incx == 1 && incy == 1)
	{
		for (CBLAS_INDEX row = 0; row < m; row++)
		{
#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)
			sgemv_row_dot_noinc(&a[row * lda], x, n, &sum);
#else
			sum = 0.0f;
			for (CBLAS_INDEX col = 0; col < n; col++)
			{
				sum += a[row * lda + col] * x[col];
			}
#endif
			y[row] = beta * y[row] + alpha * sum;
		}
	}
	else
	{
		// General case with strides
		for (CBLAS_INDEX row = 0; row < m; row++)
		{
			sum = 0.0f;
			for (CBLAS_INDEX col = 0; col < n; col++)
			{
				sum += a[row * lda + col] * x[col * incx];
			}
			y[row * incy] = beta * y[row * incy] + alpha * sum;
		}
	}
}

//------------------------------------------------------
// GEMV kernel for multi-threading - double precision (NoTrans case)
//------------------------------------------------------
void dgemv_k(cblas_args_t* args)
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

	double sum;
	
	// Optimized path for unit strides
	if (incx == 1 && incy == 1)
	{
		for (CBLAS_INDEX row = 0; row < m; row++)
		{
#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)
			dgemv_row_dot_noinc(&a[row * lda], x, n, &sum);
#else
			sum = 0.0;
			for (CBLAS_INDEX col = 0; col < n; col++)
			{
				sum += a[row * lda + col] * x[col];
			}
#endif
			y[row] = beta * y[row] + alpha * sum;
		}
	}
	else
	{
		// General case with strides
		for (CBLAS_INDEX row = 0; row < m; row++)
		{
			sum = 0.0;
			for (CBLAS_INDEX col = 0; col < n; col++)
			{
				sum += a[row * lda + col] * x[col * incx];
			}
			y[row * incy] = beta * y[row * incy] + alpha * sum;
		}
	}
}

//------------------------------------------------------
// Level-2 single-precision matrix-vector multiply
// y = alpha * A * x + beta * y
//------------------------------------------------------
void cblas_sgemv(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, CBLAS_INDEX m, CBLAS_INDEX n, float alpha, float* a, CBLAS_INDEX lda, float* x, CBLAS_INDEX incx, float beta, float* y, CBLAS_INDEX incy)
{
#ifdef CBLAS_CHECK_INPUTS

#ifdef CBLAS_XERBLA_INPUTS
	int info = 0;
	if (!a)
		info = 6;
	else if (lda < MAX(1, m))
		info = 7;
	else if (!x)
		info = 8;
	else if (incx == 0)
		info = 9;
	else if (!y)
		info = 11;
	else if (incy == 0)
		info = 12;

	if (info) {
		XERBLA(info);
		return;
	}
#else
	if (!a || lda < MAX(1, m) || !x || incx == 0 || !y || incy == 0)
	{
		assert(m >= 0 && n >= 0 && a && lda >= MAX(1, m) && x && y && incx != 0 && incy != 0);
		return;
	}
#endif
#endif
	// early returns
	if (m == 0 || n == 0 || (alpha == 0.0f && beta == 1.0f))
		return;

    CBLAS_STATS_START();

#if defined(MT_ENABLED)
    int mt_used = (m * n > CBLAS_MT_GEMV) ? 1 : 0;
    
    // Multi-threaded path for NoTrans case with RowMajor or Trans case with ColMajor
    if (mt_used && ((trans == CblasNoTrans && layout == CblasRowMajor) || (trans == CblasTrans && layout == CblasColMajor)) && blas_kernels.sgemv_k != NULL)
    {
        // Partition rows across threads
        CBLAS_INDEX thread_count = CLAMP(cblas_get_num_threads(), 1, MAX_THREADS);
        
        #ifdef _WIN32
            work_queue_t *queue = _malloca(thread_count * sizeof(work_queue_t));
            cblas_args_t *args = _malloca(thread_count * sizeof(cblas_args_t));
        #else
            work_queue_t *queue = alloca(thread_count * sizeof(work_queue_t));
            cblas_args_t *args = alloca(thread_count * sizeof(cblas_args_t));
        #endif
        
        CBLAS_INDEX rows_remaining = m;
        CBLAS_INDEX row_offset = 0;
        
        for (CBLAS_INDEX i = 0; i < thread_count; i++)
        {
            // Compute partition size: distribute rows evenly
            CBLAS_INDEX rows_per_thread = (rows_remaining + thread_count - i - 1) / (thread_count - i);
            
            args[i].m = rows_per_thread;
            args[i].n = n;
            args[i].lda = lda;
            args[i].incx = incx;
            args[i].incy = incy;
            args[i].a = &a[row_offset * lda];
            args[i].x = x;
            args[i].y = &y[row_offset * incy];
            args[i].alpha = &alpha;
            args[i].beta = &beta;
            
            queue[i].finished = 0;
            queue[i].args = &args[i];
            queue[i].kernel = blas_kernels.sgemv_k;
            queue[i].next = &queue[i + 1];
            
            row_offset += rows_per_thread;
            rows_remaining -= rows_per_thread;
        }
        
        // mark end of task queue
        queue[thread_count - 1].next = NULL;
        
        // synchronously execute task queue
        cblas_execute(thread_count, queue);
    }
    else
#else
    int mt_used = 0;
#endif
    {
        // Single-threaded fallback path
        float sum;

	if ((trans == CblasNoTrans && layout == CblasRowMajor) || (trans == CblasTrans && layout == CblasColMajor))
	{
		// Optimized path for unit strides (common case)
		if (incx == 1 && incy == 1)
		{
			for (CBLAS_INDEX row = 0; row < m; row++)
			{
#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)
				sgemv_row_dot_noinc(&a[row * lda], x, n, &sum);
#else
				sum = 0.0f;
				for (CBLAS_INDEX col = 0; col < n; col++)
				{
					sum += a[row * lda + col] * x[col];
				}
#endif
				y[row] = beta * y[row] + alpha * sum;
			}
		}
		else
		{
			// General case with strides
			for (CBLAS_INDEX row = 0; row < m; row++)
			{
				sum = 0.0f;
				for (CBLAS_INDEX col = 0; col < n; col++)
				{
					sum += a[row * lda + col] * x[col * incx];
				}
				y[row * incy] = beta * y[row * incy] + alpha * sum;
			}
		}
	}
	else
	{
		// Transpose case: column-wise access
		if (incx == 1 && incy == 1)
		{
			for (CBLAS_INDEX col = 0; col < n; col++)
			{
				sum = 0.0f;
				for (CBLAS_INDEX row = 0; row < m; row++)
				{
					sum += a[row * lda + col] * x[row];
				}
				y[col] = beta * y[col] + alpha * sum;
			}
		}
		else
		{
			for (CBLAS_INDEX col = 0; col < n; col++)
			{
				sum = 0.0f;
				for (CBLAS_INDEX row = 0; row < m; row++)
				{
					sum += a[row * lda + col] * x[row * incx];
				}
				y[col * incy] = beta * y[col * incy] + alpha * sum;
			}
		}
	}
    }

    CBLAS_STATS_END("sgemv", m * n, mt_used);
}

//------------------------------------------------------
// Level-2 double-precision matrix-vector multiply
// y = alpha * A * x + beta * y
//------------------------------------------------------
void cblas_dgemv(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, CBLAS_INDEX m, CBLAS_INDEX n, double alpha, double* a, CBLAS_INDEX lda, double* x, CBLAS_INDEX incx, double beta, double* y, CBLAS_INDEX incy)
{
#ifdef CBLAS_CHECK_INPUTS

#ifdef CBLAS_XERBLA_INPUTS
	int info = 0;
	if (!a)
		info = 6;
	else if (lda < MAX(1, m))
		info = 7;
	else if (!x)
		info = 8;
	else if (incx == 0)
		info = 9;
	else if (!y)
		info = 11;
	else if (incy == 0)
		info = 12;

	if (info) {
		XERBLA(info);
		return;
	}
#else
	if (!a || lda < MAX(1, m) || !x || incx == 0 || !y || incy == 0)
	{
		assert(m >= 0 && n >= 0 && a && lda >= MAX(1, m) && x && y && incx != 0 && incy != 0);
		return;
	}
#endif
#endif
	// early returns
	if (m == 0 || n == 0 || (alpha == 0.0 && beta == 1.0))
		return;

    CBLAS_STATS_START();

#if defined(MT_ENABLED)
    int mt_used = (m * n > CBLAS_MT_GEMV) ? 1 : 0;
    
    // Multi-threaded path for NoTrans case with RowMajor or Trans case with ColMajor
    if (mt_used && ((trans == CblasNoTrans && layout == CblasRowMajor) || (trans == CblasTrans && layout == CblasColMajor)) && blas_kernels.dgemv_k != NULL)
    {
        // Partition rows across threads
        CBLAS_INDEX thread_count = CLAMP(cblas_get_num_threads(), 1, MAX_THREADS);
        
        #ifdef _WIN32
            work_queue_t *queue = _malloca(thread_count * sizeof(work_queue_t));
            cblas_args_t *args = _malloca(thread_count * sizeof(cblas_args_t));
        #else
            work_queue_t *queue = alloca(thread_count * sizeof(work_queue_t));
            cblas_args_t *args = alloca(thread_count * sizeof(cblas_args_t));
        #endif
        
        CBLAS_INDEX rows_remaining = m;
        CBLAS_INDEX row_offset = 0;
        
        for (CBLAS_INDEX i = 0; i < thread_count; i++)
        {
            // Compute partition size: distribute rows evenly
            CBLAS_INDEX rows_per_thread = (rows_remaining + thread_count - i - 1) / (thread_count - i);
            
            args[i].m = rows_per_thread;
            args[i].n = n;
            args[i].lda = lda;
            args[i].incx = incx;
            args[i].incy = incy;
            args[i].a = &a[row_offset * lda];
            args[i].x = x;
            args[i].y = &y[row_offset * incy];
            args[i].alpha = &alpha;
            args[i].beta = &beta;
            
            queue[i].finished = 0;
            queue[i].args = &args[i];
            queue[i].kernel = blas_kernels.dgemv_k;
            queue[i].next = &queue[i + 1];
            
            row_offset += rows_per_thread;
            rows_remaining -= rows_per_thread;
        }
        
        // mark end of task queue
        queue[thread_count - 1].next = NULL;
        
        // synchronously execute task queue
        cblas_execute(thread_count, queue);
    }
    else
#else
    int mt_used = 0;
#endif
    {
        // Single-threaded fallback path
        double sum;
	if ((trans == CblasNoTrans && layout == CblasRowMajor) || (trans == CblasTrans && layout == CblasColMajor))
	{
		// Optimized path for unit strides (common case)
		if (incx == 1 && incy == 1)
		{
			for (CBLAS_INDEX row = 0; row < m; row++)
			{
#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)
				dgemv_row_dot_noinc(&a[row * lda], x, n, &sum);
#else
				sum = 0.0;
				for (CBLAS_INDEX col = 0; col < n; col++)
				{
					sum += a[row * lda + col] * x[col];
				}
#endif
				y[row] = beta * y[row] + alpha * sum;
			}
		}
		else
		{
			// General case with strides
			for (CBLAS_INDEX row = 0; row < m; row++)
			{
				sum = 0.0;
				for (CBLAS_INDEX col = 0; col < n; col++)
				{
					sum += a[row * lda + col] * x[col * incx];
				}
				y[row * incy] = beta * y[row * incy] + alpha * sum;
			}
		}
	}
	else
	{
		// Transpose case: column-wise access
		if (incx == 1 && incy == 1)
		{
			for (CBLAS_INDEX col = 0; col < n; col++)
			{
				sum = 0.0;
				for (CBLAS_INDEX row = 0; row < m; row++)
				{
					sum += a[row * lda + col] * x[row];
				}
				y[col] = beta * y[col] + alpha * sum;
			}
		}
		else
		{
			for (CBLAS_INDEX col = 0; col < n; col++)
			{
				sum = 0.0;
				for (CBLAS_INDEX row = 0; row < m; row++)
				{
					sum += a[row * lda + col] * x[row * incx];
				}
				y[col * incy] = beta * y[col * incy] + alpha * sum;
			}
		}
	}
    }

    CBLAS_STATS_END("dgemv", m * n, mt_used);
}
