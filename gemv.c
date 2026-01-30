//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"

#ifdef _WIN32
#	include <malloc.h>
#else
#	include <alloca.h>
#endif

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
	
	// Process rows in cache-friendly blocks
	for (CBLAS_INDEX i = 0; i < m; i += GEMV_BLOCK_SIZE)
	{
		CBLAS_INDEX ib = (i + GEMV_BLOCK_SIZE < m) ? GEMV_BLOCK_SIZE : (m - i);
		
		// Process ib rows at a time
		for (CBLAS_INDEX row = i; row < i + ib; row++)
		{
			sum = 0.0f;
			for (CBLAS_INDEX col = 0; col < n; col++)
			{
				sum += alpha * a[row * lda + col] * x[col * incx];
			}
			y[row * incy] = beta * y[row * incy] + sum;
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
	
	// Process rows in cache-friendly blocks
	for (CBLAS_INDEX i = 0; i < m; i += GEMV_BLOCK_SIZE)
	{
		CBLAS_INDEX ib = (i + GEMV_BLOCK_SIZE < m) ? GEMV_BLOCK_SIZE : (m - i);
		
		// Process ib rows at a time
		for (CBLAS_INDEX row = i; row < i + ib; row++)
		{
			sum = 0.0;
			for (CBLAS_INDEX col = 0; col < n; col++)
			{
				sum += alpha * a[row * lda + col] * x[col * incx];
			}
			y[row * incy] = beta * y[row * incy] + sum;
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
		if (alpha == 1.0f && beta == 1.0f)
		{
			// Process rows in cache-friendly blocks
			for (CBLAS_INDEX i = 0; i < m; i += GEMV_BLOCK_SIZE)
			{
				CBLAS_INDEX ib = (i + GEMV_BLOCK_SIZE < m) ? GEMV_BLOCK_SIZE : (m - i);
				
				// for each row in the block
				for (CBLAS_INDEX row = i; row < i + ib; row++)
				{
					sum = 0.0f;

					for (CBLAS_INDEX col = 0; col < n; col++)
					{
						sum += a[row * n + col] * x[col];
					}

					y[row] = y[row] + sum;
				}
			}
		}
		else
		{
			// Process rows in cache-friendly blocks
			for (CBLAS_INDEX i = 0; i < m; i += GEMV_BLOCK_SIZE)
			{
				CBLAS_INDEX ib = (i + GEMV_BLOCK_SIZE < m) ? GEMV_BLOCK_SIZE : (m - i);
				
				// for each row in the block
				for (CBLAS_INDEX row = i; row < i + ib; row++)
				{
					sum = 0.0f;

					for (CBLAS_INDEX col = 0; col < n; col++)
					{
						sum += alpha * a[row * n + col] * x[col];
					}

					y[row] = beta * y[row] + sum;
				}
			}
		}
	}
	else
	{
		if (alpha == 1.0f && beta == 1.0f)
		{
			for (CBLAS_INDEX col = 0; col < n; col++)
			{
				sum = 0.0f;

				for (CBLAS_INDEX row = 0; row < m; row++)
				{
					sum += a[row * n + col] * x[row];
				}

				y[col] = y[col] + sum;
			}
		}
		else
		{
			for (CBLAS_INDEX col = 0; col < n; col++)
			{
				sum = 0.0f;

				for (CBLAS_INDEX row = 0; row < m; row++)
				{
					sum += alpha * a[row * n + col] * x[row];
				}

				y[col] = beta * y[col] + sum;
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
		if (alpha == 1.0 && beta == 1.0)
		{
			// Process rows in cache-friendly blocks
			for (CBLAS_INDEX i = 0; i < m; i += GEMV_BLOCK_SIZE)
			{
				CBLAS_INDEX ib = (i + GEMV_BLOCK_SIZE < m) ? GEMV_BLOCK_SIZE : (m - i);
				
				// for each row in the block
				for (CBLAS_INDEX row = i; row < i + ib; row++)
				{
					sum = 0.0;

					for (CBLAS_INDEX col = 0; col < n; col++)
					{
						sum += a[row * n + col] * x[col];
					}

					y[row] = y[row] + sum;
				}
			}
		}
		else
		{
			// Process rows in cache-friendly blocks
			for (CBLAS_INDEX i = 0; i < m; i += GEMV_BLOCK_SIZE)
			{
				CBLAS_INDEX ib = (i + GEMV_BLOCK_SIZE < m) ? GEMV_BLOCK_SIZE : (m - i);
				
				// for each row in the block
				for (CBLAS_INDEX row = i; row < i + ib; row++)
				{
					sum = 0.0;

					for (CBLAS_INDEX col = 0; col < n; col++)
					{
						sum += alpha * a[row * n + col] * x[col];
					}

					y[row] = beta * y[row] + sum;
				}
			}
		}
	}
	else
	{
		if (alpha == 1.0 && beta == 1.0)
		{
			for (CBLAS_INDEX col = 0; col < n; col++)
			{
				sum = 0.0;

				for (CBLAS_INDEX row = 0; row < m; row++)
				{
					sum += a[row * n + col] * x[row];
				}

				y[col] = y[col] + sum;
			}
		}
		else
		{
			for (CBLAS_INDEX col = 0; col < n; col++)
			{
				sum = 0.0;

				for (CBLAS_INDEX row = 0; row < m; row++)
				{
					sum += alpha * a[row * n + col] * x[row];
				}

				y[col] = beta * y[col] + sum;
			}
		}
	}
    }

    CBLAS_STATS_END("dgemv", m * n, mt_used);
}
