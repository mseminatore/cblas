//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------
#include <stdio.h>
#include <windows.h>
#include "cblas.h"

extern volatile int cblas_max_threads;

static HANDLE cblas_threads[MAX_THREADS] = {NULL};
static DWORD cblas_thread_ids[MAX_THREADS] = {0};

static HANDLE kickoff_event = NULL;

static CRITICAL_SECTION queue_lock;
static CRITICAL_SECTION server_lock;

static work_queue_t *work_queue = NULL;

DWORD WINAPI cblas_worker_thread(void* pvoid);

#if defined (__GNUC__) && (__GNUC__ < 6)
#   define WIN_CAS(dest, exch, comp) __sync_val_compare_and_swap(dest, comp, exch)
#else
    #if defined(_WIN64)
    #   define WIN_CAS(dest, exch, comp) InterlockedCompareExchange64(dest, exch, comp)
    #else
    #   define WIN_CAS(dest, exch, comp) InterlockedCompareExchange(dest, exch, comp)
    #endif
#endif

//------------------------------------------------------
// set the active number of threads [1..cores]
//------------------------------------------------------
void cblas_set_num_threads(int threads)
{
    MT_TRACE("set threads = %d\n", threads);

    if (threads < 1)
        threads = 1;

    if (threads > MAX_THREADS)
        threads = MAX_THREADS;

    int cores = cpu_get_core_count();
    if (threads > cores)
    {
        threads = cores;
    }

    // reduce threads
    if (cblas_is_server_alive() && threads < cblas_max_threads)
    {
        EnterCriticalSection(&server_lock);

        int thread_count = cblas_max_threads;
        cblas_max_threads = threads;

        SetEvent(kickoff_event);

        for (int i = threads - 1; i < thread_count - 1; i++)
        {
            if (cblas_threads[i] != NULL)
            {
                MT_TRACE("set_num_threads: waiting on thread [%d] to quit.\n", i);

                WaitForSingleObject(cblas_threads[i], INFINITE);

                MT_TRACE("set_num_threads: thread [%d] has quit.\n", i);

                CloseHandle(cblas_threads[i]);
                cblas_threads[i] = NULL;
            }
        }

        ResetEvent(kickoff_event);

        LeaveCriticalSection(&server_lock);
    }

    // add more threads if needed
    if (cblas_is_server_alive() && threads > cblas_max_threads)
    {
        EnterCriticalSection(&server_lock);

        int start = cblas_max_threads > 0 ? cblas_max_threads - 1 : 0;
        cblas_max_threads = threads;

        for (int i = start; i < threads - 1; i++)
        {
            cblas_threads[i] = CreateThread(
                NULL,
                0,
                cblas_worker_thread,
                (void*)(i),
                0,
                &cblas_thread_ids[i]
            );
        }

        LeaveCriticalSection(&server_lock);
    }
}

//------------------------------------------------------
// initialize the thread server system
//------------------------------------------------------
int cblas_init_server()
{
    if (cblas_is_server_alive() || cblas_max_threads <= 1)
        return CBLAS_FALSE;

    // create the kickoff Event
    kickoff_event = CreateEvent(NULL, TRUE, FALSE, NULL);
    
    // initialize the queue lock
    InitializeCriticalSection(&queue_lock);
    InitializeCriticalSection(&server_lock);

    EnterCriticalSection(&server_lock);

    // create the worker threads
    for (INT_PTR i = 0; i < cblas_max_threads - 1; i++)
    {
        cblas_threads[i] = CreateThread(
            NULL, 
            0, 
            cblas_worker_thread, 
            (void*)i, 
            0,
            &cblas_thread_ids[i]
        );
    }

    cblas_set_server_alive(CBLAS_TRUE);

    LeaveCriticalSection(&server_lock);

    return CBLAS_TRUE;
}

//------------------------------------------------------
// shutdown the thread server
//------------------------------------------------------
void cblas_shutdown()
{
    if (!cblas_is_server_alive())
        return;

    cblas_set_server_alive(CBLAS_FALSE);

    // Wake all threads and wait for them to exit gracefully
    EnterCriticalSection(&server_lock);
    
    int thread_count = cblas_max_threads;
    cblas_max_threads = 1;  // Signal all threads to exit
    
    // Wake up all waiting threads
    SetEvent(kickoff_event);
    
    // Wait for all threads to complete
    for (int i = 0; i < thread_count - 1; i++)
    {
        if (cblas_threads[i] != NULL)
        {
            MT_TRACE("shutdown: waiting on thread [%d] to quit.\n", i);
            WaitForSingleObject(cblas_threads[i], INFINITE);
            MT_TRACE("shutdown: thread [%d] has quit.\n", i);
            CloseHandle(cblas_threads[i]);
            cblas_threads[i] = NULL;
        }
    }
    
    LeaveCriticalSection(&server_lock);

    // cleanup event and locks
    CloseHandle(kickoff_event);
    kickoff_event = NULL;

    DeleteCriticalSection(&queue_lock);
    DeleteCriticalSection(&server_lock);

    // cleanup stats resources
    cblas_cleanup_stats();
}

//------------------------------------------------------
// worker thread function
//------------------------------------------------------
static DWORD WINAPI cblas_worker_thread(void *pvArg)
{
    work_queue_t* work_item;

    int thread_num = (int)pvArg;

    MT_TRACE_THREAD(thread_num, "created.\n");

    while(1)
    {
        //MT_TRACE_THREAD(thread_num, "waits.\n");

        // event raised when work is added to the queue
        WaitForSingleObject(kickoff_event, INFINITE);

        //MT_TRACE_THREAD(thread_num, "is awake.\n");
  
        if (thread_num > cblas_max_threads - 2)
        {
            MT_TRACE_THREAD(thread_num, "exiting.\n");

            // excess thread, so worker thread exits
            break;
        }

        // check for more work, if so remove it from queue
#if 1
        EnterCriticalSection(&queue_lock);

        work_item = work_queue;
        if (work_item)
            work_queue = work_queue->next;

        LeaveCriticalSection(&queue_lock);
#else
        volatile work_queue_t* queue_next;

        INT_PTR prev_value;
        do {
            work_item = (volatile work_queue_t*)work_queue;
            if (!work_item)
                break;

            queue_next = (volatile work_queue_t*)work_item->next;
            prev_value = WIN_CAS((INT_PTR*)&work_queue, (INT_PTR)queue_next, (INT_PTR)work_item);
        } while (prev_value != work_item);
#endif

        // if no work, reset event and then go to sleep to wait for more work
        if (!work_item)
        {
            //MT_TRACE_THREAD(thread_num, "no work, trying again.\n");
            YieldProcessor();
            continue;
        }

        work_item->thread_num = thread_num;
        work_item->tid = cblas_thread_ids[thread_num];

        MT_TRACE_THREAD(thread_num, "executing a task.\n");

#ifdef MT_DEBUG
        // Track timing for this work item
        work_item->start_time_us = mt_get_time_us();
#endif

        // execute the task
        work_item->kernel(work_item->args);

        assert(work_item->finished == 0);
        LONG r = InterlockedIncrement(&work_item->finished);

        assert(r == 1);

#ifdef MT_DEBUG
        // Calculate and log execution time
        work_item->end_time_us = mt_get_time_us();
        double duration = work_item->end_time_us - work_item->start_time_us;
        const char* op = work_item->operation ? work_item->operation : "task";
        MT_TRACE_TIMING(thread_num, op, duration);
#endif

        MT_TRACE_THREAD(thread_num, "task completed.\n");
    }

    return 0;
}

//------------------------------------------------------
// execute a work queue synchronously
//------------------------------------------------------
void cblas_execute(CBLAS_INDEX items, work_queue_t *queue)
{
    assert(items > 0 && queue);

    if (items <= 0 || queue == NULL)
        return;

    // submit task queue
    if (items > 1 && queue->next)
        cblas_execute_async(items - 1, queue->next);

    // execute the first task on the main thread
    queue->kernel(queue->args);
    InterlockedIncrement(&queue->finished);

    // wait for the queue of work to finish
    if (items > 1 && queue->next)
        cblas_execute_async_join(items - 1, queue->next);
}

//------------------------------------------------------
// execute a work queue asynchronously
//------------------------------------------------------
void cblas_execute_async(CBLAS_INDEX items, work_queue_t* queue)
{
    assert(queue);

    MT_TRACE("adding %zu items to the queue.\n", items);

    // add new work to the end of the work_queue
    EnterCriticalSection(&queue_lock);

    if (!work_queue)
    {
        work_queue = queue;
    }
    else
    {
        MT_TRACE("work_queue was not empty!\n");

        work_queue_t* queue_item = work_queue;

        // find the end of the work queue
        while (queue_item->next)
            queue_item = queue_item->next;

        // add new work to the end
        queue_item->next = queue;
    }

#ifdef MT_DEBUG
    // Count queue depth for monitoring
    // NOTE: This is O(n) and runs inside the critical section, so it adds overhead
    // when MT_DEBUG is enabled. This is acceptable for debugging but should not
    // be enabled in production builds.
    int depth = 0;
    work_queue_t* item = work_queue;
    while (item) {
        depth++;
        item = item->next;
    }
    MT_TRACE_QUEUE_DEPTH(depth);
#endif

    LeaveCriticalSection(&queue_lock);

    // wake up the worker threads
    MT_TRACE("waking worker threads.\n");

    SetEvent(kickoff_event);
}

//------------------------------------------------------
// wait for the set of tasks to complete
//------------------------------------------------------
void cblas_execute_async_join(CBLAS_INDEX items, work_queue_t* queue)
{
    assert(queue);

    MT_TRACE("waiting on queue to complete %zu items.\n", items);

#ifdef MT_DEBUG
    // Save the start of the queue to collect timing stats later
    work_queue_t* queue_start = queue;
#endif

    while (items)
    {
        while (!queue->finished)
            YieldProcessor();

        queue = queue->next;
        items--;
    }

    MT_TRACE("queued tasks finished.\n");

#ifdef MT_DEBUG
    // Collect timing data and detect load imbalance
    // NOTE: queue_start points to stack-allocated work items in the caller.
    // This is safe here because we access them before returning (and thus before
    // they go out of scope in the calling function).
    if (queue_start) {
        double times[MAX_THREADS];
        int count = 0;
        work_queue_t* item = queue_start;
        
        while (item && count < MAX_THREADS) {
            if (item->end_time_us > 0 && item->start_time_us > 0) {
                times[count] = item->end_time_us - item->start_time_us;
                count++;
            }
            item = item->next;
        }
        
        if (count > 1) {
            MT_TRACE_LOAD_BALANCE(count, times);
        }
    }
#endif

    // if work was added to the queue after this batch we can't sleep the worker threads
    // by resetting the event
    EnterCriticalSection(&queue_lock);

        if (work_queue == NULL)
            ResetEvent(kickoff_event);

    LeaveCriticalSection(&queue_lock);
}
