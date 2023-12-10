//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------
#include <stdio.h>
#include <windows.h>
#include "cblas.h"

extern volatile int cblas_max_threads;

static HANDLE cblas_threads[MAX_THREADS];
static DWORD cblas_thread_ids[MAX_THREADS];

static HANDLE kickoff_event = NULL;

static CRITICAL_SECTION queue_lock;

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
// initialize the thread server system
//------------------------------------------------------
void cblas_init_server()
{
    // create the kickoff Event
    kickoff_event = CreateEvent(NULL, TRUE, FALSE, NULL);
    
    // initialize the queue lock
    InitializeCriticalSection(&queue_lock);

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
}

//------------------------------------------------------
// shutdown the thread server
//------------------------------------------------------
void cblas_shutdown()
{
    // TODO - shutdown threads?

    // cleanup event
    CloseHandle(kickoff_event);

    DeleteCriticalSection(&queue_lock);
}

//------------------------------------------------------
// worker thread function
//------------------------------------------------------
static DWORD WINAPI cblas_worker_thread(void *pvArg)
{
    work_queue_t* work_item;

    int thread_num = (int)pvArg;

    MT_TRACE("thread [%d] created.\n", thread_num);

    while(1)
    {
        MT_TRACE("thread [%d] waits.\n", thread_num);

        // event raised when work is added to the queue
        WaitForSingleObject(kickoff_event, INFINITE);

        MT_TRACE("thread [%d] is awake.\n", thread_num);
  
        if (thread_num > cblas_max_threads - 2)
        {
            MT_TRACE("thread [%d] exiting.\n", thread_num);

            // excess thread, so worker thread exits
            break;
        }

        // check for more work, if so remove it from queue
#if 0
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
            MT_TRACE("thread [%d] no work, trying again.\n", thread_num);
            continue;
        }

        work_item->thread_num = thread_num;
        work_item->tid = cblas_thread_ids[thread_num];

        MT_TRACE("thread [%d] executing a task.\n", thread_num);

        // execute the task
        work_item->kernel(work_item->args);

        assert(work_item->finished == 0);
        LONG r = InterlockedIncrement(&work_item->finished);

        assert(r == 1);

        MT_TRACE("thread [%d] task completed.\n", thread_num);
    }

    return 0;
}

//------------------------------------------------------
// execute a work queue synchronously
//------------------------------------------------------
void cblas_execute(int items, work_queue_t *queue)
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
void cblas_execute_async(int items, work_queue_t* queue)
{
    assert(queue);

    MT_TRACE("adding %d items to the queue.\n", items);

    // add new work to the end of the work_queue
    EnterCriticalSection(&queue_lock);

    if (!work_queue)
    {
        work_queue = queue;
    }
    else
    {
        MT_TRACE("work_queue was not empty!\n");

        work_queue_t* next_item = work_queue;

        // find the end of the work queue
        while (next_item)
            next_item = next_item->next;

        // add new work to the end
        next_item = queue;
    }

    LeaveCriticalSection(&queue_lock);

    // wake up the worker threads
    MT_TRACE("waking worker threads.\n");

    SetEvent(kickoff_event);
}

//------------------------------------------------------
// wait for the set of tasks to complete
//------------------------------------------------------
void cblas_execute_async_join(int items, work_queue_t* queue)
{
    assert(queue);

    MT_TRACE("waiting on queue to complete %d items.\n", items);

    while (items)
    {
        while (!queue->finished)
            YieldProcessor();

        queue = queue->next;
        items--;
    }

    MT_TRACE("queued tasks finished.\n");

    // if work was added to the queue after this batch we can't sleep the worker threads
    // by resetting the event
    EnterCriticalSection(&queue_lock);

        if (work_queue == NULL)
            ResetEvent(kickoff_event);

    LeaveCriticalSection(&queue_lock);
}
