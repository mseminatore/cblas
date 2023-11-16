//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------
#include <stdio.h>
#include <assert.h>
#include <windows.h>
#include "cblas.h"

extern int cblas_max_threads;

static HANDLE cblas_threads[MAX_THREADS];
static DWORD cblas_thread_ids[MAX_THREADS];

static HANDLE kickoff_event = NULL;

CRITICAL_SECTION queue_lock;

static work_queue_t *work_queue = NULL;

DWORD WINAPI cblas_worker_thread(void* pvoid);

//#define MT_DEBUG

#ifdef MT_DEBUG
#   define MT_TRACE printf
#else
#   define MT_TRACE __noop
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

    MT_TRACE("thread %d created.\n", thread_num);

    while(1)
    {
        MT_TRACE("thread %d sleeps.\n", thread_num);

        // event raised when work is added to the queue
        WaitForSingleObject(kickoff_event, INFINITE);

        MT_TRACE("thread %d woke up.\n", thread_num);

        // check for more work, if so remove it from queue
        EnterCriticalSection(&queue_lock);
        
        work_item = work_queue;
        if (work_item)
            work_queue = work_queue->next;

        LeaveCriticalSection(&queue_lock);

        // if no work, reset event and then go to sleep to wait for more work
        if (!work_item)
        {
            ResetEvent(kickoff_event);
            continue;
        }

        // execute work
        work_item->kernel(NULL);

        InterlockedIncrement(&work_item->finished);
    }

    return 0;
}

//------------------------------------------------------
// execute a work queue synchronously
//------------------------------------------------------
void cblas_execute(int items, work_queue_t *queue)
{
    assert(queue);

    cblas_execute_async(items, queue);

    // wait for the queue of work to finish
    cblas_join(items, queue);
}

//------------------------------------------------------
// execute a work queue asynchronously
//------------------------------------------------------
void cblas_execute_async(int items, work_queue_t* queue)
{
    assert(queue);

    // add new work to the end of the work_queue
    EnterCriticalSection(&queue_lock);

    work_queue_t *next_item = work_queue;

    // find the end of the work queue
    while (next_item)
        next_item = next_item->next;

    // add new work to the end
    next_item = queue;

    LeaveCriticalSection(&queue_lock);

    // wake up the worker threads
    SetEvent(kickoff_event);
}

//------------------------------------------------------
// wait for the set of tasks to complete
//------------------------------------------------------
void cblas_join(int items, work_queue_t* queue)
{
    assert(queue);

    MT_TRACE("waiting on queue to complete.\n");

    while (items)
    {
        while (!*((volatile int*)&queue->finished));

        queue = queue->next;
        items--;
    }

    MT_TRACE("queue finished.\n");
}
