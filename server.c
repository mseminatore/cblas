//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------
#include <stdio.h>
#include <stdint.h>
#include "platform/threading.h"
#include "cblas.h"

extern volatile int cblas_max_threads;

static work_queue_t *work_queue = NULL;
static platform_thread_t cblas_thread_ids[MAX_THREADS] = {0};

// Forward declaration of worker thread function
#ifdef _WIN32
static DWORD WINAPI cblas_worker_thread(void *pvoid);
#else
static void *cblas_worker_thread(void *pvoid);
#endif

static platform_mutex_t queue_lock   = PLATFORM_MUTEX_INITIALIZER;
static platform_mutex_t server_lock   = PLATFORM_MUTEX_INITIALIZER;
static platform_cond_t kickoff_event = PLATFORM_COND_INITIALIZER;

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
        platform_mutex_lock(&server_lock);

        int thread_count = cblas_max_threads;
        cblas_max_threads = threads;

        platform_cond_broadcast(&kickoff_event);

        for (int i = threads - 1; i < thread_count - 1; i++)
        {
            if (cblas_thread_ids[i] != 0)
            {
                MT_TRACE("set_num_threads: waiting on thread [%d] to quit.\n", i);

                platform_thread_join(cblas_thread_ids[i]);

                MT_TRACE("set_num_threads: thread [%d] has quit.\n", i);
                
                cblas_thread_ids[i] = 0;
            }
        }

        platform_mutex_unlock(&server_lock);
    }

    // add more threads if needed
    if (cblas_is_server_alive() && threads > cblas_max_threads)
    {
        platform_mutex_lock(&server_lock);

        int start = cblas_max_threads > 0 ? cblas_max_threads - 1 : 0;
        cblas_max_threads = threads;

        for (int i = start; i < threads - 1; i++)
        {
            platform_thread_id_t tid CBLAS_UNUSED;
            platform_thread_create(&cblas_thread_ids[i], cblas_worker_thread, i, &tid);
        }

        platform_mutex_unlock(&server_lock);
    }
}

//------------------------------------------------------
// initialize the thread server system
//------------------------------------------------------
int cblas_init_server(void)
{
    if (cblas_is_server_alive() || cblas_max_threads <= 1)
        return CBLAS_FALSE;

    // pthread_mutex_init(&queue_lock, NULL);
    // pthread_cond_init(&kickoff_event, NULL);

    platform_mutex_lock(&server_lock);

    // create the worker threads
    for (int i = 0; i < cblas_max_threads - 1; i++)
    {
        platform_thread_id_t tid CBLAS_UNUSED;
        platform_thread_create(&cblas_thread_ids[i], cblas_worker_thread, i, &tid);
    }

    cblas_set_server_alive(CBLAS_TRUE);

    platform_mutex_unlock(&server_lock);

    return CBLAS_TRUE;
}

//------------------------------------------------------
// shutdown the thread server
//------------------------------------------------------
void cblas_shutdown(void)
{
    if (!cblas_is_server_alive())
        return;

    cblas_set_server_alive(CBLAS_FALSE);

    // Wake all threads and wait for them to exit gracefully
    platform_mutex_lock(&server_lock);
    
    int thread_count = cblas_max_threads;
    cblas_max_threads = 1;  // Signal all threads to exit
    
    // Wake up all waiting threads
    platform_cond_broadcast(&kickoff_event);
    
    // Wait for all threads to complete
    for (int i = 0; i < thread_count - 1; i++)
    {
        MT_TRACE("shutdown: waiting on thread [%d] to quit.\n", i);
        platform_thread_join(cblas_thread_ids[i]);
        MT_TRACE("shutdown: thread [%d] has quit.\n", i);
        cblas_thread_ids[i] = 0;
    }
    
    platform_mutex_unlock(&server_lock);

    // Note: We do not destroy statically initialized synchronization primitives
    // (queue_lock, kickoff_event, server_lock) as they are initialized with
    // PTHREAD_MUTEX_INITIALIZER and PTHREAD_COND_INITIALIZER.
    // Destroying them can cause undefined behavior on some platforms,
    // particularly GitHub Actions runners. They will be cleaned up automatically
    // when the program exits.

    // cleanup stats resources
    cblas_cleanup_stats();
}

//------------------------------------------------------
// thread server worker thread
//------------------------------------------------------
#ifdef _WIN32
static DWORD WINAPI cblas_worker_thread(void *pvoid)
#else
static void *cblas_worker_thread(void *pvoid)
#endif
{
    work_queue_t* work_item;

    int thread_num = (int)(intptr_t)pvoid;

    MT_TRACE_THREAD(thread_num, "created.\n");

    while(1)
    {
        MT_TRACE_THREAD(thread_num, "waits.\n");

        // the lock is released if/when this thread sleeps on the condition variable
        platform_mutex_lock(&queue_lock);
        
        while (!work_queue && thread_num <= cblas_max_threads - 2)
            platform_cond_wait(&kickoff_event, &queue_lock);
        
        MT_TRACE_THREAD(thread_num, "is awake.\n");

        if (thread_num > cblas_max_threads - 2)
        {
            MT_TRACE_THREAD(thread_num, "exiting.\n");
            platform_mutex_unlock(&queue_lock);

            // excess thread, so worker thread exits
            break;
        }

        work_item = work_queue;
        if (work_item)
            work_queue = work_queue->next;

        // release the queue lock acquired via cond_wait
        platform_mutex_unlock(&queue_lock);

        // if no work, reset event and then go to sleep to wait for more work
        if (!work_item)
        {
            MT_TRACE_THREAD(thread_num, "no work, trying again.\n");
            platform_yield();
            continue;
        }

        work_item->thread_num = thread_num;

        MT_TRACE_THREAD(thread_num, "executing a task.\n");

#ifdef MT_DEBUG
        // Track timing for this work item
        work_item->start_time_us = mt_get_time_us();
#endif

        // execute the task
        work_item->kernel(work_item->args);

        assert(work_item->finished == 0);
        
        work_item->finished = 1;
        // MB;

#ifdef MT_DEBUG
        // Calculate and log execution time
        work_item->end_time_us = mt_get_time_us();
        double duration = work_item->end_time_us - work_item->start_time_us;
        const char* op = work_item->operation ? work_item->operation : "task";
        MT_TRACE_TIMING(thread_num, op, duration);
#endif

        MT_TRACE_THREAD(thread_num, "task completed.\n");
    }

#ifdef _WIN32
    return 0;
#else
    return NULL;
#endif
}

//------------------------------------------------------
// execute a work queue synchronously
//------------------------------------------------------
void cblas_execute(CBLAS_INDEX items, work_queue_t* queue)
{
    assert(items > 0 && queue);

    if (items <= 0 || queue == NULL)
        return;

    // submit task queue
    if (items > 1 && queue->next)
        cblas_execute_async(items - 1, queue->next);

    // execute the first task on the main thread
    queue->kernel(queue->args);
    
    queue->finished = 1;
    // MB;

    // wait for the queue of work to finish
    if (items > 1 && queue->next)
        cblas_execute_async_join(items - 1, queue->next);
}

//------------------------------------------------------
// execute a work queue asynchronously
//------------------------------------------------------
void cblas_execute_async(CBLAS_INDEX items, work_queue_t* queue)
{
   (void)items;
   assert(queue);

    MT_TRACE("adding %zu items to the queue.\n", items);

    // add new work to the end of the work_queue
    platform_mutex_lock(&queue_lock);

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

    platform_mutex_unlock(&queue_lock);

    // wake up the worker threads
    MT_TRACE("waking worker threads.\n");

    platform_cond_broadcast(&kickoff_event);
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
            platform_yield();

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

    // assert(work_queue == NULL);

    // TODO - if work was added to the queue after this batch we can't sleep the worker threads
    // by resetting the event
    // pthread_mutex_lock(&queue_lock);

    //     if (work_queue == NULL)
    //         ResetEvent(kickoff_event);

    // pthread_mutex_unlock(&queue_lock);
}
