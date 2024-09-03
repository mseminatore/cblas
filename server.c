//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------
#include <stdio.h>
#include <pthread.h>
#include "cblas.h"

extern volatile int cblas_max_threads;

static work_queue_t *work_queue = NULL;
static pthread_t cblas_thread_ids[MAX_THREADS];

static void *cblas_worker_thread(void* pvoid);

static pthread_mutex_t queue_lock   = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t server_lock   = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t kickoff_event = PTHREAD_COND_INITIALIZER;

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
        pthread_mutex_lock(&server_lock);

        int thread_count = cblas_max_threads;
        cblas_max_threads = threads;

        pthread_cond_broadcast(&kickoff_event);

        for (int i = threads - 1; i < thread_count - 1; i++)
        {
            MT_TRACE("set_num_threads: waiting on thread [%d] to quit.\n", i);

            pthread_join(cblas_thread_ids[i], NULL);

            MT_TRACE("set_num_threads: thread [%d] has quit.\n", i);
        }

        pthread_mutex_unlock(&server_lock);
    }

    // add more threads if needed
    if (cblas_is_server_alive() && threads > cblas_max_threads)
    {
        pthread_mutex_lock(&server_lock);

        int start = cblas_max_threads > 0 ? cblas_max_threads - 1 : 0;
        cblas_max_threads = threads;

        for (int i = start; i < threads - 1; i++)
        {
            pthread_create(&cblas_thread_ids[i], NULL, cblas_worker_thread, (void*)i);
        }

        pthread_mutex_unlock(&server_lock);
    }

    cblas_max_threads = threads;
}

//------------------------------------------------------
// initialize the thread server system
//------------------------------------------------------
int cblas_init_server()
{
    if (cblas_is_server_alive() || cblas_max_threads <= 1)
        return FALSE;

    // pthread_mutex_init(&queue_lock, NULL);
    // pthread_cond_init(&kickoff_event, NULL);

    pthread_mutex_lock(&server_lock);

    // create the worker threads
    for (int i = 0; i < cblas_max_threads - 1; i++)
    {
        pthread_create(&cblas_thread_ids[i], NULL, cblas_worker_thread, (void*)i);
    }

    cblas_set_server_alive(TRUE);

    pthread_mutex_unlock(&server_lock);

    return TRUE;
}

//------------------------------------------------------
// shutdown the thread server
//------------------------------------------------------
void cblas_shutdown()
{
    pthread_mutex_destroy(&queue_lock);
    pthread_cond_destroy(&kickoff_event);
    pthread_mutex_destroy(&server_lock);

    // TODO - delete threads?
}

//------------------------------------------------------
// thread server worker thread
//------------------------------------------------------
static void *cblas_worker_thread(void *pvoid)
{
    work_queue_t* work_item;

    int thread_num = (int)pvoid;

    MT_TRACE("thread [%d] created.\n", thread_num);

    while(1)
    {
        MT_TRACE("thread [%d] waits.\n", thread_num);

        // the lock is released if/when this thread sleeps on the condition variable
        pthread_mutex_lock(&queue_lock);
        
        while (!work_queue && thread_num <= cblas_max_threads - 2)
            pthread_cond_wait(&kickoff_event, &queue_lock);
        
        MT_TRACE("thread [%d] is awake.\n", thread_num);

        if (thread_num > cblas_max_threads - 2)
        {
            MT_TRACE("thread [%d] exiting.\n", thread_num);
            pthread_mutex_unlock(&queue_lock);

            // excess thread, so worker thread exits
            break;
        }

        work_item = work_queue;
        if (work_item)
            work_queue = work_queue->next;

        // release the queue lock acquired via cond_wait
        pthread_mutex_unlock(&queue_lock);

        // if no work, reset event and then go to sleep to wait for more work
        if (!work_item)
        {
            MT_TRACE("thread [%d] no work, trying again.\n", thread_num);
            sched_yield();
            continue;
        }

        work_item->thread_num = thread_num;

        MT_TRACE("thread [%d] executing a task.\n", thread_num);

        // execute the task
        work_item->kernel(work_item->args);

        assert(work_item->finished == 0);
        
        work_item->finished = 1;
        // MB;

        MT_TRACE("thread [%d] task completed.\n", thread_num);
    }

    return NULL;
}

//------------------------------------------------------
// execute a work queue synchronously
//------------------------------------------------------
void cblas_execute(int items, work_queue_t* queue)
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
void cblas_execute_async(int items, work_queue_t* queue)
{
   assert(queue);

    MT_TRACE("adding %d items to the queue.\n", items);

    // add new work to the end of the work_queue
    pthread_mutex_lock(&queue_lock);

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

    pthread_mutex_unlock(&queue_lock);

    // wake up the worker threads
    MT_TRACE("waking worker threads.\n");

    pthread_cond_broadcast(&kickoff_event);
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
            sched_yield();

        queue = queue->next;
        items--;
    }

    MT_TRACE("queued tasks finished.\n");

    // assert(work_queue == NULL);

    // TODO - if work was added to the queue after this batch we can't sleep the worker threads
    // by resetting the event
    // pthread_mutex_lock(&queue_lock);

    //     if (work_queue == NULL)
    //         ResetEvent(kickoff_event);

    // pthread_mutex_unlock(&queue_lock);
}
