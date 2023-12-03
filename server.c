//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------
#include <stdio.h>
#include <pthread.h>
#include "cblas.h"

extern int cblas_max_threads;

static work_queue_t *work_queue = NULL;
static pthread_t cblas_thread_ids[MAX_THREADS];

static void *cblas_worker_thread(void* pvoid);
static pthread_mutex_t queue_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t kickoff_event = PTHREAD_COND_INITIALIZER;
static pthread_mutex_t event_lock = PTHREAD_MUTEX_INITIALIZER;

//------------------------------------------------------
// initialize the thread server system
//------------------------------------------------------
void cblas_init_server()
{
    pthread_mutex_init(&queue_lock, NULL);
    pthread_mutex_init(&event_lock, NULL);
    pthread_cond_init(&kickoff_event, NULL);

    // create the worker threads
    for (int i = 0; i < cblas_max_threads - 1; i++)
    {
        pthread_create(&cblas_thread_ids[i], NULL, cblas_worker_thread, (void*)i);
    }
}

//------------------------------------------------------
// shutdown the thread server
//------------------------------------------------------
void cblas_shutdown()
{
    pthread_mutex_destroy(&queue_lock);
    pthread_mutex_destroy(&event_lock);
    pthread_cond_destroy(&kickoff_event);

    // TODO - delete threads?
}

//------------------------------------------------------
//
//------------------------------------------------------
static void *cblas_worker_thread(void *pvoid)
{
    work_queue_t* work_item;

    int thread_num = (int)pvoid;

    MT_TRACE("thread [%d] created.\n", thread_num);

    while(1)
    {
        MT_TRACE("thread [%d] waits.\n", thread_num);

        pthread_mutex_lock(&event_lock);
        
        while (!work_queue)
            pthread_cond_wait(&kickoff_event, &event_lock);
        
        pthread_mutex_unlock(&event_lock);

        MT_TRACE("thread [%d] woke up.\n", thread_num);

        while(1)
        {
            pthread_mutex_lock(&queue_lock);

            work_item = work_queue;
            if (work_item)
                work_queue = work_queue->next;

            pthread_mutex_unlock(&queue_lock);

            // if no work, reset event and then go to sleep to wait for more work
            if (!work_item)
            {
                MT_TRACE("thread [%d] no work, trying again.\n", thread_num);
                break;
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

        work_queue_t* next_item = work_queue;

        // find the end of the work queue
        while (next_item)
            next_item = next_item->next;

        // add new work to the end
        next_item = queue;
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
