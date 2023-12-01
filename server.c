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

//------------------------------------------------------
// initialize the thread server system
//------------------------------------------------------
void cblas_init_server()
{
    pthread_mutex_init(&queue_lock, NULL);


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

        pthread_cond_wait(&kickoff_event, &queue_lock);

        MT_TRACE("thread [%d] woke up.\n", thread_num);

        pthread_mutex_lock(&queue_lock);

        work_item = work_queue;
        if (work_item)
            work_queue = work_queue->next;

        pthread_mutex_unlock(&queue_lock);

        // if no work, reset event and then go to sleep to wait for more work
        if (!work_item)
        {
            MT_TRACE("thread [%d] no work, trying again.\n", thread_num);
            continue;
        }

        work_item->thread_num = thread_num;

        MT_TRACE("thread [%d] executing a task.\n", thread_num);

        // execute the task
        work_item->kernel(work_item->args);

        assert(work_item->finished == 0);
        
        work_item->finished = 1;

        MT_TRACE("thread [%d] task completed.\n", thread_num);
    }

    return NULL;
}


//------------------------------------------------------
//
//------------------------------------------------------
void cblas_execute(int items, work_queue_t* queue)
{

}

//------------------------------------------------------
//
//------------------------------------------------------
void cblas_execute_async(int items, work_queue_t* queue)
{

}

//------------------------------------------------------
//
//------------------------------------------------------
void cblas_execute_async_wait(int items, work_queue_t* queue)
{

}
