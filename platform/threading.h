//------------------------------------------------------
// platform/threading.h
//
// Platform-agnostic threading abstraction layer
// Abstracts pthread (Unix/Linux/macOS) and Win32 threading APIs
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#ifndef __PLATFORM_THREADING_H
#define __PLATFORM_THREADING_H

#ifdef _WIN32
    // Windows threading API
    #include <windows.h>
    
    // Thread handle type
    typedef HANDLE platform_thread_t;
    typedef DWORD platform_thread_id_t;
    
    // Mutex type
    typedef CRITICAL_SECTION platform_mutex_t;
    
    // Condition variable type (Windows Events)
    typedef HANDLE platform_cond_t;
    
    // Thread function return type
    typedef DWORD WINAPI platform_thread_func_t(void*);
    
    // Mutex operations
    #define PLATFORM_MUTEX_INITIALIZER {0}
    #define platform_mutex_init(m)      InitializeCriticalSection(m)
    #define platform_mutex_destroy(m)   DeleteCriticalSection(m)
    #define platform_mutex_lock(m)      EnterCriticalSection(m)
    #define platform_mutex_unlock(m)    LeaveCriticalSection(m)
    
    // Condition variable operations
    #define platform_cond_init(c)       (*(c) = CreateEvent(NULL, TRUE, FALSE, NULL))
    #define platform_cond_destroy(c)    CloseHandle(*(c))
    #define platform_cond_wait(c, m)    WaitForSingleObject(*(c), INFINITE)
    #define platform_cond_broadcast(c)  SetEvent(*(c))
    #define platform_cond_reset(c)      ResetEvent(*(c))
    
    // Thread operations
    #define platform_thread_create(t, f, a, id) \
        (*(t) = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)(f), (void*)(intptr_t)(a), 0, id))
    #define platform_thread_join(t)     WaitForSingleObject(t, INFINITE)
    #define platform_thread_close(t)    CloseHandle(t)
    
    // Atomic compare-and-swap for work queue
    #if defined(__GNUC__) && (__GNUC__ < 6)
    #   define platform_cas(dest, exch, comp) __sync_val_compare_and_swap(dest, comp, exch)
    #else
        #if defined(_WIN64)
        #   define platform_cas(dest, exch, comp) InterlockedCompareExchange64(dest, exch, comp)
        #else
        #   define platform_cas(dest, exch, comp) InterlockedCompareExchange(dest, exch, comp)
        #endif
    #endif
    
#else
    // POSIX threading API (pthread)
    #include <pthread.h>
    #include <sched.h>
    
    // Thread handle type
    typedef pthread_t platform_thread_t;
    typedef pthread_t platform_thread_id_t;
    
    // Mutex type
    typedef pthread_mutex_t platform_mutex_t;
    
    // Condition variable type
    typedef pthread_cond_t platform_cond_t;
    
    // Thread function return type
    typedef void* platform_thread_func_t(void*);
    
    // Mutex operations
    #define PLATFORM_MUTEX_INITIALIZER PTHREAD_MUTEX_INITIALIZER
    #define platform_mutex_init(m)      pthread_mutex_init(m, NULL)
    #define platform_mutex_destroy(m)   pthread_mutex_destroy(m)
    #define platform_mutex_lock(m)      pthread_mutex_lock(m)
    #define platform_mutex_unlock(m)    pthread_mutex_unlock(m)
    
    // Condition variable operations
    #define PLATFORM_COND_INITIALIZER   PTHREAD_COND_INITIALIZER
    #define platform_cond_init(c)       pthread_cond_init(c, NULL)
    #define platform_cond_destroy(c)    pthread_cond_destroy(c)
    #define platform_cond_wait(c, m)    pthread_cond_wait(c, m)
    #define platform_cond_broadcast(c)  pthread_cond_broadcast(c)
    #define platform_cond_reset(c)      ((void)0)  // No-op for pthread
    
    // Thread operations
    #define platform_thread_create(t, f, a, id) \
        pthread_create(t, NULL, f, (void*)(intptr_t)(a))
    #define platform_thread_join(t)     pthread_join(t, NULL)
    #define platform_thread_close(t)    ((void)0)  // No-op for pthread
    
    // Atomic compare-and-swap for work queue
    #define platform_cas(dest, exch, comp) __sync_val_compare_and_swap(dest, comp, exch)
    
    // Yield CPU
    #define platform_yield()            sched_yield()
    
#endif

// Windows doesn't have sched_yield in the same way
#ifdef _WIN32
    #define platform_yield()            Sleep(0)
#endif

#endif // __PLATFORM_THREADING_H
