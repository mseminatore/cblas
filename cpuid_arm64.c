//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------
#include <stdio.h>
#include <stdint.h>
#include "cblas.h"

#if !defined(_WIN32)
#   include <unistd.h>
#endif

#ifdef __APPLE__
#   include <sys/sysctl.h>
//#include <mach/machine.h>
#endif

enum {
    CPU_GENERIC,
    CPU_VORTEX_M1,
    CPU_VORTEX_M2,
    CPU_VORTEX_M3
};

static const char *cpu_names[] =
{
    "Generic ARM Core",
    "Vortex-M1",
    "Vortex-M2",
    "Vortex-M3"
};

//------------------------------------------------------
// return the name of the current processor
//------------------------------------------------------
const char *cpu_get_core_name()
{
#ifdef __APPLE__
    uint32_t entry;
    size_t len = sizeof(entry);
    sysctlbyname("hw.cpufamily", &entry, &len, NULL, 0);

    // these values taken from <mach/machine.h>
    if (entry == 131287967 || entry == 458787763)
        return cpu_names[CPU_VORTEX_M1];
    else if (entry == 3660830781)
        return cpu_names[CPU_VORTEX_M2];
    else if (entry == 2271604202)
        return cpu_names[CPU_VORTEX_M3];
    else
        return cpu_names[CPU_GENERIC];

#endif
    return cpu_names[CPU_GENERIC];
}

//------------------------------------------------------
// return any ISA features of this CPU
//------------------------------------------------------
unsigned int cpu_get_features()
{
	unsigned int features = CPU_NONE;

#if defined(__APPLE__) || defined(__aarch64__)
    features |= CPU_NEON;
    features |= CPU_NEON_FMA;
#endif

    return features;
}

//------------------------------------------------------
// return the number of CPU cores available
//------------------------------------------------------
int cpu_get_core_count()
{    
#ifdef __APPLE__
    uint32_t entry;
    size_t len = sizeof(entry);

    // TODO - per sysctl.h this might want to be hw.ncpu?
    sysctlbyname("hw.physicalcpu_max", &entry, &len, NULL, 0);
    return entry;
#endif

#ifndef _WIN32
    int cores = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (cores == -1)
        cores = 1;

    return cores;
#endif

    // catch all
    puts("warning: cpu_get_core_count() not implemented!");
    return 1;
}

//------------------------------------------------------
// return the CPU L1 cache line size
//------------------------------------------------------
int cpu_get_cacheline_size()
{
#ifdef __APPLE__
    uint32_t entry;
    size_t len = sizeof(entry);

    sysctlbyname("hw.cachelinesize", &entry, &len, NULL, 0);
    return entry;
#endif

    return 32;
}

//------------------------------------------------------
// return the L2$ size
//------------------------------------------------------
int cpu_get_l2_cache_size()
{
    uint32_t l2_cache_size = 0;

    return l2_cache_size;
}
