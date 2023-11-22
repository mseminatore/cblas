//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------
#include <stdio.h>

#ifdef __APPLE__
#include <sys/sysctl.h>
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
    "Generic",
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

    return 1;
}