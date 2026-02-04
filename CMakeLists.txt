cmake_minimum_required(VERSION 3.10)

# set the project name and version
project(cblas VERSION 1.0 LANGUAGES C)

# Configuration options
option(CBLAS_ENABLE_MT "Enable multi-threading" ON)
option(CBLAS_USE_SIMD "Enable SIMD optimizations" ON)
option(CBLAS_CHECK_INPUTS "Enable input validation" ON)
option(CBLAS_USE_STATIC_BUFFERS "Use static buffers" ON)
set(CBLAS_MAX_THREADS "64" CACHE STRING "Maximum threads")

# Generate cblas_config.h from cblas_config.h.in
configure_file(cblas_config.h.in cblas_config.h @ONLY)

# Add the binary directory to include path so generated config header can be found
include_directories(${PROJECT_BINARY_DIR})

# testing
enable_testing()
add_test(NAME blas_test COMMAND blas_test)
add_test(NAME test_strided COMMAND test_strided)
add_test(NAME test_threshold COMMAND test_threshold)
add_test(NAME test_dot_threshold COMMAND test_dot_threshold)
add_test(NAME test_concurrent COMMAND test_concurrent)
add_test(NAME test_level2_mt COMMAND test_level2_mt)
add_test(NAME test_stats COMMAND test_stats)
add_test(NAME test_mt_debug COMMAND test_mt_debug)
add_test(NAME test_autotune COMMAND test_autotune)

if ( WIN32 )
# The disabled test causes issues on Windows CI due to timeouts
# The test can be run manually if needed
# The test can take more than 2 minutes to complete
#    add_test(NAME blas_stress COMMAND blas_stress)
#    add_test(NAME win_test COMMAND win_test)
#    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

# add the includes
include_directories(${PROJECT_SOURCE_DIR})

# link math lib for gcc/linux
set(MATH_LIB "")
if ( LINUX )
    message("Linux build: add -lm to linker options")
    set(MATH_LIB "m")
endif()

message("Compiler = ${CMAKE_C_COMPILER_ID}")

if ("${CMAKE_C_COMPILER_ID}" STREQUAL "Clang" OR
    "${CMAKE_C_COMPILER_ID}" STREQUAL "AppleClang" OR
    "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O2 -Wall -Wextra -Wpedantic -Werror")
elseif ("${CMAKE_C_COMPILER_ID}" STREQUAL "MSVC")
    # Convert GCC-style flags to MSVC equivalents if they exist in CMAKE_C_FLAGS
    string(REPLACE "-Werror" "/WX" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    # Note: /WX (warnings as errors) is not used due to pre-existing warnings in the codebase
    # (C4996: deprecated functions, C4127: constant conditionals, C4702: unreachable code,
    #  C4312/C4311: pointer/int type casting in server_win32.c)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W4 /std:c11 /experimental:c11atomics")
#    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /STACK:0x800000")
endif()

# Add Intel specific compiler flags for x86_64 architecture
if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64")
    if ("${CMAKE_C_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_C_COMPILER_ID}" STREQUAL "AppleClang" OR "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx2 -mfma")
        add_compile_definitions(USE_INTEL_FMA)
    elseif ("${CMAKE_C_COMPILER_ID}" STREQUAL "MSVC")
        # MSVC on x86_64: /arch:AVX2 enables AVX2 and FMA
        # Note: MSVC doesn't define __AVX2__ automatically, so we define it ourselves
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /arch:AVX2")
        add_compile_definitions(USE_INTEL_FMA)
        add_compile_definitions(__AVX2__)
    endif()
endif()

message("CFLAGS = ${CMAKE_C_FLAGS}")

# link with correct thread server
if ( WIN32 )
    set(SERVER_FILE "server_win32.c")
else()
    set(SERVER_FILE "server.c")
endif()

# link with correct cpuid code
if ( CMAKE_HOST_SYSTEM_PROCESSOR MATCHES arm64 OR CMAKE_HOST_SYSTEM_PROCESSOR MATCHES aarch64)
    set(CPUID_FILE "cpuid_arm64.c")
else()
    set(CPUID_FILE "cpuid_x64.c")
endif()

message("SERVER_FILE: " ${SERVER_FILE})
message("CPUID_FILE: " ${CPUID_FILE})

# add the library
add_library(cblas STATIC copy.c axpy.c dot.c swap.c scal.c axpby.c rot.c asum.c nrm2.c ger.c gemv.c gemm.c rotg.c util.c setv.c kernels/dot_k.c 
    kernels/dot_k_neon.c kernels/dot_k_sse.c kernels/dot_k_avx.c kernels/dot_k_fma.c 
    kernels/asum_k_sse.c kernels/asum_k_neon.c
    kernels/copy_k_avx.c kernels/copy_k_neon.c
    ${CPUID_FILE} 
    ${SERVER_FILE}
)
#target_sources(cblas PUBLIC FILE_SET HEADERS FILES cblas.h)

#
# add the executables
#
add_executable(blas_test test_main.c test.c)
target_link_libraries(blas_test cblas ${MATH_LIB})

add_executable(blas_stress test_main.c test_stress.c)
target_link_libraries(blas_stress cblas ${MATH_LIB})

add_executable(gemm_perf gemm_perf.c)
target_link_libraries(gemm_perf cblas ${MATH_LIB})

add_executable(dgemm_perf dgemm_perf.c)
target_link_libraries(dgemm_perf cblas ${MATH_LIB})

add_executable(ger_perf ger_perf.c)
target_link_libraries(ger_perf cblas ${MATH_LIB})

add_executable(dger_perf dger_perf.c)
target_link_libraries(dger_perf cblas ${MATH_LIB})

add_executable(gemv_perf gemv_perf.c)
target_link_libraries(gemv_perf cblas ${MATH_LIB})

add_executable(dot_perf dot_perf.c)
target_link_libraries(dot_perf cblas ${MATH_LIB})

add_executable(axpy_perf axpy_perf.c)
target_link_libraries(axpy_perf cblas ${MATH_LIB})

add_executable(copy_perf copy_perf.c)
target_link_libraries(copy_perf cblas ${MATH_LIB})

add_executable(dot_threshold_tuning dot_threshold_tuning.c)
target_link_libraries(dot_threshold_tuning cblas ${MATH_LIB})

add_executable(nrm2_asum_rot_perf nrm2_asum_rot_perf.c)
target_link_libraries(nrm2_asum_rot_perf cblas ${MATH_LIB})

add_executable(mem_perf mem_perf.c)
target_link_libraries(mem_perf cblas ${MATH_LIB})

add_executable(win_test test_main.c win_test.c)
target_link_libraries(win_test cblas ${MATH_LIB})

add_executable(test_strided test_main.c test_strided.c)
target_link_libraries(test_strided cblas ${MATH_LIB})

add_executable(test_stats test_stats.c)
target_link_libraries(test_stats cblas ${MATH_LIB})

add_executable(test_threshold test_threshold.c)
target_link_libraries(test_threshold cblas ${MATH_LIB})

add_executable(test_dot_threshold test_dot_threshold.c)
target_link_libraries(test_dot_threshold cblas ${MATH_LIB})

find_package(Threads)

add_executable(test_concurrent test_main.c test_concurrent.c)
if(Threads_FOUND)
    target_link_libraries(test_concurrent cblas ${MATH_LIB} Threads::Threads)
else()
    target_link_libraries(test_concurrent cblas ${MATH_LIB} pthread)
endif()

add_executable(test_mt_debug test_mt_debug.c)
target_link_libraries(test_mt_debug cblas ${MATH_LIB})

add_executable(test_level2_mt test_main.c test_level2_mt.c)
target_link_libraries(test_level2_mt cblas ${MATH_LIB})

add_executable(test_autotune test_main.c test_autotune.c)
target_link_libraries(test_autotune cblas ${MATH_LIB})

# install targets
install(TARGETS cblas DESTINATION lib)
install(FILES cblas.h DESTINATION include)
install(FILES ${PROJECT_BINARY_DIR}/cblas_config.h DESTINATION include)

# cmake --build . --config Release
# cmake --install . --prefix c:\opt -v
