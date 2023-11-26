cmake_minimum_required(VERSION 3.10)

# set the project name and version
project(cblas VERSION 1.0 LANGUAGES C)

# testing
enable_testing()
add_test(NAME blas_test COMMAND blas_test)

# add the includes
include_directories(${PROJECT_SOURCE_DIR})

# link math lib for gcc/linux
set(MATH_LIB "")
if ( LINUX )
    message("Linux build: add -lm to linker options")
    set(MATH_LIB "m")
endif()

# if ("${CMAKE_C_COMPILER_ID}" STREQUAL "Clang")
#     set_property(TARGET foo APPEND PROPERTY COMPILE_OPTIONS -g -O2)
# elseif ("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
#     set_property(TARGET foo APPEND PROPERTY COMPILE_OPTIONS -g -O2)
# elseif ("${CMAKE_C_COMPILER_ID}" STREQUAL "MSVC")
#     set_property(TARGET foo APPEND PROPERTY COMPILE_OPTIONS /W3)
# endif()

# link with correct thread server
if ( WIN32 )
    set(SERVER_FILE "server_win32.c")
else()
    set(SERVER_FILE "server.c")
endif()

# link with correct cpuid code
if ( CMAKE_HOST_SYSTEM_PROCESSOR MATCHES arm64)
    set(CPUID_FILE "cpuid_arm64.c")
else()
    set(CPUID_FILE "cpuid_x64.c")
endif()

message("SERVER_FILE: " ${SERVER_FILE})
message("CPUID_FILE: " ${CPUID_FILE})

# add the library
add_library(cblas STATIC copy.c axpy.c dot.c swap.c scal.c axpby.c rot.c asum.c nrm2.c ger.c gemv.c rotg.c util.c ${SERVER_FILE} ${CPUID_FILE})

#
# add the executables
#
add_executable(blas_test test_main.c test.c)
target_link_libraries(blas_test cblas ${MATH_LIB})

add_executable(blas_stress test_main.c test_stress.c)
target_link_libraries(blas_stress cblas ${MATH_LIB})

add_executable(blas_perf blas_perf.c)
target_link_libraries(blas_perf cblas ${MATH_LIB})
