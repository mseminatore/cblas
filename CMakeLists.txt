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

message("Compiler = ${CMAKE_C_COMPILER_ID}")

if ("${CMAKE_C_COMPILER_ID}" STREQUAL "Clang")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O2")    
elseif ("${CMAKE_C_COMPILER_ID}" STREQUAL "AppleClang")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O2")    
elseif ("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O2")
elseif ("${CMAKE_C_COMPILER_ID}" STREQUAL "MSVC")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /arch:AVX2 /std:c11")
#    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /STACK:0x800000")
endif()

message("CFLAGS = ${CMAKE_C_FLAGS}")

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
add_library(cblas STATIC copy.c axpy.c dot.c swap.c scal.c axpby.c rot.c asum.c nrm2.c ger.c gemv.c gemm.c rotg.c util.c setv.c ${SERVER_FILE} ${CPUID_FILE})
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

add_executable(ger_perf ger_perf.c)
target_link_libraries(ger_perf cblas ${MATH_LIB})

add_executable(mem_perf mem_perf.c)
target_link_libraries(mem_perf cblas ${MATH_LIB})

# install targets
install(TARGETS cblas DESTINATION lib)
install(FILES cblas.h DESTINATION include)

# cmake --build . --config Release
# cmake --install . --prefix c:\opt -v
