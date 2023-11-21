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

# link with correct thread server
if ( WIN32 )
    set(SERVER_FILE "server_win32.c")
else()
    set(SERVER_FILE "server.c")
endif()

# add the library
add_library(cblas STATIC copy.c axpy.c dot.c swap.c scal.c axpby.c rot.c asum.c nrm2.c ger.c gemv.c rotg.c util.c ${SERVER_FILE})

#
# add the executables
#
add_executable(blas_test test.c)
target_link_libraries(blas_test cblas ${MATH_LIB})

add_executable(blas_stress test_stress.c)
target_link_libraries(blas_stress cblas ${MATH_LIB})
