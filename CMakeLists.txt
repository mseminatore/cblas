cmake_minimum_required(VERSION 3.10)

# set the project name and version
project(cblas VERSION 1.0 LANGUAGES C)


# add the includes
include_directories(${PROJECT_SOURCE_DIR})

# link math lib for gcc/linux
set(MATH_LIB "")
if ( LINUX )
    message("Linux build: add -lm to linker options")
    set(MATH_LIB "m")
endif()

# add the library
add_library(cblas STATIC copy.c axpy.c dot.c swap.c scal.c axpby.c rot.c asum.c)

#
# add the executables
#
add_executable(test test.c)
target_link_libraries(test cblas ${MATH_LIB})