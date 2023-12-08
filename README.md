# CBLAS - small, fast, portable subset of the standard BLAS libraries 

[![CMake](https://github.com/mseminatore/cblas/actions/workflows/cmake-single-platform.yml/badge.svg)](https://github.com/mseminatore/cblas/actions/workflows/cmake-single-platform.yml)

# What is CBLAS?

CBLAS is an experimental implementation of a subset of the full BLAS (Basic 
Linear Algebra Subprograms) library standard. You can find the documentation
[here](https://www.netlib.org/blas/).

The library is currently built and tested on Windows (MSVC and Clang), MacOS,
and Ubuntu Linux. The library supports multi-threading for performance, however
not all functions currently support threading. If advanced SIMD instructions
(AVX, AVX2, NEON, FMA) are available they will be used.

This project started as a basic implementation of the BLAS
routines needed for my [libann](https://www.github.com/mseminatore/ann) neural
networking library. Curiosity about performance differences evolved the project
into an exploration of deep optimization for modern CPU architectures.

If you are curious to learn more about how BLAS-like libraries can be optimized
I highly recommend [this](https://github.com/flame/how-to-optimize-gemm/wiki#the-gotoblasblis-approach-to-optimizing-matrix-matrix-multiplication---step-by-step) 
tutorial from the authors of GotoBLAS/BLIS/Flame.

# What CBLAS is not

This library is not intended to be complete. There are many portions of the
BLAS standard that are intentionally left as unimplemented. For example,
there is no complex number support. Nor does the library intend to compete
with commercial offerings like [OpenBLAS](https://www.openblas.net), 
[Intel MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html),
or the [AMD Optimizing CPU Libraries](https://www.amd.com/en/developer/aocl.html).

> If you are using the library and would like to request additional BLAS
> function support please open an issue.

# Which BLAS functions are supported

The following BLAS library functions are currently supported by the library.

## Level 1 BLAS functions: vector-vector ops

The BLAS standard defines function prefixes to
distinguish between variations of the same function. The prefix *s*
denotes single-precision and *d* denotes double-precision. An *x* is used as a
placeholder for *s* or *d* variants in the tables below.

All library functions are prefixed with *cblas_* so, for example, the
function for a single-precision vector-vector copy would be *cblas_scopy()*.

Function | Description
-------- | -----------
xrotg | generate a plane rotation
xrot | apply plane rotation
xswap | swap two vectors x and y
xcopy | copy one vector to another
xaxpy | compute y = a * *x* + *y*
xaxpby | compute y = a * *x* + b * *y*
xdot | compute dot product of two vectors
xnrm2 | euclidean norm of a vector
xasum | sum of absolute value of vector elements
ixamax | index of max absolute value of a vector

## Level 2 BLAS functions: matrix-vector ops

Function | Description
-------- | -----------
ger | rank 1 update A = a * x * y' + A
gemv | matrix-vector multiply

## Level 3 BLAS functions: matrix-matrix ops

function | Description
-------- | -----------
xgemm | general matrix multiply


