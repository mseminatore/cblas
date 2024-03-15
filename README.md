# CBLAS - small, fast, portable subset of the standard BLAS libraries 

[![CMake](https://github.com/mseminatore/cblas/actions/workflows/cmake-single-platform.yml/badge.svg)](https://github.com/mseminatore/cblas/actions/workflows/cmake-single-platform.yml)

# What is CBLAS?

CBLAS is an experimental implementation of a subset of the full BLAS (Basic 
Linear Algebra Subprograms) library standard. You can find the documentation
and reference implementations [here](https://www.netlib.org/blas/).

The library is built and tested on a wide range of platform and OS 
combinations including Windows (MSVC and Clang), MacOS (Clang), 
Ubuntu Linux (gcc), and Raspbian OS (gcc). 

The library supports SIMD and multi-threading for performance. However, not 
all functions have been optimized to take advantage 
of these features. The primary focus for SIMD and Multi-threading work will be
on the level-3 functions, followed by level-2, etc.

> It is a not yet fully realized goal that, if advanced SIMD instructions for
> a particular platform are available (AVX, AVX2, NEON, FMA), that they will 
> be leveraged. This is a work in progress.

This project started as a basic implementation of the BLAS
routines required for my [libann](https://www.github.com/mseminatore/ann) neural
networking library. Curiosity about performance evolved the project
into an exploratory playground for deep optimization for modern CPU architectures.

If you are curious to learn more about how BLAS-like libraries can be optimized
I highly recommend 
[this](https://github.com/flame/how-to-optimize-gemm/wiki#the-gotoblasblis-approach-to-optimizing-matrix-matrix-multiplication---step-by-step) 
tutorial from the authors of GotoBLAS/BLIS/Flame.

# What CBLAS is not

This library is not intended to be a fully complete BLAS implementation. There 
are many portions of the BLAS standard that are intentionally left as 
unimplemented. For example, there is no complex number support. Nor does the 
library intend to compete with commercial offerings like [OpenBLAS](https://www.openblas.net), 
[Intel MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html),
or the [AMD Optimizing CPU Libraries](https://www.amd.com/en/developer/aocl.html).

> If you are using the CBLAS library and would like to request additional BLAS
> function support please open an 
> [issue](https://github.com/mseminatore/cblas/issues) or vote up an existing issue.

# Building CBLAS from source

If you do not already have a pre-built version of **libcblas** then you can
build from source using `make` as follow:

```
> git clone https://github.com/mseminatore/cblas
> cd cblas
> make
```

Or if you prefer to use `CMake`, then:

```
> git clone https://github.com/mseminatore/cblas
> cd cblas
> mkdir build
> cd build
> cmake ..
> cmake --build .
```

# Which BLAS functions are supported

The following BLAS library functions are currently supported by the library.

## Level 1 BLAS functions: vector-vector ops

The BLAS standard defines function prefixes to
distinguish between variations of the same function. The function prefix **s**
denotes the single-precision version and **d** denotes the double-precision
version. A library prefix of **cblas_** is used for the C implementation.

All library functions are prefixed with **cblas_** so, for example, the
function for a single-precision vector-vector copy would be **cblas_scopy()**.

The table below lists the single-precision version of the currently supported
BLAS functions.

Function | Description
-------- | -----------
cblas_sasum | sum of absolute value of vector elements
cblas_saxpy | computes **y** = alpha * **x** + **y**
cblas_saxpby | computes **y** = alpha * **x** + beta * **y**
cblas_scopy | copy one vector to another **y** = **x** 
cblas_sdot | computes dot product of two vectors **y** = **x** dot **y**
cblas_snrm2 | euclidean norm of a vector
cblas_srotg | generate a plane rotation
cblas_srot | apply plane rotation
cblas_ssetv | set vector elements to a value
cblas_sswap | swap two vectors **x** and **y**
cblas_isamax | index of max absolute value of a vector

## Level 2 BLAS functions: matrix-vector ops

Function | Description
-------- | -----------
cblas_sger | rank-1 update **A** = alpha * **x** * **y'** + **A**
cblas_sgemv | matrix-vector multiply **C** = alpha * **A** * **x** + beta * **y**

## Level 3 BLAS functions: matrix-matrix ops

function | Description
-------- | -----------
cblas_sgemm | general matrix multiply **C** = alpha * **A** * **B** + beta * **C**


