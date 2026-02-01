# CBLAS - small, fast, portable subset of the standard BLAS libraries 

[![CMake](https://github.com/mseminatore/cblas/actions/workflows/cmake-single-platform.yml/badge.svg)](https://github.com/mseminatore/cblas/actions/workflows/cmake-single-platform.yml)
![GitHub License](https://img.shields.io/github/license/mseminatore/cblas)

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

> It is a not yet fully realized goal that advanced SIMD instructions 
> (AVX, AVX2, NEON, FMA) will be leveraged if they are available for a given
> platform. SIMD support is a work in progress.

This project started as a basic implementation of the BLAS
routines that were required for my [libann](https://www.github.com/mseminatore/ann) neural
networking library. Curiosity about maximizing performance evolved the project
into an exploratory playground for deep optimization on modern CPU architectures.

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

## CMake Configuration Options

CBLAS provides several CMake options to customize the build:

Option | Default | Description
------ | ------- | -----------
`CBLAS_ENABLE_MT` | ON | Enable multi-threading support
`CBLAS_USE_SIMD` | ON | Enable SIMD optimizations (SSE/AVX)
`CBLAS_CHECK_INPUTS` | ON | Enable input validation and error checking
`CBLAS_USE_STATIC_BUFFERS` | ON | Use static buffers instead of stack-based
`CBLAS_MAX_THREADS` | 64 | Maximum number of threads supported

### Configuration Examples

Build with multi-threading disabled:
```bash
cmake .. -DCBLAS_ENABLE_MT=OFF
```

Build with SIMD optimizations disabled:
```bash
cmake .. -DCBLAS_USE_SIMD=OFF
```

Build with custom maximum threads:
```bash
cmake .. -DCBLAS_MAX_THREADS=128
```

Build with input validation disabled (for maximum performance):
```bash
cmake .. -DCBLAS_CHECK_INPUTS=OFF
```

Combine multiple options:
```bash
cmake .. -DCBLAS_ENABLE_MT=ON -DCBLAS_MAX_THREADS=32 -DCBLAS_USE_SIMD=ON
```

## Makefile Configuration Options

The Makefile provides the same configuration options as CMake for consistency:

Option | Default | Description
------ | ------- | -----------
`CBLAS_ENABLE_MT` | 1 | Enable multi-threading support
`CBLAS_USE_SIMD` | 1 | Enable SIMD optimizations (SSE/AVX)
`CBLAS_CHECK_INPUTS` | 1 | Enable input validation and error checking
`CBLAS_USE_STATIC_BUFFERS` | 1 | Use static buffers instead of stack-based
`CBLAS_MAX_THREADS` | 64 | Maximum number of threads supported

### Configuration Examples

Build with multi-threading disabled:
```bash
make CBLAS_ENABLE_MT=0
```

Build with SIMD optimizations disabled:
```bash
make CBLAS_USE_SIMD=0
```

Build with custom maximum threads:
```bash
make CBLAS_MAX_THREADS=128
```

Build with input validation disabled (for maximum performance):
```bash
make CBLAS_CHECK_INPUTS=0
```

Combine multiple options:
```bash
make CBLAS_ENABLE_MT=1 CBLAS_MAX_THREADS=32 CBLAS_USE_SIMD=1
```

# Auto-Tuning Multi-Threading Thresholds

CBLAS includes an auto-tuning infrastructure that can optimize multi-threading thresholds based on your specific hardware configuration. By default, the library uses hardcoded thresholds that work well across a range of systems, but auto-tuning can provide 5-10% performance improvements by adapting to your CPU's characteristics.

## What is Auto-Tuning?

Auto-tuning runs micro-benchmarks at initialization to determine the optimal problem size at which multi-threading becomes beneficial. This crossover point depends on:
- Number of CPU cores
- Memory bandwidth
- Cache sizes
- Thread overhead on your system

## Enabling Auto-Tuning

Auto-tuning is controlled via the `CBLAS_AUTO_TUNE` environment variable:

```bash
# Enable auto-tuning
export CBLAS_AUTO_TUNE=1
./your_program

# Or for a single run
CBLAS_AUTO_TUNE=1 ./your_program
```

When enabled, you'll see output like:
```
CBLAS: Auto-tuning MT thresholds for 4 threads...
  Calibrating DOT threshold... 65536
  Calibrating COPY threshold... 32768
  Calibrating AXPY threshold... 65536
  GER/GEMV/GEMM thresholds (heuristic): 8192
CBLAS: Auto-tuning complete.
```

## Default vs Auto-Tuned Thresholds

The library includes the following default thresholds (in number of elements):

Operation | Default Threshold | Description
--------- | ----------------- | -----------
DOT | 32768 | Dot product (vector-vector)
AXPY | 32768 | Vector addition with scaling
COPY | 16384 | Vector copy
GER | 2048 | General rank-1 update (matrix)
GEMV | 4096 | Matrix-vector multiplication
GEMM | 4096 | Matrix-matrix multiplication

Auto-tuning will adjust these values based on your hardware. On systems with:
- **Few cores (1-2)**: Thresholds typically increase (more work needed to benefit from MT)
- **Many cores (8+)**: Thresholds typically decrease (MT beneficial earlier)
- **High memory bandwidth**: Thresholds for memory-bound operations (COPY, DOT) may increase
- **Low memory bandwidth**: Thresholds may decrease to better utilize multiple cores

## Programmatic Control

You can also control thresholds from your code:

```c
#include "cblas.h"

int main() {
    // Initialize with 4 threads
    cblas_init(4);
    
    // Option 1: Use default thresholds
    cblas_reset_thresholds();
    
    // Option 2: Auto-tune for this system
    cblas_autotune_thresholds();
    
    // Your BLAS operations...
    
    cblas_shutdown();
    return 0;
}
```

## Performance Considerations

- **Initialization Time**: Auto-tuning adds 2-5 seconds to initialization as it runs benchmarks
- **When to Use**: Best for long-running applications where initialization cost is amortized
- **When to Skip**: For short-lived programs, the default thresholds are likely sufficient
- **Thread Count**: Auto-tuning results are specific to the thread count used during calibration

## Viewing Current Thresholds

The current thresholds are always visible in the library configuration output.

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

# Documentation

- [Threading Architecture](docs/THREADING.md) - Comprehensive guide to CBLAS multi-threading system


