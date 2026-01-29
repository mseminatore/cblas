# CMake generated Testfile for 
# Source directory: /home/runner/work/cblas/cblas
# Build directory: /home/runner/work/cblas/cblas/build_tsan
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(blas_test "/home/runner/work/cblas/cblas/build_tsan/blas_test")
set_tests_properties(blas_test PROPERTIES  _BACKTRACE_TRIPLES "/home/runner/work/cblas/cblas/CMakeLists.txt;21;add_test;/home/runner/work/cblas/cblas/CMakeLists.txt;0;")
add_test(test_strided "/home/runner/work/cblas/cblas/build_tsan/test_strided")
set_tests_properties(test_strided PROPERTIES  _BACKTRACE_TRIPLES "/home/runner/work/cblas/cblas/CMakeLists.txt;22;add_test;/home/runner/work/cblas/cblas/CMakeLists.txt;0;")
add_test(test_threshold "/home/runner/work/cblas/cblas/build_tsan/test_threshold")
set_tests_properties(test_threshold PROPERTIES  _BACKTRACE_TRIPLES "/home/runner/work/cblas/cblas/CMakeLists.txt;23;add_test;/home/runner/work/cblas/cblas/CMakeLists.txt;0;")
add_test(test_concurrent "/home/runner/work/cblas/cblas/build_tsan/test_concurrent")
set_tests_properties(test_concurrent PROPERTIES  _BACKTRACE_TRIPLES "/home/runner/work/cblas/cblas/CMakeLists.txt;24;add_test;/home/runner/work/cblas/cblas/CMakeLists.txt;0;")
