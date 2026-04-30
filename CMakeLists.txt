cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(transpose_test LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 61)
set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)

# Build gputt as a subdirectory; disable Python bindings.
set(GPUTT_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
add_subdirectory(gputt)

add_executable(transpose_test
    main.cu
    transpose.cpp
    transpose_gputt.cu
)
target_compile_features(transpose_test PRIVATE cxx_std_17)
target_link_libraries(transpose_test PRIVATE gputt cublas)
