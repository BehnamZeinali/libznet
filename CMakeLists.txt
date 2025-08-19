cmake_minimum_required(VERSION 3.10)
project(znet)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable debug flags manually
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Include directories
include_directories(include)

# Source files
set(SOURCES
    src/autograd/tensor.cpp
    src/autograd/autograd_function.cpp
    src/autograd/ops_impl.cpp
    src/autograd/kernels.cpp 
    src/autograd/grad_mode.cpp
    src/autograd/tensor_iter.cpp
    src/autograd/shape_ops.cpp
    
    
    src/nn/module.cpp 
    src/nn/linear_impl.cpp 
    src/nn/loss_impl.cpp 
    src/nn/relu_impl.cpp 
    
    src/optim/sgd.cpp

    main_dnn_4d.cpp
)

# Build the final executable
add_executable(znet_dnn_4d ${SOURCES})
