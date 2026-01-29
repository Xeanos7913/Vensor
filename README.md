# Vensor
A C++ machine learning library utilizing Vulkan for GPU acceleration. Developed to explore modern GPU computing, low-level graphics APIs, and high-performance machine learning implementations.

## Features
- Sequential model architecture supporting various layer types
- GPU-accelerated tensor operations
- MNIST implementation example included
- Basic graphics engine (VkCalcium.hpp) included
- Basic shader writing DSL (Livia.hpp) included

## Implemented Components

### Layers
- Neural Network Layers: Linear, Conv2d, Conv2d Transposed, FlashAttention (Causal self-attention)
- Activation Functions: ReLU, Softmax, TanH
- Normalization: BatchNorm1d/2d, LayerNorm
- Loss Functions: MSE, Cross-Entropy, KL-Divergence (for VAEs only)
- Utility Layers: EmbeddingLookup, ResidualConnect, MaxPooling, Bilinear interpolation Upsampling

### Optimizers
- Stochastic Gradient Descent (SGD) [there's a typo in my code, it's actually "SDG" instead of "SGD" xp ]
- AdamW

### Core Operations
- Matrix multiplication with backward pass
- Tensor addition (in-place and standard)
- Statistical operations (mean, random initialization)
- Tensor comparison utilities

## Dependencies
- Vulkan SDK
- volk
- stb_image & stb_image_write
- VKBootstrap
- GLM
- GLFW (optional, for VkCalcium)
- glslang compiler

## Livia
- Livia.hpp contains a work in progress implementation of a shader generator. You can use Livia to write compute shaders in glsl from c++. I created it because I was inspired by Triton. It's a Domain Specific Language (is it? I don't know what it is but it generates usable shaders) for writing compute shaders in glsl.
- I do eventually plan to compile Livia's intermediate representation directly into SPIR-V in the future.
- I will then rewrite all the shaders in this repo using Livia.
- Why the name Livia? Uhh

## Notes
The project is under active development. Current implementation focuses on core functionality and shader kernel implementations. I'd not recommend actually using this library for anything serious right now. Treat it like learning material.