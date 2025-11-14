# Vensor
A C++ machine learning library utilizing Vulkan for GPU acceleration. Developed to explore modern GPU computing, low-level graphics APIs, and high-performance machine learning implementations.

## Features
- Sequential model architecture supporting various layer types
- GPU-accelerated tensor operations
- MNIST implementation example included
- Basic graphics engine (VkCalcium.hpp) included

## Implemented Components

### Layers
- Neural Network Layers: Linear, Conv2d (3x3 and general), Conv2d Transposed
- Activation Functions: ReLU, Softmax
- Normalization: BatchNorm1d/2d, LayerNorm
- Loss Functions: MSE, Cross-Entropy, KL-Divergence
- Utility Layers: EmbeddingLookup, ResidualConnect, MaxPooling

### Optimizers
- Stochastic Gradient Descent (SGD)

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

## Roadmap
- Model weight import/export functionality
- PyTorch model compatibility
- Enhanced compute graph representation
- Additional optimizers and layers
- Improved random number generation

## Notes
The project is under active development. Current implementation focuses on core functionality and shader kernel implementations.
