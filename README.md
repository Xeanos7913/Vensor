# Vensor
A simple C++ machine learning library, using Vulkan for GPU acceleration. I made this library to familiarise myself with many aspects of things, including just c++, working with low level graphics APIs like Vulkan and writing good GPU side code that runs fast. I started developing Vensor back near the end of july 2025. Procrastinated a lot, but I finally got to a point where it works to an okay extent.

## Features
Vensor works using Layers and Sequences of Layers. You load up a "Sequential" with various layers, one after the other, being through with their dimentions, and then you pass in the input tensor. An example of how to use Vensor to make neural networks is provided in the main.cpp file. Namely, a basic implementation of the MNIST handwritten digit recognision with Conv2d3x3, Batchnorm2d, Linear and Bacthnorm1d layers.

## What it lacks
Well, just about everything apart from the basic function. There's also no proper compute graph representation. I was too busy focusing on the shader kernel implementations and kind of just winged the higher level API of the engine.

## Future plans
To have the ability to save and load model weights from other sources would be nice. And also to load pytorch models would be cool to have. I'd also like to have a better compute graph representation.

## Bonus
There's a tiny ~2k line graphics engine inside Vensor called VkCalcium.hpp. It's heavily flawed (backface culling and horizontal mouse is reversed for linux and windows), and the whole file is commented out by default, but it does work.

## Dependencies
1. Standard Vulkan headers
2. stb_image (https://github.com/nothings/stb)
3. stb_image_write (https://github.com/nothings/stb)
4. VKBootstrap (https://github.com/charles-lunarg/vk-bootstrap)
5. glm (https://github.com/g-truc/glm)
6. glfw (for VkCalcum.hpp. You don't need this to just run Vensor's machine learning engine) (https://github.com/glfw/glfw)
7. glslang compiler to compile shaders (usually the vulkan SDK has it by default)

## Layers Implemented
1. Linear
2. ReLU
3. LinearReLU
4. BatchNorm1d
5. BatchNorm2d
6. LayerNorm1d (It's basically just LayerNorm that norms over the last dimention of the input tensor)
7. Conv2d3x3 (kernel size is fixed at 3x3 for this layer. I should probably make some kind of shader builder that calculates the constants needed for MxN conv2d)
8. MSEloss
9. SoftmaxCrossEntropy (softmax + cross_entropy loss merged together)
10. EmbeddingLookup (copy embedded vectors from embedding tensor using a one-hot vector as input)
11. Sequential
12. ResidualConnect (makes a residual connection from input Module's output tensor to the previous module's output tensor)