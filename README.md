# Vensor
A simple C++ machine learning library, using Vulkan for GPU acceleration. I made this library to familiarise myself with many aspects of things, including just c++, working with low level graphics APIs like Vulkan and writing good GPU side code that runs fast. I started developing Vensor back near the end of july 2025. Procrastinated a lot, but I finally got to a point where it works to an okay extent.

## Features
Vensor works using Layers and Sequences of Layers. You load up a "Sequential" with various layers, one after the other, being through with their dimentions, and then you pass in the input tensor. An example of how to use Vensor to make neural networks is provided in the main.cpp file. Namely, a basic implementation of the MNIST handwritten digit recognision with Conv2d3x3, Batchnorm2d, Linear and Bacthnorm1d layers.

## What it lacks
Well, just about everything apart from the basic function. You can't even save a neural network that you train yet. There's also no residual connection system. No proper compute graph representation. I was too busy focusing on the shader kernel implementations and kind of just winged the higher level API of the engine.

## Future plans
To have the ability to save and load model weights would be nice. And also to load pytorch models would be cool to have. I'd also like to have a better compute graph representation and implement residual connections.

## Bonus
There's a tiny ~2k line graphics engine inside Vensor called VkCalcium.hpp. It's heavily flawed, and the whole file is commented out by default, but it does work.

## Dependencies
1. Standard Vulkan headers
2. stb_image
3. std_image_write
4. VKBootstrap
5. glm
6. glfw (for VkCalcum.hpp. You don't need this to just run Vensor's machine learning engine)
7. glslang compiler to compile shaders (usually the vulkan SDK has it by default)
