#pragma once  
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_LEFT_HANDED
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES

#ifdef DEBUG
#define DEBUG_PRINT(x) do { std::cout << x << "\n"; } while (0)
#else
#define DEBUG_PRINT(x)
#endif

#include <vector>  
#include <unordered_map>  
#include <initializer_list>  
#include <stdexcept>  
#include <iostream>  
#include <thread>  
#include <mutex>  
#include <memory>  
#include <random>
#include "VkMemAlloc.hpp"  
#include "ComputeStack.hpp"  
#include <glm/glm.hpp>
#include <numeric>

std::vector<char> readShaderBytecode(const std::string& filename) {
    DEBUG_PRINT("Making Shader: " << filename);
    std::ifstream file(filename, std::ios::ate | std::ios::binary);  // Open file at the end in binary mode
    
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open shader file: " + filename);
    }

    size_t fileSize = file.tellg();  // Get file size
    std::vector<char> buffer(fileSize);

    file.seekg(0);  // Go back to the beginning
    file.read(buffer.data(), fileSize);  // Read file into buffer
    file.close();
    return buffer;
}

enum class Operation {
    ADDITION,
    MULTIPLICATION,
    OTHER
};

static void broadcastShapes(const std::vector<uint32_t>& a,  
   const std::vector<uint32_t>& b,  
   Operation op = Operation::ADDITION) {  
   if (a.empty() || b.empty()) {  
       throw std::invalid_argument("Tensor shapes cannot be empty");  
   }

   if (op == Operation::ADDITION) {  
       // Check for broadcasting compatibility
       size_t max_dims = std::max(a.size(), b.size());
       for (size_t i = 0; i < max_dims; ++i) {
           uint32_t dim_a = (i < a.size()) ? a[a.size() - 1 - i] : 1;
           uint32_t dim_b = (i < b.size()) ? b[b.size() - 1 - i] : 1;
           if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
               throw std::invalid_argument("Shapes are not compatible for broadcasting in addition");
           }
       }
   } else if (op == Operation::MULTIPLICATION) {  
       // Check for matrix multiplication compatibility
       if (a.size() < 1 || b.size() < 1) {  
           throw std::invalid_argument("Shapes must have at least 1 dimension for matrix-vector or matrix multiplication");  
       }  

       // Check batch dimensions for rank 2 or rank 3 tensors
       if (a.size() >= 2 && b.size() >= 2) {
           size_t min_dims = std::min(a.size(), b.size());
           for (size_t i = 0; i < a.size() - min_dims; ++i) {  
               if (a[i] != b[i] && a[i] != 1 && b[i] != 1) {  
                   throw std::invalid_argument("Batch dimensions must match exactly or be 1 for matrix-vector or matrix multiplication");  
               }  
           }  
       }

       // Check matrix multiplication dimensions
       uint32_t a_k = a[a.size() - 1];  
       uint32_t b_k = b[b.size() - 2];  
       if (a_k != b_k) {  
           throw std::invalid_argument("Inner dimensions do not match for matrix-vector or matrix multiplication");  
       }  
   } else {
       return; // No checks for OTHER operations
   }  
}

// Helper function to print shapes for debugging
std::string shapeToString(const std::vector<uint32_t>& shape) {
    std::string result = "(";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) result += ", ";
        result += std::to_string(shape[i]);
    }
    result += ")";
    return result;
}
struct tensor_impl {
    VkDeviceAddress data;
    VkDeviceAddress grad;
    VkDeviceAddress strides;
    VkDeviceAddress shape;

    uint32_t num_elements;          // Total number of elements
    uint32_t num_dims;              // Number of dimensions
    uint32_t requires_gradient = 1;     // Boolean: needs gradient computation
    uint32_t is_leaf = 1;               // Boolean: leaf tensor in computation graph
};

struct tensor_push_const {
    VkDeviceAddress uniformAddress;
    glm::uvec3 grid_size;
	uint32_t mode = 0; // 0 = forward, 1 = backward
    uint64_t pad = 0;

    static uint32_t getPushConstantRange() {
        return sizeof(tensor_push_const);
	}
};

struct tensor_cmp_context{
    tensor_impl input_a;
    tensor_impl input_b;
    tensor_impl output_tensor;
};

struct matmul_context {
    tensor_impl input_tensor;
    tensor_impl weight_tensor;
    tensor_impl output_tensor;
    uint32_t mode;              // 0=forward, 1=backward
    uint32_t batch_size;
    uint32_t m, n, k;           // Matrix dimensions: A(m,k) @ B(k,n) = C(m,n)
    uint32_t accumulate_grad = 1;   // Whether to accumulate or overwrite gradients
    uint32_t use_bias = 0;
};

struct mean_context {
    tensor_impl input_tensor;
    tensor_impl output_tensor;
};

struct linear_context {
    tensor_impl input_tensor;
    tensor_impl weight_tensor;
    tensor_impl bias_tensor;        // Optional, can be null
    tensor_impl output_tensor;
    uint32_t mode;                  // 0=tiling, 1=no tiling
    uint32_t batch_size;
    uint32_t m, n, k;
	uint32_t accumulate_grad = 1;       // 0 = overwrite, 1 = accumulate
	uint32_t use_bias;              // 0 = no bias, 1 = use bias
};

struct matadd_context {
    tensor_impl input_a;
    tensor_impl input_b;
    tensor_impl output_tensor;
    uint32_t mode;
	uint32_t batch_size;
    uint32_t m, n;
	uint32_t accumulate_grad;   // Whether to accumulate or overwrite gradients
};

struct matadd_inplace_context {
    tensor_impl input_a;   // first input tensor (also output if mode = 0)
    tensor_impl input_b;   // second input tensor to add
    tensor_impl input_c;   // output tensor (used if mode = 1)
    uint mode;            // 0 = in-place (a = a + b), 1 = separate output (c = a + b)
    uint batch_size;
    uint m, n;
    uint accumulate_grad; // 0: overwrite, 1: += for grads
};

struct matmul_elementwise_context {
    tensor_impl input_a;   // first input tensor (also output if mode = 0)
    tensor_impl input_b;   // second input tensor to multiply
    tensor_impl input_c;   // output tensor (used if mode = 1)
    uint mode;            // 0 = in-place (a = a * b), 1 = separate output (c = a * b)
    uint batch_size;
    uint m, n;
    uint accumulate_grad; // 0: overwrite, 1: += for grads
};

struct logvar_to_std_context {
    tensor_impl logvar;        // input: log-variance tensor
    tensor_impl std_out;       // output: standard deviation tensor
    uint accumulate_grad;     // 0: overwrite, 1: += for grads
};

struct embedding_lookup_context {
    tensor_impl embedding_tensor;
    tensor_impl indices_tensor;
    tensor_impl output_tensor;
};

struct tensor_relu_context {
    tensor_impl input_tensor;
    tensor_impl output_tensor;
    uint32_t m, n;
    uint32_t mode;
};

struct tensor_softmax_context {
    tensor_impl input_tensor;
    tensor_impl output_tensor;
    uint32_t m, n;
};

struct tensor_cross_entropy_context {
    tensor_impl logits_tensor;
    tensor_impl target_tensor;
    tensor_impl loss_tensor;
	tensor_impl softmax_tensor; // Optional: store softmax output
    uint32_t m, n;
	uint32_t batch_size;
	uint32_t compute_softmax;   //whether to write softmax output into softmax_tensor
	uint32_t accumulate_grad = 1; // Whether to accumulate or overwrite gradients
};

// batchnorm 1D context
struct tensor_batchnorm_context {
    tensor_impl input_tensor;  // [B, M, N] - input feature vector
    tensor_impl weight_tensor; // [M] - weight (gamma)
    tensor_impl bias_tensor;   // [M] - bias (beta)
    tensor_impl running_mean;  // [M] - running mean (for inference)
    tensor_impl running_var;   // [M] - running variance (for inference)
    tensor_impl out_tensor;    // [B, M, N] - output vector
    tensor_impl save_mean;     // [M] - saved mean (for backward)
    tensor_impl save_var;      // [M] - saved variance (for backward)
    uint mode;                // 0 = train, 1 = eval
    uint batch_size;
    uint accumulate_grad;     // 0: overwrite, 1: += for grads
    float momentum;         // momentum for running stats
    float eps;              // epsilon for numerical stability
};

struct tensor_batchnorm2d_context {
    tensor_impl input_tensor;    // [B, C, H, W]
    tensor_impl weight_tensor;   // [C]
    tensor_impl bias_tensor;     // [C]
    tensor_impl running_mean;    // [C]
    tensor_impl running_var;     // [C]
    tensor_impl out_tensor;      // [B, C, H, W]
    tensor_impl save_mean;       // [C]
    tensor_impl save_var;        // [C]
    uint32_t mode;
    uint32_t batch_size;
    uint32_t channels;
    uint32_t height;
    uint32_t width;
    uint32_t accumulate_grad;
    float momentum;
    float eps;
};

struct tensor_layernorm1d_context {
    tensor_impl input_tensor;    // [B, M, N]
    tensor_impl weight_tensor;   // [N] - normalized shape
    tensor_impl bias_tensor;     // [N] - normalized shape
    tensor_impl out_tensor;      // [B, M, N]
    tensor_impl save_mean;       // [B, M] - mean for each sample
    tensor_impl save_rstd;       // [B, M] - reciprocal std for each sample
    uint mode;                   // 0 = train, 1 = eval
    uint normalized_size;       // N - size of normalized dimension
    uint batch_stride;          // M * N - elements per batch
    uint batch_size;
    uint accumulate_grad;
    float eps;
};

struct mse_loss_context {
    tensor_impl target_tensor;      // [B, ...]
    tensor_impl predicted_tensor;   // [B, ...]
    tensor_impl loss_tensor;        // [B, ...]
    uint32_t batch_size;
    uint32_t elements_per_batch;
};

struct kld_loss_context {
    tensor_impl mu_tensor;
    tensor_impl logvar_tensor;
    tensor_impl loss_tensor;
    uint batch_size;
    uint elements_per_batch;
    float beta;
};

struct tensor_conv2d_3x3_context {
    tensor_impl input_tensor;  // [N, C_in, H_in, W_in]
    tensor_impl weight_tensor; // [C_out, C_in, K_h, K_w]
    tensor_impl bias_tensor;   // [C_out]
    tensor_impl out_tensor;    // [N, C_out, H_out, W_out]
    uint stride_h;
    uint stride_w;
    uint pad_h;
    uint pad_w;
    uint dilation_h;
    uint dilation_w;
    uint groups;
    uint accumulate_grad;
    uint kernel_type;
};

struct tensor_conv2d_context {
    tensor_impl input_tensor;  // [N, C_in, H_in, W_in]
    tensor_impl weight_tensor; // [C_out, C_in, K_h, K_w]
    tensor_impl bias_tensor;   // [C_out]
    tensor_impl out_tensor;    // [N, C_out, H_out, W_out]
    uint stride_h;
    uint stride_w;
    uint pad_h;
    uint pad_w;
    uint dilation_h;
    uint dilation_w;
    uint kernel_h;            // Dynamic kernel height
    uint kernel_w;            // Dynamic kernel width
    uint groups;
    uint accumulate_grad;
    uint kernel_type;
};

struct tensor_transposed_conv2d_context {
    tensor_impl input_tensor;  // [N, C_in, H_in, W_in] - smaller input
    tensor_impl weight_tensor; // [C_in, C_out, K_h, K_w] - note: C_in first for transposed conv
    tensor_impl bias_tensor;   // [C_out]
    tensor_impl out_tensor;    // [N, C_out, H_out, W_out] - larger output
    uint stride_h;
    uint stride_w;
    uint pad_h;
    uint pad_w;
    uint dilation_h;
    uint dilation_w;
    uint kernel_h;            // Dynamic kernel height
    uint kernel_w;            // Dynamic kernel width
    uint output_pad_h;        // Output padding for transposed conv
    uint output_pad_w;
    uint groups;
    uint accumulate_grad;
    uint kernel_type;
};

struct tensor_max_pool_context {
    tensor_impl input_tensor;    // [B, C, H, W]
    tensor_impl output_tensor;   // [B, C, H_out, W_out]
    uint kernel_size_h;
    uint kernel_size_w;
    uint stride_h;
    uint stride_w;
};

struct tensor_ops_uniform_address {
    matmul_context context;
};

struct tensor_fill_uniform_address {
    tensor_impl tensor;
};

template<typename T>
struct tensor_fill_rand_uniform_address {
	tensor_impl tensor;
	uint32_t type; // 0 = gaussian, 1 = uniform
    uint32_t m, n;
	T min;
    T max;
    uint seed;
};

template<typename T>
struct tensor_fill_range_uniform_address {
	VkDeviceAddress tensor_data;
	VkDeviceAddress tensor_stride;
	T start;
	T step;
};

template<typename T>
struct TensorPool;

// Source - https://stackoverflow.com/a
// Posted by Davide Cannizzo, modified by community. See post 'Timeline' for change history
// Retrieved 2025-11-14, License - CC BY-SA 4.0
// Some actual voodoo magic
#define define const struct
#define coperator(ReturnType, OperatorName, FirstOperandType, SecondOperandType) OperatorName ## _ {} OperatorName; template <typename T> struct OperatorName ## Proxy{public:OperatorName ## Proxy(const T& t) : t_(t){}const T& t_;static ReturnType _ ## OperatorName ## _(const FirstOperandType a, const SecondOperandType b);};template <typename T> OperatorName ## Proxy<T> operator<(const T& lhs, const OperatorName ## _& rhs){return OperatorName ## Proxy<T>(lhs);}ReturnType operator>(const OperatorName ## Proxy<FirstOperandType>& lhs, const SecondOperandType& rhs){return OperatorName ## Proxy<FirstOperandType>::_ ## OperatorName ## _(lhs.t_, rhs);}template <typename T> inline ReturnType OperatorName ## Proxy<T>::_ ## OperatorName ## _(const FirstOperandType a, const SecondOperandType b)
// this monster allows me to have any custom operators, eg. the matmul operator '@'

template<typename T>  
struct Tensor {  

    // Pool
    TensorPool<T>* pool = nullptr;

   // cpu memory  
   std::vector<uint32_t> shape;  
   std::vector<uint32_t> strides;  
   std::vector<uint32_t> effectiveStrides;

   // gpu memory. We're using the StandaloneBuffer's persistent staging buffer as CPU backing memory.  
   std::unique_ptr<StandaloneBuffer<T>> dataBuffer;
   std::unique_ptr<StandaloneBuffer<T>> gradientBuffer;
   std::unique_ptr<StandaloneBuffer<uint32_t>> stridesBuffer;
   std::unique_ptr<StandaloneBuffer<uint32_t>> shapeBuffer;

   Allocator* allocator = nullptr;

   std::string name;

   std::function<void()> back;

    Tensor(){}; // empty default ctor

   Tensor(std::initializer_list<uint32_t> dims, Allocator* allocator)
       : shape(dims),
       allocator(allocator),
       dataBuffer(std::make_unique<StandaloneBuffer<T>>(1, allocator, VK_SHADER_STAGE_ALL)),
       gradientBuffer(std::make_unique<StandaloneBuffer<T>>(1, allocator, VK_SHADER_STAGE_ALL)),
       stridesBuffer(std::make_unique<StandaloneBuffer<uint32_t>>(1, allocator, VK_SHADER_STAGE_ALL)),
       shapeBuffer(std::make_unique<StandaloneBuffer<uint32_t>>(1, allocator, VK_SHADER_STAGE_ALL))
   {
       // Total size
       size_t total_size = 1;
       for (auto dim : shape)
           total_size *= dim;

       dataBuffer->resize(total_size);
       gradientBuffer->resize(total_size);
       shapeBuffer->alloc(shape);

       // --- Normal strides ---
       strides.resize(shape.size());
       strides.back() = 1;
       for (int i = (int)shape.size() - 2; i >= 0; --i)
           strides[i] = strides[i + 1] * shape[i + 1];

       // --- Effective strides for matmul ---
       effectiveStrides.resize(3);
       effectiveStrides[0] = shape.size() >= 2 ? shape[shape.size() - 1] * shape[shape.size() - 2] : 1;
       effectiveStrides[1] = shape.size() >= 2 ? shape[shape.size() - 1] : 1;
       effectiveStrides[2] = 1;
	   stridesBuffer->alloc(strides);
   }

    Tensor(std::vector<uint32_t> dims, Allocator* allocator)
        : shape(dims),
        allocator(allocator),
        dataBuffer(std::make_unique<StandaloneBuffer<T>>(1, allocator, VK_SHADER_STAGE_ALL)),
        gradientBuffer(std::make_unique<StandaloneBuffer<T>>(1, allocator, VK_SHADER_STAGE_ALL)),
        stridesBuffer(std::make_unique<StandaloneBuffer<uint32_t>>(1, allocator, VK_SHADER_STAGE_ALL)),
        shapeBuffer(std::make_unique<StandaloneBuffer<uint32_t>>(1, allocator, VK_SHADER_STAGE_ALL))
    {
        // Total size
        size_t total_size = 1;
        for (auto dim : shape)
            total_size *= dim;

        dataBuffer->resize(total_size);
        gradientBuffer->resize(total_size);
        shapeBuffer->alloc(shape);

        // --- Normal strides ---
        strides.resize(shape.size());
        strides.back() = 1;
        for (int i = (int)shape.size() - 2; i >= 0; --i)
            strides[i] = strides[i + 1] * shape[i + 1];

        // --- Effective strides for matmul ---
        effectiveStrides.resize(3);
        effectiveStrides[0] = shape.size() >= 2 ? shape[shape.size() - 1] * shape[shape.size() - 2] : 1;
        effectiveStrides[1] = shape.size() >= 2 ? shape[shape.size() - 1] : 1;
        effectiveStrides[2] = 1;
        stridesBuffer->alloc(strides);
    }


    Tensor(const std::string& filename, Allocator* allocator)
        : allocator(allocator),
        dataBuffer(std::make_unique<StandaloneBuffer<T>>(1, allocator, VK_SHADER_STAGE_ALL)),
        gradientBuffer(std::make_unique<StandaloneBuffer<T>>(1, allocator, VK_SHADER_STAGE_ALL)),
        stridesBuffer(std::make_unique<StandaloneBuffer<uint32_t>>(1, allocator, VK_SHADER_STAGE_ALL)),
        shapeBuffer(std::make_unique<StandaloneBuffer<uint32_t>>(1, allocator, VK_SHADER_STAGE_ALL))
    {
        load_from_file(filename);
    }

    uint32_t get_num_elements(uint32_t start_dim = 0, uint32_t end_dim = 0){
        return std::accumulate(shape.begin() + start_dim, shape.end() - end_dim, 1, std::multiplies<uint32_t>());
    }

   Tensor(const std::vector<uint32_t>& dims) : shape(dims) {  
       size_t total_size = 1;  
       for (auto dim : dims) {  
           total_size *= dim;  
       }  
       dataBuffer->resize(total_size);  

       strides.resize(shape.size());  
       strides.back() = 1;  
       for (int i = shape.size() - 2; i >= 0; --i) {  
           strides[i] = strides[i + 1] * shape[i + 1];  
       }  
    }  

    void backward() const {
        if (back) {std::cout << "calling backward for tensor " << name << "\n"; back(); }
    }

    void view(std::initializer_list<uint32_t> dims) {
        std::vector<uint32_t> new_shape(dims);
        if (new_shape == shape) return; // shapes equal -> no-op

        size_t total_size = 1;
        for (auto dim : new_shape) {
            total_size *= dim;
        }
        if (total_size != dataBuffer->size()) {
            throw std::invalid_argument("New shape must have the same number of elements");
        }
        shape = std::move(new_shape);

        // --- Normal strides ---
        strides.resize(shape.size());
        strides.back() = 1;
        for (int i = (int)shape.size() - 2; i >= 0; --i)
            strides[i] = strides[i + 1] * shape[i + 1];

        // --- Effective strides for matmul ---
        effectiveStrides.resize(3);
        effectiveStrides[0] = shape.size() >= 2 ? shape[shape.size() - 1] * shape[shape.size() - 2] : 1;
        effectiveStrides[1] = shape.size() >= 2 ? shape[shape.size() - 1] : 1;
        effectiveStrides[2] = 1;
        stridesBuffer->alloc(strides);
        shapeBuffer->alloc(shape);
    }

    void view(std::vector<uint32_t>& dims) {
        if (dims == shape) return; // shapes equal -> no-op

        size_t total_size = 1;
        for (auto dim : dims) {
            total_size *= dim;
        }
        if (total_size != dataBuffer->size()) {
            throw std::invalid_argument("New shape must have the same number of elements");
        }
        shape = dims;

        // --- Normal strides ---
        strides.resize(shape.size());
        strides.back() = 1;
        for (int i = (int)shape.size() - 2; i >= 0; --i)
            strides[i] = strides[i + 1] * shape[i + 1];

        // --- Effective strides for matmul ---
        effectiveStrides.resize(3);
        effectiveStrides[0] = shape.size() >= 2 ? shape[shape.size() - 1] * shape[shape.size() - 2] : 1;
        effectiveStrides[1] = shape.size() >= 2 ? shape[shape.size() - 1] : 1;
        effectiveStrides[2] = 1;
        stridesBuffer->alloc(strides);
        shapeBuffer->alloc(shape);
    }

    // doesn't transpose the data. Just does a view
    void transpose(){
        if (shape.size() < 2) {
            throw std::invalid_argument("Tensor must have at least 2 dimensions to transpose");
        }
        std::swap(shape[shape.size() - 1], shape[shape.size() - 2]);

        // --- Normal strides ---
        strides.resize(shape.size());
        strides.back() = 1;
        for (int i = (int)shape.size() - 2; i >= 0; --i)
            strides[i] = strides[i + 1] * shape[i + 1];

        // --- Effective strides for matmul ---
        effectiveStrides.resize(3);
        effectiveStrides[0] = shape.size() >= 2 ? shape[shape.size() - 1] * shape[shape.size() - 2] : 1;
        effectiveStrides[1] = shape.size() >= 2 ? shape[shape.size() - 1] : 1;
        effectiveStrides[2] = 1;
        stridesBuffer->alloc(strides);
        shapeBuffer->alloc(shape);
    }

    Tensor<T>& operator*(Tensor<T>&other);
    Tensor<T>& operator*(T other);
    Tensor<T>& operator+(Tensor<T>& other);
    Tensor<T>& operator+(T other);
    Tensor<T>& matmul(Tensor<T>& other); // matmul operator
    Tensor<T>& exp(); // output = tensor.exp(); // element-wise exponentiation
    void elementwise_multiply(Tensor<T>& other);
    void elementwise_add(Tensor<T>& other);
    
    void save_to_file(const std::string& filename) const {
        std::ofstream outFile(filename, std::ios::binary);
        if (!outFile) {
            throw std::runtime_error("Failed to open file for writing: " + filename);
        }

        // Write a magic number to identify tensor files
        const uint32_t magicNumber = 0x544E5352; // "TNSR" in ASCII
        outFile.write(reinterpret_cast<const char*>(&magicNumber), sizeof(magicNumber));

        // Write the name
        uint32_t nameLength = static_cast<uint32_t>(name.size());
        outFile.write(reinterpret_cast<const char*>(&nameLength), sizeof(nameLength));
        outFile.write(name.c_str(), nameLength);

        // Write the shape
        uint32_t shapeSize = static_cast<uint32_t>(shape.size());
        outFile.write(reinterpret_cast<const char*>(&shapeSize), sizeof(shapeSize));
        outFile.write(reinterpret_cast<const char*>(shape.data()), shapeSize * sizeof(uint32_t));

        // Write the strides
        uint32_t stridesSize = static_cast<uint32_t>(strides.size());
        outFile.write(reinterpret_cast<const char*>(&stridesSize), sizeof(stridesSize));
        outFile.write(reinterpret_cast<const char*>(strides.data()), stridesSize * sizeof(uint32_t));

        // Write the data buffer
        auto dataBuf = dataBuffer->downloadBuffer();
        uint32_t dataSize = static_cast<uint32_t>(dataBuf.size());
        outFile.write(reinterpret_cast<const char*>(&dataSize), sizeof(dataSize));
        outFile.write(reinterpret_cast<const char*>(dataBuf.data()), dataSize * sizeof(T));

        // Write the gradient buffer
        auto gradBuf = gradientBuffer->downloadBuffer();
        uint32_t gradSize = static_cast<uint32_t>(gradBuf.size());
        outFile.write(reinterpret_cast<const char*>(&gradSize), sizeof(gradSize));
        outFile.write(reinterpret_cast<const char*>(gradBuf.data()), gradSize * sizeof(T));

        outFile.close();
        std::cout << "Tensor saved to " << filename << "\n";
    }

    void save_to_stream(std::ofstream& out) const {
        if (!out)
            throw std::runtime_error("Invalid output stream");

        const uint32_t magicNumber = 0x544E5352; // "TNSR"
        out.write(reinterpret_cast<const char*>(&magicNumber), sizeof(magicNumber));

        uint32_t nameLength = static_cast<uint32_t>(name.size());
        out.write(reinterpret_cast<const char*>(&nameLength), sizeof(nameLength));
        out.write(name.c_str(), nameLength);

        uint32_t shapeSize = static_cast<uint32_t>(shape.size());
        out.write(reinterpret_cast<const char*>(&shapeSize), sizeof(shapeSize));
        out.write(reinterpret_cast<const char*>(shape.data()), shapeSize * sizeof(uint32_t));

        uint32_t stridesSize = static_cast<uint32_t>(strides.size());
        out.write(reinterpret_cast<const char*>(&stridesSize), sizeof(stridesSize));
        out.write(reinterpret_cast<const char*>(strides.data()), stridesSize * sizeof(uint32_t));

        auto dataBuf = dataBuffer->downloadBuffer();
        uint32_t dataSize = static_cast<uint32_t>(dataBuf.size());
        out.write(reinterpret_cast<const char*>(&dataSize), sizeof(dataSize));
        out.write(reinterpret_cast<const char*>(dataBuf.data()), dataSize * sizeof(T));

        auto gradBuf = gradientBuffer->downloadBuffer();
        uint32_t gradSize = static_cast<uint32_t>(gradBuf.size());
        out.write(reinterpret_cast<const char*>(&gradSize), sizeof(gradSize));
        out.write(reinterpret_cast<const char*>(gradBuf.data()), gradSize * sizeof(T));
    }

    void load_from_file(const std::string& filename) {
        std::ifstream inFile(filename, std::ios::binary);
        if (!inFile) {
            throw std::runtime_error("Failed to open file for reading: " + filename);
        }
        // Read and verify the magic number
        uint32_t magicNumber;
        inFile.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
        if (magicNumber != 0x544E5352) { // "TNSR" in ASCII
            throw std::runtime_error("Invalid tensor file: " + filename);
        }
        // Read the name
        uint32_t nameLength;
        inFile.read(reinterpret_cast<char*>(&nameLength), sizeof(nameLength));
        name.resize(nameLength);
        inFile.read(&name[0], nameLength);
        std::cout << "Loading tensor: " << name << "\n";

        // Read the shape
        uint32_t shapeSize;
        inFile.read(reinterpret_cast<char*>(&shapeSize), sizeof(shapeSize));
        shape.resize(shapeSize);
        inFile.read(reinterpret_cast<char*>(shape.data()), shapeSize * sizeof(uint32_t));
        shapeBuffer->alloc(shape);
        // Read the strides
        uint32_t stridesSize;
        inFile.read(reinterpret_cast<char*>(&stridesSize), sizeof(stridesSize));
        strides.resize(stridesSize);
        inFile.read(reinterpret_cast<char*>(strides.data()), stridesSize * sizeof(uint32_t));
        stridesBuffer->alloc(strides);
        // Read the data buffer
        uint32_t dataSize;
        inFile.read(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));
        dataBuffer->resize(dataSize);
        std::vector<T> dataBuf(dataSize);
        inFile.read(reinterpret_cast<char*>(dataBuf.data()), dataSize * sizeof(T));
        dataBuffer->alloc(dataBuf);
        // Read the gradient buffer
        uint32_t gradSize;
        inFile.read(reinterpret_cast<char*>(&gradSize), sizeof(gradSize));
        gradientBuffer->resize(gradSize);
        std::vector<T> gradBuf(gradSize);
        inFile.read(reinterpret_cast<char*>(gradBuf.data()), gradSize * sizeof(T));
        gradientBuffer->alloc(gradBuf);
        inFile.close();
        std::cout << "Tensor loaded from " << filename << "\n";
    }

    void load_from_stream(std::ifstream& in) {
        if (!in)
            throw std::runtime_error("Invalid input stream");

        // Verify magic number
        uint32_t magicNumber;
        in.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
        if (magicNumber != 0x544E5352)
            throw std::runtime_error("Invalid tensor stream (bad magic number)");

        // Name
        uint32_t nameLength;
        in.read(reinterpret_cast<char*>(&nameLength), sizeof(nameLength));
        name.resize(nameLength);
        in.read(&name[0], nameLength);
        std::cout << "Loading tensor: " << name << "\n";

        // Shape
        uint32_t shapeSize;
        in.read(reinterpret_cast<char*>(&shapeSize), sizeof(shapeSize));
        shape.resize(shapeSize);
        in.read(reinterpret_cast<char*>(shape.data()), shapeSize * sizeof(uint32_t));
        shapeBuffer->alloc(shape);

        // Strides
        uint32_t stridesSize;
        in.read(reinterpret_cast<char*>(&stridesSize), sizeof(stridesSize));
        strides.resize(stridesSize);
        in.read(reinterpret_cast<char*>(strides.data()), stridesSize * sizeof(uint32_t));
        stridesBuffer->alloc(strides);

        // Data buffer
        uint32_t dataSize;
        in.read(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));
        std::vector<T> dataBuf(dataSize);
        in.read(reinterpret_cast<char*>(dataBuf.data()), dataSize * sizeof(T));
        dataBuffer->alloc(dataBuf);

        // Gradient buffer
        uint32_t gradSize;
        in.read(reinterpret_cast<char*>(&gradSize), sizeof(gradSize));
        std::vector<T> gradBuf(gradSize);
        in.read(reinterpret_cast<char*>(gradBuf.data()), gradSize * sizeof(T));
        gradientBuffer->alloc(gradBuf);

        std::cout << "Tensor loaded from stream\n";
    }

   size_t getIndex(std::initializer_list<uint32_t> indices) const {  
       if (indices.size() != shape.size()) {  
           throw std::out_of_range("Number of indices must match the number of dimensions");  
       }  
       size_t index = 0;  
       auto stride_it = strides.begin();  
       for (auto idx : indices) {  
           index += idx * (*stride_it);  
           ++stride_it;  
       }  
       return index;  
   }

   tensor_impl getTensorImpl() const {
       tensor_impl impl{};
       impl.data = dataBuffer->getBufferAddress();
       impl.grad = gradientBuffer->getBufferAddress();
       impl.shape = shapeBuffer->getBufferAddress();
       impl.strides = stridesBuffer->getBufferAddress();
       impl.num_elements = dataBuffer->size();
       impl.num_dims = shape.size();
       impl.requires_gradient = gradientBuffer->size() > 0 ? 1 : 0;
       impl.is_leaf = 1;
       return impl;
   }

   void setElement(T element, std::initializer_list<uint32_t> indices) {
	   dataBuffer->set(getIndex(indices), element);
   }

   const T& operator[](std::initializer_list<uint32_t> indices) const {  
       return dataBuffer->get(getIndex(indices));  
   }

   void addElement(T element, std::initializer_list<uint32_t> indices) {  
       dataBuffer->set(getIndex(indices), element);  
   }  

   void print() const {  
       auto buf = dataBuffer->downloadBuffer();
	   std::cout << "Tensor shape: " << shapeToString(shape) << "\n";
	   std::cout << "Tensor data: ";
	   for (size_t i = 0; i < buf.size(); ++i) {
		   std::cout << buf[i] << " ";
	   }
	   std::cout << "\n";
   }

   void printShape() const {
        std::cout << "Tensor shape: (";
        for (auto& dim : shape) {
            std::cout << dim << ", ";
        }
        std::cout << ")\n";
   }

   void printGradient() const {
       auto buf = gradientBuffer->downloadBuffer();
       std::cout << "Tensor shape: " << shapeToString(shape) << "\n";
       std::cout << "Tensor gradient: ";
       for (size_t i = 0; i < buf.size(); ++i) {
           std::cout << buf[i] << " ";
       }
       std::cout << "\n";
   }
};

template<typename T>
struct OpNode {
    std::vector<Tensor<T>*> inputs;
    std::vector<Tensor<T>*> outputs;
    std::function<void()> backward;  // runs appropriate GPU kernel
    uint32_t ref_count = 0;          // for topological execution cleanup
};

template <typename T>
struct BiOpNode;

template<typename T>
struct TensorPool {
    std::unordered_map<std::string, std::unique_ptr<Tensor<T>>> tensors;

    Allocator* allocator = nullptr;
    uint32_t tile_size[3] = { 16, 16, 1 };
    TensorPool(Allocator* alloc)
            : allocator(alloc),
            inplaceAdditionShader(
                readShaderBytecode("compiled_shaders/tensor_inplace_addition.comp.spv"), alloc, tile_size),
            inplaceAdditionShaderBackward(
                readShaderBytecode("compiled_shaders/tensor_inplace_addition_backward.comp.spv"), alloc, tile_size),
            elementwiseMultiplicationShader(
                readShaderBytecode("compiled_shaders/element_wise_multiply.comp.spv"), alloc, nullptr),
            elementwiseMultiplicationShaderBackward(
                readShaderBytecode("compiled_shaders/element_wise_multiply_backward.comp.spv"), alloc, nullptr),
            batchnormShader(
                readShaderBytecode("compiled_shaders/Batchnorm.comp.spv"), alloc, nullptr),
            batchnormShaderBackward(
                readShaderBytecode("compiled_shaders/Batchnorm_backward.comp.spv"), alloc, nullptr),
            batchnorm2dShader(
                readShaderBytecode("compiled_shaders/Batchnorm2d.comp.spv"), alloc, nullptr),
            batchnorm2dShaderBackward(
                readShaderBytecode("compiled_shaders/Batchnorm2d_backward.comp.spv"), alloc, nullptr),
            layernorm1dShader(
                readShaderBytecode("compiled_shaders/Layernorm1d.comp.spv"), alloc, nullptr),
            layernorm1dShaderBackward(
                readShaderBytecode("compiled_shaders/Layernorm1d_backward.comp.spv"), alloc, nullptr),
            linearReLUShader(
                readShaderBytecode("compiled_shaders/Linear.comp.spv"), alloc, nullptr),
            linearReLUShaderBackward(
                readShaderBytecode("compiled_shaders/Linear_backward.comp.spv"), alloc, nullptr),
            linearShader(
                readShaderBytecode("compiled_shaders/Linear_no_relu.comp.spv"), alloc, nullptr),
            linearShaderBackward(
                readShaderBytecode("compiled_shaders/Linear_no_relu_backward.comp.spv"), alloc, nullptr),
            logvarToStdShader(
                readShaderBytecode("compiled_shaders/logvar_to_std.comp.spv"), alloc, nullptr),
            logvarToStdShaderBackward(
                readShaderBytecode("compiled_shaders/logvar_to_std_backward.comp.spv"), alloc, nullptr),
            expShader(
                readShaderBytecode("compiled_shaders/tensor_exp.comp.spv"), alloc, nullptr),
            expShaderBackward(
                readShaderBytecode("compiled_shaders/tensor_exp_backward.comp.spv"), alloc, nullptr),
            ReLUShader(
                readShaderBytecode("compiled_shaders/ReLU.comp.spv"), alloc, nullptr),
            ReLUShaderBackward(
                readShaderBytecode("compiled_shaders/ReLU_backward.comp.spv"), alloc, nullptr),
            fillRandomShader(
                readShaderBytecode("compiled_shaders/tensor_fill_random.comp.spv"), alloc, nullptr),
            crossEntropyShader(
                readShaderBytecode("compiled_shaders/Fused_Cross_Entropy.comp.spv"), alloc, nullptr),
            crossEntropyShaderBackward(
                readShaderBytecode("compiled_shaders/Fused_Cross_Entropy_backward.comp.spv"), alloc, nullptr),
            mseLossShader(
                readShaderBytecode("compiled_shaders/MSE_loss.comp.spv"), alloc, nullptr),
            kldLossShader(
                readShaderBytecode("compiled_shaders/KLD_loss.comp.spv"), alloc, nullptr),
            mean_shader(
                readShaderBytecode("compiled_shaders/find_mean.comp.spv"), alloc, nullptr),
            softmaxShader(
                readShaderBytecode("compiled_shaders/Softmax.comp.spv"), alloc, nullptr),
            embedLookupShader(
                readShaderBytecode("compiled_shaders/Embedding_table.comp.spv"), alloc, nullptr),
            embedLookupShaderBackward(
                readShaderBytecode("compiled_shaders/Embedding_table_backward.comp.spv"), alloc, nullptr),
            conv2dShader3x3(
                readShaderBytecode("compiled_shaders/Conv2d3x3.comp.spv"), alloc, nullptr),
            conv2dShader3x3Backward(
                readShaderBytecode("compiled_shaders/Conv2d3x3_backward.comp.spv"), alloc, nullptr),
            conv2dShader(
                readShaderBytecode("compiled_shaders/Conv2d.comp.spv"), alloc, nullptr),
            conv2dShaderBackward(
                readShaderBytecode("compiled_shaders/Conv2d_backward.comp.spv"), alloc, nullptr),
            transposedConv2dShader(
                readShaderBytecode("compiled_shaders/Conv2d_Transposed.comp.spv"), alloc, nullptr),
            transposedConv2dShaderBackward(
                readShaderBytecode("compiled_shaders/Conv2d_Transposed_backward.comp.spv"), alloc, nullptr),
            maxPoolShader(
                readShaderBytecode("compiled_shaders/MaxPooling.comp.spv"), alloc, tile_size),
            maxPoolShaderBackward(
                readShaderBytecode("compiled_shaders/MaxPooling_backward.comp.spv"), alloc, nullptr),
            cmpShader(
                readShaderBytecode("compiled_shaders/Is_tensor_equal.comp.spv"), alloc, nullptr)
    {
        DEBUG_PRINT("Initialized TensorPool");
    }

    TensorPool(){
        // empty ctor
        DEBUG_PRINT("Tensor pool needs Allocator for proper construction. e");
    }

    Tensor<T>& getTensor(const std::string& name, std::initializer_list<size_t> dims) {
        if (tensors.find(name) == tensors.end()) {
            tensors[name] = std::make_unique<Tensor<T>>(dims, allocator);
        }
        return *tensors[name];
    }

    Tensor<T>& getTensor(const std::string& name, const std::vector<size_t>& dims) {
        if (tensors.find(name) == tensors.end()) {
            throw std::runtime_error("Tensor with this name does not exist");
        }
        return *tensors[name];
    }

    Tensor<T>& createTensor(std::initializer_list<uint32_t> dims, const std::string& name) {
        if (tensors.find(name) == tensors.end()) {
            tensors[name] = std::make_unique<Tensor<T>>(dims, allocator);
            tensors[name]->name = name;
            tensors[name]->pool = this;
            return *tensors[name];
        }
        else {
            DEBUG_PRINT("tensor with name: " << name << " already exists. Returning that tensor.");
            return *tensors[name];
        }
    }

    Tensor<T>& createTensor(const std::vector<uint32_t>& dims, const std::string& name) {
        if (tensors.find(name) == tensors.end()) {
            tensors[name] = std::make_unique<Tensor<T>>(dims, allocator);
            tensors[name]->name = name;
            tensors[name]->pool = this;
            return *tensors[name];
        }
        else {
            DEBUG_PRINT("tensor with name: " << name << " already exists. Returning that tensor.");
            return *tensors[name];
        }
    }

    void destroy_tensor(const std::string& name) {
        auto it = tensors.find(name);
        if (it != tensors.end()) {
            tensors.erase(it);
        }
        else {
            throw std::runtime_error("Tensor with this name does not exist");
        }
    }

    void zero_out_all_grads(){
        for (auto& [name, tensor] : tensors){
            tensor->gradientBuffer->clearBuffer();
        }
    }
    
    void tensor_ReLU(const std::string& output_tensor, const std::string& input_tensor, uint32_t mode = 0) {
        tensor_relu_context uniform{};
        uniform.input_tensor = tensors[input_tensor]->getTensorImpl();
        uniform.output_tensor = tensors[output_tensor]->getTensorImpl();
        uniform.mode = mode; // forwarded to shader if needed

        // Support any-rank tensor: compute total number of elements and dispatch linearly
        uint32_t total_elements = tensors[input_tensor]->get_num_elements();
        // fill m/n for compatibility (shader can ignore if not needed)
        uniform.m = 1;
        uniform.n = total_elements;

        DEBUG_PRINT("ReLU'ing tensor " << input_tensor << " total elements: " << total_elements
                    << " shape: " << shapeToString(tensors[input_tensor]->shape));

        auto cielDiv = [](uint32_t a, uint32_t b) { return (a + b - 1) / b; };
        uint32_t groupX = cielDiv(total_elements, 256u);
        uint32_t groupY = 1;
        uint32_t groupZ = 1;

        DEBUG_PRINT("Dispatch: " << groupX << " x " << groupY << " x " << groupZ
            << " (covering total elements " << total_elements << ")");

        uint32_t workgroup[3] = { groupX, groupY, groupZ };
        tensor_push_const pushConsts{};
        pushConsts.grid_size = { groupX, groupY, groupZ };

        if (mode == 0) {
            pushConsts.mode = mode;
            pushConsts.uniformAddress = ReLUShader.uniformBuffer->getBufferAddress();
            ReLUShader.loadUniform(uniform, pushConsts);
            ReLUShader.execute(workgroup);
        } else if (mode == 1) {
            pushConsts.mode = mode;
            pushConsts.uniformAddress = ReLUShaderBackward.uniformBuffer->getBufferAddress();
            ReLUShaderBackward.loadUniform(uniform, pushConsts);
            ReLUShaderBackward.execute(workgroup);
        }
    };

    // output tensor = (B, token_count, embedding_dim)
    // embedding tensor = (vocab_size, embedding_dim)
    // indices tensor = (B, token_count)
    void tensor_embed_lookup(const std::string& output_tensor, const std::string& embedding_tensor, const std::string& indices_tensor, uint32_t mode = 0) {
        uint32_t local_size_x = 256;
        size_t batch_size = tensors[indices_tensor]->shape[0];
        size_t token_count = tensors[indices_tensor]->shape[1];
        size_t embedding_dim = tensors[embedding_tensor]->shape[1];

        size_t total_invocations = batch_size * token_count;
        size_t group_count = (total_invocations + local_size_x - 1) / local_size_x;

        embedding_lookup_context uniform{};
        uniform.embedding_tensor = tensors[embedding_tensor]->getTensorImpl();
        uniform.indices_tensor = tensors[indices_tensor]->getTensorImpl();
        uniform.output_tensor = tensors[output_tensor]->getTensorImpl();

        DEBUG_PRINT("Embedding lookup tensor " << embedding_tensor << " with indices from " << indices_tensor << " into " << output_tensor);
        uint32_t workgroup[3] = { (uint32_t)group_count, 1, 1 };
        tensor_push_const pushConsts{};

        if(mode == 0){
            pushConsts.mode = mode; // forward pass
            pushConsts.uniformAddress = embedLookupShader.uniformBuffer->getBufferAddress();
            pushConsts.grid_size = { (uint32_t)group_count, 1, 1 };
            embedLookupShader.loadUniform(uniform, pushConsts);
            embedLookupShader.execute(workgroup);
        }
        else if (mode == 1) {
            pushConsts.mode = mode; // backward pass
            pushConsts.uniformAddress = embedLookupShaderBackward.uniformBuffer->getBufferAddress();
            pushConsts.grid_size = { (uint32_t)group_count, 1, 1 };
            embedLookupShaderBackward.loadUniform(uniform, pushConsts);
            embedLookupShaderBackward.execute(workgroup);
        }
    };

    // simple 3x3 convolution with padding 1, stride 1, dilation 1, groups 1
    void tensor_conv2d_3x3(const std::string& output_tensor, const std::string& input_tensor, const std::string& weight_tensor, const std::string& bias_tensor, uint32_t mode = 0,
        uint32_t stride_w = 1, uint32_t stride_h = 1, uint32_t pad_h = 1, uint32_t pad_w = 1, uint32_t dilation_h = 1, uint32_t dilation_w = 1, uint32_t groups = 1) {
        tensor_conv2d_3x3_context uniform{};
        uniform.input_tensor = tensors[input_tensor]->getTensorImpl();
        uniform.weight_tensor = tensors[weight_tensor]->getTensorImpl();
        uniform.bias_tensor = tensors[bias_tensor]->getTensorImpl();
        uniform.out_tensor = tensors[output_tensor]->getTensorImpl();
        uniform.stride_h = stride_h;
        uniform.stride_w = stride_w;
        uniform.pad_h = pad_h;
        uniform.pad_w = pad_w;
        uniform.dilation_h = dilation_h;
        uniform.dilation_w = dilation_w;
        uniform.groups = groups;
        uniform.accumulate_grad = 1; // accumulate gradients

        DEBUG_PRINT("Conv2D 3x3 tensor " << input_tensor << " with weights from " << weight_tensor << " into " << output_tensor);
        
        auto cielDiv = [](uint32_t a, uint32_t b) { return (a + b - 1) / b; };
        
        tensor_push_const pushConsts{};
        if (mode == 0) {
            uint32_t groupX = cielDiv(tensors[output_tensor]->shape[3], 16u); // W_out
            uint32_t groupY = cielDiv(tensors[output_tensor]->shape[2], 16u); // H_out
            uint32_t groupZ = tensors[output_tensor]->shape[0] * tensors[weight_tensor]->shape[0]; // batch * Filters

            uint32_t workgroup[3] = { groupX, groupY, groupZ };
            pushConsts.mode = mode; // forward pass
            pushConsts.uniformAddress = conv2dShader3x3.uniformBuffer->getBufferAddress();
            pushConsts.grid_size = { groupX, groupY, groupZ };
            conv2dShader3x3.loadUniform(uniform, pushConsts);
            conv2dShader3x3.execute(workgroup);
        }
        else if (mode == 1) {
            pushConsts.uniformAddress = conv2dShader3x3Backward.uniformBuffer->getBufferAddress();
            
            // Kernel 0: Compute gradient w.r.t. input
            {
                uint32_t groupX = cielDiv(tensors[input_tensor]->shape[3], 16u); // W_in
                uint32_t groupY = cielDiv(tensors[input_tensor]->shape[2], 16u); // H_in
                uint32_t groupZ = tensors[input_tensor]->shape[0] * tensors[input_tensor]->shape[1]; // batch * C_in
                uint32_t workgroup[3] = { groupX, groupY, groupZ };
                
                pushConsts.grid_size = { groupX, groupY, groupZ };
                pushConsts.mode = 0; // input gradient
                uniform.kernel_type = 0;
                conv2dShader3x3Backward.loadUniform(uniform, pushConsts);
                conv2dShader3x3Backward.execute(workgroup);
            }
            
            // Kernel 1: Compute gradient w.r.t. weights
            {
                uint32_t groupX = cielDiv(tensors[input_tensor]->shape[3], 16u); // W_in
                uint32_t groupY = cielDiv(tensors[input_tensor]->shape[2], 16u); // H_in
                uint32_t groupZ = tensors[input_tensor]->shape[0] * tensors[weight_tensor]->shape[0]; // batch * F
                uint32_t workgroup[3] = { groupX, groupY, groupZ };
                
                pushConsts.grid_size = { groupX, groupY, groupZ };
                pushConsts.mode = 1; // weight gradient
                uniform.kernel_type = 1;
                conv2dShader3x3Backward.loadUniform(uniform, pushConsts);
                conv2dShader3x3Backward.execute(workgroup);
            }
            
            // Kernel 2: Compute gradient w.r.t. bias
            {
                uint32_t groupX = tensors[weight_tensor]->shape[0]; // F (num filters)
                uint32_t groupY = 1;
                uint32_t groupZ = 1;
                uint32_t workgroup[3] = { groupX, groupY, groupZ };
                
                pushConsts.grid_size = { groupX, groupY, groupZ };
                pushConsts.mode = 2; // bias gradient
                uniform.kernel_type = 2;
                conv2dShader3x3Backward.loadUniform(uniform, pushConsts);
                conv2dShader3x3Backward.execute(workgroup);
            }
        }
    };

    // general MxN convolution supporting kernel sizes up to 15x15
    void tensor_conv2d(const std::string& output_tensor, const std::string& input_tensor, const std::string& weight_tensor, const std::string& bias_tensor, uint32_t kernel_h, uint32_t kernel_w, uint32_t mode = 0,
        uint32_t stride_w = 1, uint32_t stride_h = 1, uint32_t pad_h = 1, uint32_t pad_w = 1, uint32_t dilation_h = 1, uint32_t dilation_w = 1, uint32_t groups = 1) {
        tensor_conv2d_context uniform{};
        uniform.input_tensor = tensors[input_tensor]->getTensorImpl();
        uniform.weight_tensor = tensors[weight_tensor]->getTensorImpl();
        uniform.bias_tensor = tensors[bias_tensor]->getTensorImpl();
        uniform.out_tensor = tensors[output_tensor]->getTensorImpl();
        uniform.stride_h = stride_h;
        uniform.stride_w = stride_w;
        uniform.pad_h = pad_h;
        uniform.pad_w = pad_w;
        uniform.dilation_h = dilation_h;
        uniform.dilation_w = dilation_w;
        uniform.kernel_h = kernel_h;
        uniform.kernel_w = kernel_w;
        uniform.groups = groups;
        uniform.accumulate_grad = 1; // accumulate gradients

        DEBUG_PRINT("Conv2D tensor " << input_tensor << " with weights from " << weight_tensor << " into " << output_tensor);
        
        auto cielDiv = [](uint32_t a, uint32_t b) { return (a + b - 1) / b; };
        
        tensor_push_const pushConsts{};
        if (mode == 0) {
            uint32_t groupX = cielDiv(tensors[output_tensor]->shape[3], 16u); // W_out
            uint32_t groupY = cielDiv(tensors[output_tensor]->shape[2], 16u); // H_out
            uint32_t groupZ = tensors[output_tensor]->shape[0] * tensors[weight_tensor]->shape[0]; // batch * Filters

            uint32_t workgroup[3] = { groupX, groupY, groupZ };
            pushConsts.mode = mode; // forward pass
            pushConsts.uniformAddress = conv2dShader.uniformBuffer->getBufferAddress();
            pushConsts.grid_size = { groupX, groupY, groupZ };
            conv2dShader.loadUniform(uniform, pushConsts);
            conv2dShader.execute(workgroup);
        }
        else if (mode == 1) {
            pushConsts.uniformAddress = conv2dShaderBackward.uniformBuffer->getBufferAddress();
            
            // Kernel 0: Compute gradient w.r.t. input
            {
                uint32_t groupX = cielDiv(tensors[input_tensor]->shape[3], 16u); // W_in
                uint32_t groupY = cielDiv(tensors[input_tensor]->shape[2], 16u); // H_in
                uint32_t groupZ = tensors[input_tensor]->shape[0] * tensors[input_tensor]->shape[1]; // batch * C_in
                uint32_t workgroup[3] = { groupX, groupY, groupZ };
                
                pushConsts.grid_size = { groupX, groupY, groupZ };
                pushConsts.mode = 0; // input gradient
                uniform.kernel_type = 0;
                conv2dShaderBackward.loadUniform(uniform, pushConsts);
                conv2dShaderBackward.execute(workgroup);
            }
            
            // Kernel 1: Compute gradient w.r.t. weights
            {
                uint32_t groupX = cielDiv(tensors[input_tensor]->shape[3], 16u); // W_in
                uint32_t groupY = cielDiv(tensors[input_tensor]->shape[2], 16u); // H_in
                uint32_t groupZ = tensors[input_tensor]->shape[0] * tensors[weight_tensor]->shape[0]; // batch * F
                uint32_t workgroup[3] = { groupX, groupY, groupZ };
                
                pushConsts.grid_size = { groupX, groupY, groupZ };
                pushConsts.mode = 1; // weight gradient
                uniform.kernel_type = 1;
                conv2dShaderBackward.loadUniform(uniform, pushConsts);
                conv2dShaderBackward.execute(workgroup);
            }
            
            // Kernel 2: Compute gradient w.r.t. bias
            {
                uint32_t groupX = tensors[weight_tensor]->shape[0]; // F (num filters)
                uint32_t groupY = 1;
                uint32_t groupZ = 1;
                uint32_t workgroup[3] = { groupX, groupY, groupZ };
                
                pushConsts.grid_size = { groupX, groupY, groupZ };
                pushConsts.mode = 2; // bias gradient
                uniform.kernel_type = 2;
                conv2dShaderBackward.loadUniform(uniform, pushConsts);
                conv2dShaderBackward.execute(workgroup);
            }
        }
    };

    // general transposed (de)conv2d supporting dynamic kernel sizes
    void tensor_transposed_conv2d(const std::string& output_tensor,
                                const std::string& input_tensor,
                                const std::string& weight_tensor,
                                const std::string& bias_tensor,
                                uint32_t kernel_h, uint32_t kernel_w,
                                uint32_t mode = 0,
                                uint32_t stride_w = 1, uint32_t stride_h = 1,
                                uint32_t pad_h = 0, uint32_t pad_w = 0,
                                uint32_t dilation_h = 1, uint32_t dilation_w = 1,
                                uint32_t output_pad_h = 0, uint32_t output_pad_w = 0,
                                uint32_t groups = 1) {
        tensor_transposed_conv2d_context uniform{};
        uniform.input_tensor = tensors[input_tensor]->getTensorImpl();   // [N, C_in, H_in, W_in]
        uniform.weight_tensor = tensors[weight_tensor]->getTensorImpl(); // [C_in, C_out, K_h, K_w]
        uniform.bias_tensor = tensors[bias_tensor]->getTensorImpl();     // [C_out]
        uniform.out_tensor = tensors[output_tensor]->getTensorImpl();    // [N, C_out, H_out, W_out]
        uniform.stride_h = stride_h;
        uniform.stride_w = stride_w;
        uniform.pad_h = pad_h;
        uniform.pad_w = pad_w;
        uniform.dilation_h = dilation_h;
        uniform.dilation_w = dilation_w;
        uniform.kernel_h = kernel_h;
        uniform.kernel_w = kernel_w;
        uniform.output_pad_h = output_pad_h;
        uniform.output_pad_w = output_pad_w;
        uniform.groups = groups;
        uniform.accumulate_grad = 1; // accumulate gradients

        DEBUG_PRINT("TransposedConv2D tensor " << input_tensor << " with weights from " << weight_tensor << " into " << output_tensor);

        auto cielDiv = [](uint32_t a, uint32_t b) { return (a + b - 1) / b; };

        tensor_push_const pushConsts{};
        if (mode == 0) {
            // Forward pass: map threads over output spatial dims and output channels
            uint32_t groupX = cielDiv(tensors[output_tensor]->shape[3], 16u); // W_out
            uint32_t groupY = cielDiv(tensors[output_tensor]->shape[2], 16u); // H_out
            // output channels (C_out) are weight_tensor->shape[1] per your struct
            uint32_t groupZ = tensors[output_tensor]->shape[0] * tensors[weight_tensor]->shape[1]; // batch * C_out

            uint32_t workgroup[3] = { groupX, groupY, groupZ };
            pushConsts.mode = 0; // forward
            pushConsts.uniformAddress = transposedConv2dShader.uniformBuffer->getBufferAddress();
            pushConsts.grid_size = { groupX, groupY, groupZ };
            transposedConv2dShader.loadUniform(uniform, pushConsts);
            transposedConv2dShader.execute(workgroup);
        }
        else if (mode == 1) {
            const uint32_t TILE = 16u;
            pushConsts.uniformAddress = transposedConv2dShaderBackward.uniformBuffer->getBufferAddress();

            // Kernel 0: gradient w.r.t. input (smaller input spatial)
            {
                uint32_t H_in = tensors[input_tensor]->shape[2];
                uint32_t W_in = tensors[input_tensor]->shape[3];
                uint32_t B = tensors[input_tensor]->shape[0];
                uint32_t C_in = tensors[input_tensor]->shape[1];

                uint32_t grid_x = cielDiv(H_in, TILE); // tile over height
                uint32_t grid_y = cielDiv(W_in, TILE); // tile over width
                uint32_t grid_z = B * C_in;            // batch * C_in

                uint32_t workgroup[3] = { grid_x, grid_y, grid_z };
                pushConsts.grid_size = { grid_x, grid_y, grid_z };
                pushConsts.mode = 0; // input gradient
                uniform.kernel_type = 0;
                transposedConv2dShaderBackward.loadUniform(uniform, pushConsts);
                transposedConv2dShaderBackward.execute(workgroup);
            }

            // Kernel 1: gradient w.r.t. weights
            {
                uint32_t H_out = tensors[output_tensor]->shape[2];
                uint32_t W_out = tensors[output_tensor]->shape[3];
                uint32_t B = tensors[input_tensor]->shape[0];
                uint32_t C_in = tensors[input_tensor]->shape[1];

                uint32_t grid_x = cielDiv(H_out, TILE); // tile over output height
                uint32_t grid_y = cielDiv(W_out, TILE); // tile over output width
                uint32_t grid_z = B * C_in;             // batch * C_in (note: C_in, not C_out)

                uint32_t workgroup[3] = { grid_x, grid_y, grid_z };
                pushConsts.grid_size = { grid_x, grid_y, grid_z };
                pushConsts.mode = 1; // weight gradient
                uniform.kernel_type = 1;
                transposedConv2dShaderBackward.loadUniform(uniform, pushConsts);
                transposedConv2dShaderBackward.execute(workgroup);
            }

            // Kernel 2: gradient w.r.t. bias
            {
                uint32_t C_out = tensors[weight_tensor]->shape[1]; // C_out
                uint32_t workgroup[3] = { C_out, 1u, 1u };
                pushConsts.grid_size = { C_out, 1u, 1u };
                pushConsts.mode = 2; // bias gradient
                uniform.kernel_type = 2;
                transposedConv2dShaderBackward.loadUniform(uniform, pushConsts);
                transposedConv2dShaderBackward.execute(workgroup);
            }
        }
    }

    void tensor_max_pool(const std::string& input_tensor, const std::string& output_tensor, uint32_t kernel_h, uint32_t kernel_w, uint32_t stride_h, uint32_t stride_w, uint32_t mode = 0){
        tensor_max_pool_context uniform;
        uniform.input_tensor = tensors[input_tensor]->getTensorImpl();
        uniform.output_tensor = tensors[output_tensor]->getTensorImpl();
        uniform.kernel_size_h = kernel_h;
        uniform.kernel_size_w = kernel_w;
        uniform.stride_h = stride_h;
        uniform.stride_w = stride_w;

        uint32_t numel = uniform.output_tensor.num_elements;
        uint32_t groupX = (numel + 255) / 256;
        uint32_t workgrp[3] = {groupX, 1, 1};
        tensor_push_const push;
        if(mode == 0){
            push.mode = mode;
            push.uniformAddress = maxPoolShader.uniformBuffer->getBufferAddress();
            maxPoolShader.loadUniform(uniform, push);
            maxPoolShader.execute(workgrp);
        }else if (mode == 1) {
            push.mode = mode;
            push.uniformAddress = maxPoolShaderBackward.uniformBuffer->getBufferAddress();
            maxPoolShaderBackward.loadUniform(uniform, push);
            maxPoolShaderBackward.execute(workgrp);
        }
    }

    void tensor_softmax(const std::string& output_tensor, const std::string& input_tensor, uint32_t mode = 0) {
        // Check if shapes match
        try {
            broadcastShapes(tensors[output_tensor]->shape, tensors[input_tensor]->shape, Operation::ADDITION);
        }
        catch (const std::invalid_argument& e) {
            std::cerr << "Error in tensor_softmax: " << e.what() << "\n";
            throw;
        }
        tensor_softmax_context uniform{};
        uniform.input_tensor = tensors[input_tensor]->getTensorImpl();
        uniform.output_tensor = tensors[output_tensor]->getTensorImpl();
        // Helpers
        auto prod = [](const std::vector<uint32_t>& v, size_t l, size_t r)->uint32_t {
            uint64_t p = 1; for (size_t i = l; i < r; ++i) p *= v[i];
            if (p > UINT32_MAX) throw std::overflow_error("Batch too large");
            return (uint32_t)p;
            };
        const auto& shape = tensors[input_tensor]->shape; // X[..., M, N]
        if (shape.size() < 2)
            throw std::invalid_argument("Rank must be >= 2");
        uint32_t M = shape[shape.size() - 2];
        uint32_t N = shape.back();
        uint32_t batch_size = prod(shape, 0, shape.size() - 2);
        DEBUG_PRINT("Softmax'ing tensor " << input_tensor << " of shape " << shapeToString(shape));
        uniform.m = M;
        uniform.n = N;
        // Compute dispatch grid to cover the OUTPUT tensor C[B,M,N]
        auto ceilDiv = [](uint32_t a, uint32_t b) { return (a + b - 1) / b; };
        // Grid covers the output tensor dimensions C[M,N] per batch
        uint32_t groupX = ceilDiv(M, 256u);  // Cover all N columns
        uint32_t groupY = 1;  // Single row (softmax over last dim)
        uint32_t groupZ = batch_size;       // One group per batch element
        DEBUG_PRINT("Dispatch: " << groupX << " � " << groupY << " � " << groupZ
            << " (covering output " << M << "�" << N << " per batch)");
        uint32_t workgroup[3] = { groupX, groupY, groupZ };
        tensor_push_const pushConsts{};
        pushConsts.uniformAddress = softmaxShader.uniformBuffer->getBufferAddress();
        pushConsts.grid_size = { groupX, groupY, groupZ };
        pushConsts.mode = mode; // Forward pass
        softmaxShader.loadUniform(uniform, pushConsts);
        softmaxShader.execute(workgroup);
    };

    void tensor_cross_entropy(const std::string& loss_tensor, const std::string& logits_tensor, const std::string& target_tensor, const std::string& softmax_tensor = "", uint32_t mode = 0) {
        // Check if shapes match
        try {
            broadcastShapes(tensors[logits_tensor]->shape, tensors[target_tensor]->shape, Operation::ADDITION);
            if (!softmax_tensor.empty()) {
                broadcastShapes(tensors[logits_tensor]->shape, tensors[softmax_tensor]->shape, Operation::ADDITION);
            }
        }
        catch (const std::invalid_argument& e) {
            std::cerr << "Error in tensor_cross_entropy: " << e.what() << "\n";
            throw;
        }
        tensor_cross_entropy_context uniform{};
        uniform.logits_tensor = tensors[logits_tensor]->getTensorImpl();
        uniform.target_tensor = tensors[target_tensor]->getTensorImpl();
        uniform.loss_tensor = tensors[loss_tensor]->getTensorImpl();
        if (!softmax_tensor.empty()) {
            uniform.softmax_tensor = tensors[softmax_tensor]->getTensorImpl();
        }
        // Helpers
        auto prod = [](const std::vector<uint32_t>& v, size_t l, size_t r)->uint32_t {
            uint64_t p = 1; for (size_t i = l; i < r; ++i) p *= v[i];
            if (p > UINT32_MAX) throw std::overflow_error("Batch too large");
            return (uint32_t)p;
            };
        const auto& shape = tensors[logits_tensor]->shape; // X[..., M, N]
        if (shape.size() < 2)
            throw std::invalid_argument("Rank must be >= 2");
        uint32_t M = shape[shape.size() - 2];
        uint32_t N = shape.back();
        uint32_t batch_size = prod(shape, 0, shape.size() - 2);
        DEBUG_PRINT("Cross-entropy'ing tensor " << logits_tensor << " of shape " << shapeToString(shape));
        uniform.m = M;
        uniform.n = N;
        uniform.batch_size = batch_size;
        uniform.compute_softmax = softmax_tensor.empty() ? 0 : 1;
        // Compute dispatch grid to cover the OUTPUT tensor C[B,M,N]
        auto ceilDiv = [](uint32_t a, uint32_t b) { return (a + b - 1) / b; };
        // Grid covers the output tensor dimensions C[M,N] per batch
        uint32_t groupX = ceilDiv(N, 256u); // Cover all N columns
        uint32_t groupY = M;                // cross entropy over all rows
        uint32_t groupZ = batch_size;       // One group per batch element
        DEBUG_PRINT("Dispatch: " << groupX << " � " << groupY << " � " << groupZ
            << " (covering output " << M << "�" << N << " per batch)");
        uint32_t workgroup[3] = { groupX, groupY, groupZ };
        tensor_push_const pushConsts{};
        pushConsts.grid_size = { groupX, groupY, groupZ };
        if (mode == 0) {
            pushConsts.mode = mode; // Forward pass
            pushConsts.uniformAddress = crossEntropyShader.uniformBuffer->getBufferAddress();
            crossEntropyShader.loadUniform(uniform, pushConsts);
            crossEntropyShader.execute(workgroup);
        }
        else if (mode == 1) {
            pushConsts.mode = mode; // Backward pass
            pushConsts.uniformAddress = crossEntropyShaderBackward.uniformBuffer->getBufferAddress();
            crossEntropyShaderBackward.loadUniform(uniform, pushConsts);
            crossEntropyShaderBackward.execute(workgroup);
        }
    };

    void tensor_batchnorm_1d(const std::string& input_tensor,
                         const std::string& weight_tensor,
                         const std::string& bias_tensor,
                         const std::string& running_mean,
                         const std::string& running_var,
                         const std::string& out_tensor,
                         const std::string& save_mean,
                         const std::string& save_var,
                         uint32_t mode = 0 // 0 = forward, 1 = backward
                         )
    {
        // Validate existence & shapes
        try {
            broadcastShapes(tensors[input_tensor]->shape, tensors[weight_tensor]->shape, Operation::OTHER);
            broadcastShapes(tensors[input_tensor]->shape, tensors[bias_tensor]->shape, Operation::OTHER);
            // We expect running_mean/var and save_mean/var to match the feature dims
        }
        catch (const std::invalid_argument& e) {
            std::cerr << "Error in tensor_batchnorm_1d (broadcast): " << e.what() << "\n";
            throw;
        }

        tensor_batchnorm_context uniform{};

        auto prod = [](const std::vector<uint32_t>& v, size_t l, size_t r)->uint32_t {
            uint64_t p = 1; for (size_t i = l; i < r; ++i) p *= v[i];
            if (p > UINT32_MAX) throw std::overflow_error("Batch too large");
            return (uint32_t)p;
        }; 

        const auto& inShape = tensors[input_tensor]->shape;
        if (inShape.size() < 2)
            throw std::invalid_argument("Input rank must be >= 2 for BatchNorm1D (need [B..., M, N])");

        // last-2 dims are feature dims [M, N]
        uint32_t M = inShape[inShape.size() - 2];
        uint32_t N = inShape.back();
        uint32_t batch_size = prod(inShape, 0, inShape.size() - 2);

        // Fill uniform/context expected by shader (field names follow your GLSL Context)
        uniform.input_tensor   = tensors[input_tensor]->getTensorImpl();
        uniform.weight_tensor  = tensors[weight_tensor]->getTensorImpl();
        uniform.bias_tensor    = tensors[bias_tensor]->getTensorImpl();
        uniform.running_mean   = tensors[running_mean]->getTensorImpl();
        uniform.running_var    = tensors[running_var]->getTensorImpl();
        uniform.out_tensor     = tensors[out_tensor]->getTensorImpl();

        // if no save_mean is provided, it is assumed to be eval mode
        if(!save_mean.empty()){
            uniform.save_mean      = tensors[save_mean]->getTensorImpl();
            uniform.save_var       = tensors[save_var]->getTensorImpl();
            uniform.mode           = 0; // 0 = train, 1 = eval
        }else {
            uniform.mode = 1; 
        }
        
        uniform.batch_size = batch_size;
        
        uniform.accumulate_grad = 1;    // default; caller can change if needed
        uniform.momentum = 0.1f;
        uniform.eps = 1e-05;

        // dispatch grid calculation
        auto ceilDiv = [](uint32_t a, uint32_t b) { return (a + b - 1) / b; };

        uint32_t dispatchM = M;
        uint32_t dispatchN = N;

        uint32_t groupX = M;
        uint32_t groupY = N;
        uint32_t groupZ = 1;

        DEBUG_PRINT("BatchNorm dispatch: " << groupX << " x " << groupY << " x " << groupZ
                << " (covering " << dispatchM << "x" << dispatchN << " per batch)");

        uint32_t workgroup[3] = { groupX, groupY, groupZ };

        tensor_push_const pushConsts{};
        pushConsts.grid_size = { groupX, groupY, groupZ };

        // choose shader based on mode
        if (mode == 0) {
            pushConsts.mode = 0; // forward
            pushConsts.uniformAddress = batchnormShader.uniformBuffer->getBufferAddress();
            batchnormShader.loadUniform(uniform, pushConsts);
            batchnormShader.execute(workgroup);
        } else {
            pushConsts.mode = 1; // backward
            pushConsts.uniformAddress = batchnormShaderBackward.uniformBuffer->getBufferAddress();
            batchnormShaderBackward.loadUniform(uniform, pushConsts);
            batchnormShaderBackward.execute(workgroup);
        }
    }

    void tensor_batchnorm_2d(const std::string& input_tensor,
                          const std::string& weight_tensor,
                          const std::string& bias_tensor,
                          const std::string& running_mean,
                          const std::string& running_var,
                          const std::string& out_tensor,
                          const std::string& save_mean,
                          const std::string& save_var,
                          uint32_t mode = 0, // 0 = forward, 1 = backward
                          float momentum = 0.1f,
                          float eps = 1e-05f)
    {
        // Validate tensor existence
        //if (tensors.find(input_tensor) == tensors.end() ||
        //    tensors.find(weight_tensor) == tensors.end() ||
        //    tensors.find(bias_tensor) == tensors.end() ||
        //    tensors.find(running_mean) == tensors.end() ||
        //    tensors.find(running_var) == tensors.end() ||
        //    tensors.find(out_tensor) == tensors.end() ||
        //    tensors.find(save_mean) == tensors.end() ||
        //    tensors.find(save_var) == tensors.end()) {
        //    throw std::invalid_argument("One or more tensors not found for BatchNorm2D");
        //}

        const auto& inShape = tensors[input_tensor]->shape;
        
        // Validate input shape: expect [B, C, H, W] (rank 4)
        if (inShape.size() != 4) {
            throw std::invalid_argument("BatchNorm2D expects input shape [B, C, H, W] (rank 4), got rank " + 
                                    std::to_string(inShape.size()));
        }

        uint32_t B = inShape[0]; // batch size
        uint32_t C = inShape[1]; // channels
        uint32_t H = inShape[2]; // height
        uint32_t W = inShape[3]; // width

        // Validate parameter shapes: weight, bias, running_mean, running_var should all be [C]
        const auto& weightShape = tensors[weight_tensor]->shape;
        const auto& biasShape = tensors[bias_tensor]->shape;
        const auto& runningMeanShape = tensors[running_mean]->shape;
        const auto& runningVarShape = tensors[running_var]->shape;

        if (weightShape.size() != 1 || weightShape[0] != C) {
            throw std::invalid_argument("Weight tensor must have shape [C=" + std::to_string(C) + "]");
        }
        if (biasShape.size() != 1 || biasShape[0] != C) {
            throw std::invalid_argument("Bias tensor must have shape [C=" + std::to_string(C) + "]");
        }
        if (runningMeanShape.size() != 1 || runningMeanShape[0] != C) {
            throw std::invalid_argument("Running mean tensor must have shape [C=" + std::to_string(C) + "]");
        }
        if (runningVarShape.size() != 1 || runningVarShape[0] != C) {
            throw std::invalid_argument("Running var tensor must have shape [C=" + std::to_string(C) + "]");
        }

        // Validate output shape has same number of elements as input
        const auto& outShape = tensors[out_tensor]->shape;
        const size_t inputElements = std::accumulate(inShape.begin(), inShape.end(), 
                                                    1ULL, std::multiplies<size_t>());
        const size_t outputElements = std::accumulate(outShape.begin(), outShape.end(), 
                                                    1ULL, std::multiplies<size_t>());

        if (inputElements != outputElements) {
            throw std::invalid_argument(
                "Output tensor must have same number of elements as input (got " +
                std::to_string(outputElements) + ", expected " + 
                std::to_string(inputElements) + ")"
            );
        }

        // Fill context structure
        tensor_batchnorm2d_context uniform{};
        
        uniform.input_tensor = tensors[input_tensor]->getTensorImpl();
        uniform.weight_tensor = tensors[weight_tensor]->getTensorImpl();
        uniform.bias_tensor = tensors[bias_tensor]->getTensorImpl();
        uniform.running_mean = tensors[running_mean]->getTensorImpl();
        uniform.running_var = tensors[running_var]->getTensorImpl();
        uniform.out_tensor = tensors[out_tensor]->getTensorImpl();

        // if no save_mean is provided, it is assumed to be eval mode
        if (!save_mean.empty()){
            uniform.save_mean = tensors[save_mean]->getTensorImpl();
            uniform.save_var = tensors[save_var]->getTensorImpl();
            uniform.mode = 0; // 0 = training (forward), 1 = evaluation (forward)
        }
        else {
            uniform.mode = 1;
        }
        
        uniform.batch_size = B;
        uniform.channels = C;
        uniform.height = H;
        uniform.width = W;
        
        uniform.accumulate_grad = 1; // default; can be set by caller if needed
        uniform.momentum = momentum;
        uniform.eps = eps;

        // Dispatch grid: one workgroup per channel
        // Each workgroup has 256 threads that cooperate to process all B*H*W elements for one channel
        uint32_t groupX = C; // One workgroup per channel
        uint32_t groupY = 1;
        uint32_t groupZ = 1;

        DEBUG_PRINT("BatchNorm2D dispatch: " << groupX << " x " << groupY << " x " << groupZ
                    << " (B=" << B << ", C=" << C << ", H=" << H << ", W=" << W 
                    << ", spatial_size=" << (H * W) << ", mode=" << mode << ")");

        uint32_t workgroup[3] = { groupX, groupY, groupZ };
        
        tensor_push_const pushConsts{};
        pushConsts.grid_size = { groupX, groupY, groupZ };

        // Choose shader based on mode
        if (mode == 0) {
            // Forward pass
            pushConsts.mode = 0; // forward
            pushConsts.uniformAddress = batchnorm2dShader.uniformBuffer->getBufferAddress();
            batchnorm2dShader.loadUniform(uniform, pushConsts);
            batchnorm2dShader.execute(workgroup);
        } else {
            // Backward pass
            pushConsts.mode = 1; // backward
            pushConsts.uniformAddress = batchnorm2dShaderBackward.uniformBuffer->getBufferAddress();
            batchnorm2dShaderBackward.loadUniform(uniform, pushConsts);
            batchnorm2dShaderBackward.execute(workgroup);
        }
    }

    void tensor_layernorm(const std::string& input_tensor,
                         const std::string& weight_tensor,
                         const std::string& bias_tensor,
                         const std::string& out_tensor,
                         const std::string& save_mean,
                         const std::string& save_rstd,
                         uint32_t mode = 0) // 0 = forward, 1 = backward
    {
        // Validate existence & get shapes
        try {
            const auto& inShape = tensors[input_tensor]->shape;
            const auto& wShape = tensors[weight_tensor]->shape;
            const auto& bShape = tensors[bias_tensor]->shape;
            
            if (inShape.empty() || wShape.empty() || bShape.empty()) {
                throw std::invalid_argument("Tensors cannot be empty");
            }
            
            // Weight and bias shapes must match
            if (wShape != bShape) {
                throw std::invalid_argument("Weight and bias shapes must match");
            }
        }
        catch (const std::invalid_argument& e) {
            std::cerr << "Error in tensor_layernorm_1d: " << e.what() << "\n";
            throw;
        }

        tensor_layernorm1d_context uniform{};

        auto prod = [](const std::vector<uint32_t>& v, size_t l, size_t r)->uint32_t {
            uint64_t p = 1; 
            for (size_t i = l; i < r; ++i) p *= v[i];
            if (p > UINT32_MAX) throw std::overflow_error("Product too large");
            return (uint32_t)p;
        }; 

        const auto& inShape = tensors[input_tensor]->shape;
        const auto& normShape = tensors[weight_tensor]->shape;

        if (inShape.size() < normShape.size() + 1) {
            throw std::invalid_argument("Input rank must be >= weight rank + 1 (need [B, ...])");
        }

        // Batch is always first dimension
        uint32_t batch_size = inShape[0];
        
        // Calculate normalized_size (product of dimensions we're normalizing over)
        uint32_t normalized_size = prod(normShape, 0, normShape.size());
        
        // Calculate number of separate sequences to normalize
        // This is batch_size * product of remaining dimensions not being normalized
        uint32_t num_samples = batch_size;
        for (size_t i = 1; i < inShape.size() - normShape.size(); i++) {
            num_samples *= inShape[i];
        }
        
        // Stride between batch elements
        uint32_t batch_stride = prod(inShape, 1, inShape.size());

        // Fill uniform/context expected by shader
        uniform.input_tensor   = tensors[input_tensor]->getTensorImpl();
        uniform.weight_tensor  = tensors[weight_tensor]->getTensorImpl();
        uniform.bias_tensor    = tensors[bias_tensor]->getTensorImpl();
        uniform.out_tensor     = tensors[out_tensor]->getTensorImpl();
        uniform.batch_size = batch_size;

        if(!save_mean.empty()){
            uniform.save_mean      = tensors[save_mean]->getTensorImpl();
            uniform.save_rstd      = tensors[save_rstd]->getTensorImpl();
            uniform.mode = 0;
        }else {
            uniform.mode = 1;
        }

        uniform.normalized_size = normalized_size;
        uniform.batch_stride    = batch_stride;
        uniform.accumulate_grad = 1;
        uniform.eps = 1e-05f;

        // Each workgroup processes one sequence
        uint32_t groupX = num_samples;
        uint32_t groupY = 1;
        uint32_t groupZ = 1;

        DEBUG_PRINT("LayerNorm dispatch: " << groupX << " x " << groupY << " x " << groupZ
                << " (normalizing " << num_samples << " sequences of size " << normalized_size << ")");

        uint32_t workgroup[3] = { groupX, groupY, groupZ };

        tensor_push_const pushConsts{};
        pushConsts.grid_size = { batch_size, groupY, groupZ };

        if (mode == 0) {
            pushConsts.mode = 0;
            pushConsts.uniformAddress = layernorm1dShader.uniformBuffer->getBufferAddress();
            layernorm1dShader.loadUniform(uniform, pushConsts);
            layernorm1dShader.execute(workgroup);
        } else {
            pushConsts.mode = 1;
            pushConsts.uniformAddress = layernorm1dShaderBackward.uniformBuffer->getBufferAddress();
            layernorm1dShaderBackward.loadUniform(uniform, pushConsts);
            layernorm1dShaderBackward.execute(workgroup);
        }
    }

    // tensor_c = tensor_a + tensor_b or tensor_a = tensor_a + tensor_b
    void tensor_add_inplace(const std::string& tensor_b, const std::string& tensor_a, const std::string& tensor_c = "", uint32_t mode = 0) {
        matadd_inplace_context uniform{};
        
        // Helpers
        auto prod = [](const std::vector<uint32_t>& v, size_t l, size_t r)->uint32_t {
            uint64_t p = 1; 
            for (size_t i = l; i < r; ++i) p *= v[i];
            if (p > UINT32_MAX) throw std::overflow_error("Batch too large");
            return (uint32_t)p;
        };

        const auto& shapeA = tensors[tensor_a]->shape; // A[..., M, N]
        const auto& shapeB = tensors[tensor_b]->shape; // B[..., M, N] (in-place output)
        
        // last-2 dims
        uint32_t M = shapeB.size() >= 2 ? shapeB[shapeB.size() - 2] : 1;
        uint32_t N = shapeB.back();
        
        // flatten all leading dims -> B
        uint32_t batch_size = prod(shapeB, 0, shapeB.size() - 2);
        
        // uniforms
        uniform.input_a = tensors[tensor_a]->getTensorImpl();
        uniform.input_b = tensors[tensor_b]->getTensorImpl(); // in-place output
        if(!tensor_c.empty()){
            if(tensors[tensor_a]->shape != tensors[tensor_c]->shape) throw std::invalid_argument("Only tensor_b can be broadcasted. Tensor_a and c must have the same shapes");
            uniform.input_c = tensors[tensor_c]->getTensorImpl();
            uniform.mode = 1;
        }else {
            uniform.mode = 0;
        }
        uniform.batch_size = batch_size;
        uniform.m = M;
        uniform.n = N;
        uniform.accumulate_grad = 1;
        
        // Compute dispatch grid
        // For broadcasting support in shader: pick the tensor with the most elements
        uint32_t num_elements_a = tensors[tensor_a]->get_num_elements();
        uint32_t num_elements_b = tensors[tensor_b]->get_num_elements();
        uint32_t max_elements = std::max(num_elements_a, num_elements_b);

        auto cielDiv = [](uint32_t a, uint32_t b) { return (a + b - 1) / b; };

        uint32_t groupX = cielDiv(max_elements, 256u);
        uint32_t groupY = 1;
        uint32_t groupZ = 1;

        DEBUG_PRINT("Dispatch (in-place, broadcasting): ");
        DEBUG_PRINT(groupX << " × " << groupY
                    << " × " << groupZ << " (covering max elements " << max_elements << ")");

        uint32_t workgroup[3] = { groupX, groupY, groupZ };
        
        tensor_push_const pushConsts{};
        pushConsts.grid_size = { groupX, groupY, groupZ };
        
        if (mode == 0) {
            // Forward pass
            pushConsts.mode = mode;
            pushConsts.uniformAddress = inplaceAdditionShader.uniformBuffer->getBufferAddress();
            inplaceAdditionShader.loadUniform(uniform, pushConsts);
            inplaceAdditionShader.execute(workgroup);
        }
        else if (mode == 1) {
            // Backward pass
            pushConsts.mode = mode;
            pushConsts.uniformAddress = inplaceAdditionShaderBackward.uniformBuffer->getBufferAddress();
            inplaceAdditionShaderBackward.loadUniform(uniform, pushConsts);
            inplaceAdditionShaderBackward.execute(workgroup);
        }
    }

    // tensor_c = tensor_a * tensor_b or tensor_a = tensor_a * tensor_b
    void tensor_multiply_elementwise(const std::string& tensor_b, const std::string& tensor_a, const std::string& tensor_c = "", uint32_t mode = 0) {
        matmul_elementwise_context uniform{};
        
        // Helpers
        auto prod = [](const std::vector<uint32_t>& v, size_t l, size_t r)->uint32_t {
            uint64_t p = 1; 
            for (size_t i = l; i < r; ++i) p *= v[i];
            if (p > UINT32_MAX) throw std::overflow_error("Batch too large");
            return (uint32_t)p;
        };

        const auto& shapeA = tensors[tensor_a]->shape; // A[..., M, N]
        const auto& shapeB = tensors[tensor_b]->shape; // B[..., M, N]
        
        // last-2 dims
        uint32_t M = shapeB.size() >= 2 ? shapeB[shapeB.size() - 2] : 1;
        uint32_t N = shapeB.back();
        
        // flatten all leading dims -> B
        uint32_t batch_size = prod(shapeB, 0, shapeB.size() - 2);
        
        // uniforms
        uniform.input_a = tensors[tensor_a]->getTensorImpl();
        uniform.input_b = tensors[tensor_b]->getTensorImpl();
        if(!tensor_c.empty()){
            uniform.input_c = tensors[tensor_c]->getTensorImpl();
            uniform.mode = 1;
        }else {
            uniform.mode = 0;
        }
        uniform.batch_size = batch_size;
        uniform.m = M;
        uniform.n = N;
        uniform.accumulate_grad = 1;
        
        // Compute dispatch grid
        // For broadcasting support in shader: pick the tensor with the most elements
        uint32_t num_elements_a = tensors[tensor_a]->get_num_elements();
        uint32_t num_elements_b = tensors[tensor_b]->get_num_elements();
        uint32_t max_elements = std::max(num_elements_a, num_elements_b);

        auto cielDiv = [](uint32_t a, uint32_t b) { return (a + b - 1) / b; };

        uint32_t groupX = cielDiv(max_elements, 256u);
        uint32_t groupY = 1;
        uint32_t groupZ = 1;

        DEBUG_PRINT("Dispatch (in-place, broadcasting): ");
        DEBUG_PRINT(groupX << " × " << groupY
                    << " × " << groupZ << " (covering max elements " << max_elements << ")");

        uint32_t workgroup[3] = { groupX, groupY, groupZ };
        
        tensor_push_const pushConsts{};
        pushConsts.grid_size = { groupX, groupY, groupZ };
        
        if (mode == 0) {
            // Forward pass
            pushConsts.mode = mode;
            pushConsts.uniformAddress = elementwiseMultiplicationShader.uniformBuffer->getBufferAddress();
            elementwiseMultiplicationShader.loadUniform(uniform, pushConsts);
            elementwiseMultiplicationShader.execute(workgroup);
        }
        else if (mode == 1) {
            // Backward pass
            pushConsts.mode = mode;
            pushConsts.uniformAddress = elementwiseMultiplicationShaderBackward.uniformBuffer->getBufferAddress();
            elementwiseMultiplicationShaderBackward.loadUniform(uniform, pushConsts);
            elementwiseMultiplicationShaderBackward.execute(workgroup);
        }
    }

    // computes std = exp(0.5 * logvar)
    Tensor<T>& tensor_logvar_to_std(const std::string& logvar_tensor, std::string std_tensor = "", uint32_t mode = 0) {
        logvar_to_std_context uniform{};
        
        // Helpers
        auto prod = [](const std::vector<uint32_t>& v, size_t l, size_t r)->uint32_t {
            uint64_t p = 1; 
            for (size_t i = l; i < r; ++i) p *= v[i];
            if (p > UINT32_MAX) throw std::overflow_error("Batch too large");
            return (uint32_t)p;
        };

        // if no output is given, create output. If created output already exists, use that
        Tensor<T>* output;
        if(std_tensor.empty()){
            output = &createTensor(tensors[logvar_tensor]->shape, logvar_tensor + "-std_dev_tensor");
            std_tensor = output->name;
        }

        const auto& shapeA = tensors[logvar_tensor]->shape; // A[..., M, N]
        const auto& shapeB = tensors[std_tensor]->shape; // B[..., M, N]
        
        // last-2 dims
        uint32_t M = shapeB.size() >= 2 ? shapeB[shapeB.size() - 2] : 1;
        uint32_t N = shapeB.back();
        
        // flatten all leading dims -> B
        uint32_t batch_size = prod(shapeB, 0, shapeB.size() - 2);
        
        // uniforms
        uniform.logvar = tensors[logvar_tensor]->getTensorImpl();
        uniform.std_out = tensors[std_tensor]->getTensorImpl(); // output
        uniform.accumulate_grad = 1;
        
        // Compute dispatch grid
        uint32_t num_elements_a = tensors[logvar_tensor]->get_num_elements();

        auto cielDiv = [](uint32_t a, uint32_t b) { return (a + b - 1) / b; };

        uint32_t groupX = cielDiv(num_elements_a, 256u);
        uint32_t groupY = 1;
        uint32_t groupZ = 1;

        DEBUG_PRINT("Dispatch (in-place, broadcasting): ");
        DEBUG_PRINT(groupX << " × " << groupY
                    << " × " << groupZ << " (covering max elements " << num_elements_a << ")");

        uint32_t workgroup[3] = { groupX, groupY, groupZ };
        
        tensor_push_const pushConsts{};
        pushConsts.grid_size = { groupX, groupY, groupZ };
        
        if (mode == 0) {
            // Forward pass
            pushConsts.mode = mode;
            pushConsts.uniformAddress = logvarToStdShader.uniformBuffer->getBufferAddress();
            logvarToStdShader.loadUniform(uniform, pushConsts);
            logvarToStdShader.execute(workgroup);
            return *output;
        }
        else if (mode == 1) {
            // Backward pass
            pushConsts.mode = mode;
            pushConsts.uniformAddress = logvarToStdShaderBackward.uniformBuffer->getBufferAddress();
            logvarToStdShaderBackward.loadUniform(uniform, pushConsts);
            logvarToStdShaderBackward.execute(workgroup);
            return *output;
        }
    }

    // computes output = exp(input)
    Tensor<T>& tensor_exp(const std::string& input_tensor, std::string output_tensor = "", uint32_t mode = 0) {
        logvar_to_std_context uniform{};
        
        // Helpers
        auto prod = [](const std::vector<uint32_t>& v, size_t l, size_t r)->uint32_t {
            uint64_t p = 1; 
            for (size_t i = l; i < r; ++i) p *= v[i];
            if (p > UINT32_MAX) throw std::overflow_error("Batch too large");
            return (uint32_t)p;
        };

        // if no output is given, create output. If created output already exists, use that
        Tensor<T>* output;
        if(output_tensor.empty()){
            output = &createTensor(tensors[input_tensor]->shape, input_tensor + "-std_dev_tensor");
            output_tensor = output->name;
        }

        const auto& shapeA = tensors[input_tensor]->shape; // A[..., M, N]
        const auto& shapeB = tensors[output_tensor]->shape; // B[..., M, N]
        
        // last-2 dims
        uint32_t M = shapeB.size() >= 2 ? shapeB[shapeB.size() - 2] : 1;
        uint32_t N = shapeB.back();
        
        // flatten all leading dims -> B
        uint32_t batch_size = prod(shapeB, 0, shapeB.size() - 2);
        
        // uniforms
        uniform.logvar = tensors[input_tensor]->getTensorImpl();
        uniform.std_out = tensors[output_tensor]->getTensorImpl(); // output
        uniform.accumulate_grad = 1;
        
        // Compute dispatch grid
        uint32_t num_elements_a = tensors[input_tensor]->get_num_elements();

        auto cielDiv = [](uint32_t a, uint32_t b) { return (a + b - 1) / b; };

        uint32_t groupX = cielDiv(num_elements_a, 256u);
        uint32_t groupY = 1;
        uint32_t groupZ = 1;

        DEBUG_PRINT("Dispatch (in-place, broadcasting): ");
        DEBUG_PRINT(groupX << " × " << groupY
                    << " × " << groupZ << " (covering max elements " << num_elements_a << ")");

        uint32_t workgroup[3] = { groupX, groupY, groupZ };
        
        tensor_push_const pushConsts{};
        pushConsts.grid_size = { groupX, groupY, groupZ };
        
        if (mode == 0) {
            // Forward pass
            pushConsts.mode = mode;
            pushConsts.uniformAddress = expShader.uniformBuffer->getBufferAddress();
            expShader.loadUniform(uniform, pushConsts);
            expShader.execute(workgroup);
            return *output;
        }
        else if (mode == 1) {
            // Backward pass
            pushConsts.mode = mode;
            pushConsts.uniformAddress = expShaderBackward.uniformBuffer->getBufferAddress();
            expShaderBackward.loadUniform(uniform, pushConsts);
            expShaderBackward.execute(workgroup);
            return *output;
        }
    }

    void tensor_linear_ReLU(const std::string& output_tensor,
        const std::string& input_tensor,
        const std::string& weight_tensor,
        const std::string& bias_tensor = "",
        uint32_t mode = 0) {

        linear_context uniform{};
        uniform.input_tensor = tensors[input_tensor]->getTensorImpl();
        uniform.weight_tensor = tensors[weight_tensor]->getTensorImpl();
        if (!bias_tensor.empty()) {
            uniform.bias_tensor = tensors[bias_tensor]->getTensorImpl();
            uniform.use_bias = 1;
        } else {
            uniform.use_bias = 0;
        }
        uniform.output_tensor = tensors[output_tensor]->getTensorImpl();
        uniform.mode = 0; // Tiling mode

        auto prod = [](const std::vector<uint32_t>& v, size_t l, size_t r)->uint32_t {
            uint64_t p = 1;
            for (size_t i = l; i < r; ++i) p *= v[i];
            if (p > UINT32_MAX) throw std::overflow_error("Batch too large");
            return static_cast<uint32_t>(p);
        };

        // Input shape X[..., M, K]
        const auto& shape = tensors[input_tensor]->shape;
        if (shape.size() < 2)
            throw std::invalid_argument("Rank must be >= 2");
        uint32_t M = shape[shape.size() - 2];
        uint32_t K = shape.back();

        // Weight shape W[..., N, K] (transposed)
        const auto& weight_shape = tensors[weight_tensor]->shape;
        if (weight_shape.size() < 2)
            throw std::invalid_argument("Weight rank must be >= 2");
        uint32_t KB = weight_shape.back();
        uint32_t N = weight_shape[weight_shape.size() - 2];
        if (KB != K)
            throw std::invalid_argument("Inner dims mismatch: X[...,M,K] @ W[...,N,K] (transposed)");

        uint32_t batch_size = prod(shape, 0, shape.size() - 2);

        DEBUG_PRINT("Linear'ing tensor " << input_tensor << " of shape "
            << shapeToString(shape) << " with transposed weights " << weight_tensor
            << " of shape " << shapeToString(weight_shape));

        uniform.m = M;
        uniform.n = N;
        uniform.k = K;
        uniform.batch_size = batch_size;
        uniform.accumulate_grad = 1; // add

        // --- Warp-tiling parameters (must match shader) ---
        constexpr uint32_t BM = 128;  // Block tile size M
        constexpr uint32_t BN = 128;  // Block tile size N
        constexpr uint32_t BK = 8;    // Block tile size K
        constexpr uint32_t NUM_THREADS = 256;  // Threads per workgroup

        // --- Compute dispatch grid based on mode ---
        auto ceilDiv = [](uint32_t a, uint32_t b) { return (a + b - 1) / b; };

        uint32_t dispatchM, dispatchN, dispatchBatch;

        if (mode == 0) {
            // Mode 0: Use output tensor dimensions
            const auto& output_shape = tensors[output_tensor]->shape;
            if (output_shape.size() < 2)
                throw std::invalid_argument("Output rank must be >= 2");

            dispatchM = output_shape[output_shape.size() - 2];
            dispatchN = output_shape[output_shape.size() - 1];
            dispatchBatch = prod(output_shape, 0, output_shape.size() - 2);

            DEBUG_PRINT("Dispatch (output dims): ");
        }
        else {
            // Mode 1: Use max dimensions of all tensors
            auto maxDim = [](const std::vector<uint32_t>& a,
                const std::vector<uint32_t>& b) {
                    size_t L = std::max(a.size(), b.size());
                    std::vector<uint32_t> res(L, 1);
                    for (size_t i = 0; i < L; i++) {
                        uint32_t av = (i < a.size() ? a[i] : 1);
                        uint32_t bv = (i < b.size() ? b[i] : 1);
                        res[i] = std::max(av, bv);
                    }
                    return res;
                };

            std::vector<uint32_t> maxShape = tensors[input_tensor]->shape;
            auto transposedWeightShape = tensors[weight_tensor]->shape;
            if (transposedWeightShape.size() >= 2) {
                std::swap(transposedWeightShape[transposedWeightShape.size() - 1], 
                         transposedWeightShape[transposedWeightShape.size() - 2]);
            }
            maxShape = maxDim(maxShape, transposedWeightShape);
            if (!bias_tensor.empty())
                maxShape = maxDim(maxShape, tensors[bias_tensor]->shape);
            maxShape = maxDim(maxShape, tensors[output_tensor]->shape);

            if (maxShape.size() < 2)
                throw std::invalid_argument("Max rank must be >= 2");

            dispatchM = maxShape[maxShape.size() - 2];
            dispatchN = maxShape[maxShape.size() - 1];
            dispatchBatch = prod(maxShape, 0, maxShape.size() - 2);

            DEBUG_PRINT("Dispatch (max dims): ");
        }

        // Calculate workgroup dimensions based on warp-tiling parameters
        // Each workgroup processes a BM x BN tile with NUM_THREADS threads
        uint32_t groupX = ceilDiv(dispatchN, BN);
        uint32_t groupY = ceilDiv(dispatchM, BM);
        uint32_t groupZ = dispatchBatch;

        DEBUG_PRINT(groupX << " × " << groupY
            << " × " << groupZ << " workgroups (covering " << dispatchM << "×"
            << dispatchN << " per batch)" << "Workgroup size: " << NUM_THREADS << " threads (" 
            << BM << "×" << BN << " tile)");

        uint32_t workgroup[3] = { groupX, groupY, groupZ };

        tensor_push_const pushConsts{};
        pushConsts.grid_size = { groupX, groupY, groupZ };

        if (mode == 0) {
            pushConsts.mode = mode;
            uniform.weight_tensor = tensors[weight_tensor]->getTensorImpl();
            pushConsts.uniformAddress = linearReLUShader.uniformBuffer->getBufferAddress();
            linearReLUShader.loadUniform(uniform, pushConsts);
            linearReLUShader.execute(workgroup);
        }
        else if (mode == 1) {
            pushConsts.mode = mode;
            pushConsts.uniformAddress = linearReLUShaderBackward.uniformBuffer->getBufferAddress();
            linearReLUShaderBackward.loadUniform(uniform, pushConsts);
            linearReLUShaderBackward.execute(workgroup);
        }
    }

    void tensor_linear(const std::string& output_tensor,
        const std::string& input_tensor,
        const std::string& weight_tensor,
        const std::string& bias_tensor = "",
        uint32_t mode = 0) {
        linear_context uniform{};
        uniform.input_tensor = tensors[input_tensor]->getTensorImpl();
        uniform.weight_tensor = tensors[weight_tensor]->getTensorImpl();
        if (!bias_tensor.empty()) {
            uniform.bias_tensor = tensors[bias_tensor]->getTensorImpl();
            uniform.use_bias = 1;
        } else {
            uniform.use_bias = 0;
        }
        uniform.output_tensor = tensors[output_tensor]->getTensorImpl();
        uniform.mode = 0; // Tiling mode

        auto prod = [](const std::vector<uint32_t>& v, size_t l, size_t r)->uint32_t {
            uint64_t p = 1;
            for (size_t i = l; i < r; ++i) p *= v[i];
            if (p > UINT32_MAX) throw std::overflow_error("Batch too large");
            return static_cast<uint32_t>(p);
        };

        // Input shape X[..., M, K]
        const auto& shape = tensors[input_tensor]->shape;
        if (shape.size() < 2)
            throw std::invalid_argument("Rank must be >= 2");
        uint32_t M = shape[shape.size() - 2];
        uint32_t K = shape.back();

        // Weight shape W[..., N, K] (transposed)
        const auto& weight_shape = tensors[weight_tensor]->shape;
        if (weight_shape.size() < 2)
            throw std::invalid_argument("Weight rank must be >= 2");
        uint32_t KB = weight_shape.back();
        uint32_t N = weight_shape[weight_shape.size() - 2];
        if (KB != K)
            throw std::invalid_argument("Inner dims mismatch: X[...,M,K] @ W[...,N,K] (transposed)");

        uint32_t batch_size = prod(shape, 0, shape.size() - 2);

        DEBUG_PRINT("Linear'ing tensor " << input_tensor << " of shape "
            << shapeToString(shape) << " with transposed weights " << weight_tensor
            << " of shape " << shapeToString(weight_shape));

        uniform.m = M;
        uniform.n = N;
        uniform.k = K;
        uniform.batch_size = batch_size;
        uniform.accumulate_grad = 1; // add

        // --- Warp-tiling parameters (must match shader) ---
        constexpr uint32_t BM = 128;  // Block tile size M
        constexpr uint32_t BN = 128;  // Block tile size N
        constexpr uint32_t BK = 8;    // Block tile size K
        constexpr uint32_t NUM_THREADS = 256;  // Threads per workgroup

        // --- Compute dispatch grid based on mode ---
        auto ceilDiv = [](uint32_t a, uint32_t b) { return (a + b - 1) / b; };

        uint32_t dispatchM, dispatchN, dispatchBatch;

        if (mode == 0) {
            // Mode 0: Use output tensor dimensions
            const auto& output_shape = tensors[output_tensor]->shape;
            if (output_shape.size() < 2)
                throw std::invalid_argument("Output rank must be >= 2");

            dispatchM = output_shape[output_shape.size() - 2];
            dispatchN = output_shape[output_shape.size() - 1];
            dispatchBatch = prod(output_shape, 0, output_shape.size() - 2);

            DEBUG_PRINT("Dispatch (output dims): ");
        }
        else {
            // Mode 1: Use max dimensions of all tensors
            auto maxDim = [](const std::vector<uint32_t>& a,
                const std::vector<uint32_t>& b) {
                    size_t L = std::max(a.size(), b.size());
                    std::vector<uint32_t> res(L, 1);
                    for (size_t i = 0; i < L; i++) {
                        uint32_t av = (i < a.size() ? a[i] : 1);
                        uint32_t bv = (i < b.size() ? b[i] : 1);
                        res[i] = std::max(av, bv);
                    }
                    return res;
                };

            std::vector<uint32_t> maxShape = tensors[input_tensor]->shape;
            auto transposedWeightShape = tensors[weight_tensor]->shape;
            if (transposedWeightShape.size() >= 2) {
                std::swap(transposedWeightShape[transposedWeightShape.size() - 1], 
                         transposedWeightShape[transposedWeightShape.size() - 2]);
            }
            maxShape = maxDim(maxShape, transposedWeightShape);
            if (!bias_tensor.empty())
                maxShape = maxDim(maxShape, tensors[bias_tensor]->shape);
            maxShape = maxDim(maxShape, tensors[output_tensor]->shape);

            if (maxShape.size() < 2)
                throw std::invalid_argument("Max rank must be >= 2");

            dispatchM = maxShape[maxShape.size() - 2];
            dispatchN = maxShape[maxShape.size() - 1];
            dispatchBatch = prod(maxShape, 0, maxShape.size() - 2);

            DEBUG_PRINT("Dispatch (max dims): ");
        }

        // Calculate workgroup dimensions based on warp-tiling parameters
        // Each workgroup processes a BM x BN tile with NUM_THREADS threads
        uint32_t groupX = ceilDiv(dispatchN, BN);
        uint32_t groupY = ceilDiv(dispatchM, BM);
        uint32_t groupZ = dispatchBatch;

        DEBUG_PRINT(groupX << " × " << groupY
            << " × " << groupZ << " workgroups (covering " << dispatchM << "×"
            << dispatchN << " per batch)" << "Workgroup size: " << NUM_THREADS << " threads (" 
            << BM << "×" << BN << " tile)");

        uint32_t workgroup[3] = { groupX, groupY, groupZ };

        tensor_push_const pushConsts{};
        pushConsts.grid_size = { groupX, groupY, groupZ };

        if (mode == 0) {
            pushConsts.mode = mode;
            uniform.weight_tensor = tensors[weight_tensor]->getTensorImpl();
            pushConsts.uniformAddress = linearShader.uniformBuffer->getBufferAddress();
            linearShader.loadUniform(uniform, pushConsts);
            linearShader.execute(workgroup);
        }
        else if (mode == 1) {
            pushConsts.mode = mode;
            pushConsts.uniformAddress = linearShaderBackward.uniformBuffer->getBufferAddress();
            linearShaderBackward.loadUniform(uniform, pushConsts);
            linearShaderBackward.execute(workgroup);
        }
    }

    // automatically populates predicted tensor's gradient buffer with the single call. No need to call .backward()
    void mse_loss(const std::string& predicted, const std::string& target, const std::string& loss){
        mse_loss_context uniform;
        uniform.loss_tensor = tensors[loss]->getTensorImpl();
        uniform.predicted_tensor = tensors[predicted]->getTensorImpl();
        uniform.target_tensor = tensors[target]->getTensorImpl();

        // num_elements per batch element
        constexpr uint32_t VEC_TILE_SIZE = 4;  // As defined in shader
        constexpr uint32_t ELEMENTS_PER_VEC4 = 4;
        constexpr uint32_t TILE = VEC_TILE_SIZE * ELEMENTS_PER_VEC4;  // 16 elements per thread
        constexpr uint32_t GRP = 256;

        uint32_t num_elements = std::accumulate(
            tensors[target]->shape.begin() + 1, 
            tensors[target]->shape.end(), 
            1, 
            std::multiplies<uint32_t>()
        );

        uint32_t batch_size = tensors[target]->shape[0];

        uniform.batch_size = batch_size;
        uniform.elements_per_batch = num_elements;

        // Each workgroup has GRP threads, each processing TILE elements
        uint32_t num_workgroups_x = (num_elements + (GRP * TILE) - 1) / (GRP * TILE);
        
        uint32_t workgroup[3] = {num_workgroups_x, 1, batch_size};

        DEBUG_PRINT("MSE loss on tensor: " << predicted << " Using targets: " << target << " with dispatch: " << "(" << num_workgroups_x << ", " << "1, " << batch_size << ")");
        tensor_push_const push;
        push.uniformAddress = mseLossShader.uniformBuffer->getBufferAddress();
        mseLossShader.loadUniform(uniform, push);
        mseLossShader.execute(workgroup);
    }

    void kld_loss(const std::string& mu_tensor, const std::string& logvar_tensor, const std::string& loss_tensor){
        kld_loss_context uniform;
        uniform.batch_size = tensors[mu_tensor]->shape[0];
        uniform.logvar_tensor = tensors[logvar_tensor]->getTensorImpl();
        uniform.mu_tensor = tensors[mu_tensor]->getTensorImpl();
        uniform.elements_per_batch = std::accumulate(tensors[mu_tensor]->shape.begin() + 1, tensors[mu_tensor]->shape.end(), 1, std::multiplies<uint32_t>());
        uniform.loss_tensor = tensors[loss_tensor]->getTensorImpl();
        uniform.beta = 1.0f;

        uint32_t grpx = (uniform.elements_per_batch + (256 * 4) - 1)/(256 * 4);
        uint32_t wrkgrp[3] = {grpx, 1, uniform.batch_size};

        DEBUG_PRINT("KLD loss on tensor: " << mu_tensor << ", " << logvar_tensor << " with dispatch: (" << grpx << ", " << "1, " << uniform.batch_size << ")");

        tensor_push_const push;
        push.mode = 0;
        push.uniformAddress = kldLossShader.uniformBuffer->getBufferAddress();
        kldLossShader.loadUniform(uniform, push);
        kldLossShader.execute(wrkgrp);
    }

    void tensor_fill_random(const std::string& tensor, T min, T max) {
        tensor_fill_rand_uniform_address<T> uniform{};

		uniform.tensor = tensors[tensor]->getTensorImpl();
		uniform.type = std::is_floating_point<T>::value ? 0 : (std::is_integral<T>::value ? 1 : 2); // 0 = float, 1 = int, 2 = uint
        if (uniform.type == 0) {
            // For floating point types, we can use the min and max directly
            uniform.min = min;
            uniform.max = max;
        } else if (uniform.type == 1) {
            // For integral types, we need to cast min and max to T
            uniform.min = static_cast<T>(min);
            uniform.max = static_cast<T>(max);
        } else {
            // For unsigned types, we also cast
            uniform.min = static_cast<T>(min);
            uniform.max = static_cast<T>(max);
		}

        uniform.max = max;
        uniform.min = min;
        uniform.type = 0; // gaussian

        std::random_device rd; // random seed
        uniform.seed = static_cast<uint>(rd());
        
        tensor_push_const p;
        p.uniformAddress = fillRandomShader.uniformBuffer->getBufferAddress();

        uint32_t total_elements = std::accumulate(tensors[tensor]->shape.begin(), tensors[tensor]->shape.end(), 1, std::multiplies<uint32_t>());
		
        // Compute dispatch grid to cover the OUTPUT tensor C[B,M,N]
        auto ceilDiv = [](uint32_t a, uint32_t b) { return (a + b - 1) / b; };

        // Grid covers the output tensor's data buffer.
        uint32_t groupX = ceilDiv(total_elements, 256u);
        uint32_t groupY = 1u;
        uint32_t groupZ = 1u;
        
        p.grid_size = glm::uvec3(groupX, groupY, groupZ);
        fillRandomShader.loadUniform(uniform, p);
        std::array<uint32_t, 3> workgroup = { groupX, groupY, groupZ };

		DEBUG_PRINT("workgroup: " << workgroup[0] << " " << workgroup[1] << " " << workgroup[2] << "for tensor: " << tensor);

        fillRandomShader.execute(workgroup.data());
    }

    bool are_tensors_equal(const std::string& tensor_a, const std::string& tensor_b){
        auto &local_tensor = createTensor({1}, "temp_compare");
        tensor_cmp_context uniform{};
        uniform.input_a = tensors[tensor_a]->getTensorImpl();
        uniform.input_b = tensors[tensor_b]->getTensorImpl();
        uniform.output_tensor = local_tensor.getTensorImpl();

        uint32_t num_elements_a = tensors[tensor_a]->get_num_elements();
        uint32_t num_elements_b = tensors[tensor_b]->get_num_elements();

        uint32_t max_elements = std::max(num_elements_a, num_elements_b);
        
        tensor_push_const uniformPush{};
        uniformPush.grid_size = {1,1,1};
        uniformPush.mode = 0;
        uniformPush.uniformAddress = cmpShader.uniformBuffer->getBufferAddress();

        cmpShader.loadUniform(uniform, uniformPush);

        auto cieldiv = [](uint32_t a, uint32_t b) { return (a + b - 1) / b; };
        uint32_t groupX = cieldiv(max_elements, 256u);
        uint32_t grp[3] = {groupX, 1, 1};
        cmpShader.execute(grp);
        auto data = local_tensor.dataBuffer->downloadBuffer();
        if (data[0] == 0) {
            destroy_tensor(local_tensor.name);
            return true;
        }
        else {
            destroy_tensor(local_tensor.name);
            return false;
        }
    }

    T find_mean_of_tensor(const std::string& tensor){
        auto& local_tensor = createTensor({1}, "temp_find_mean");
        mean_context uniform;
        uniform.input_tensor = tensors[tensor]->getTensorImpl();
        uniform.output_tensor = local_tensor.getTensorImpl();
        uint32_t num_elem = tensors[tensor]->get_num_elements();
        DEBUG_PRINT("finding mean of tensor: " << tensor << " of shape: " << shapeToString(tensors[tensor]->shape));
        auto cieldiv = [](uint32_t a, uint32_t b) { return (a + b - 1) / b; };
        uint32_t grpX = cieldiv(num_elem, 256u);
        uint32_t grp[3] = {grpX, 1, 1};
        tensor_push_const push;
        push.uniformAddress = mean_shader.uniformBuffer->getBufferAddress();
        push.grid_size = glm::uvec3(1, 1, 1);
        mean_shader.loadUniform(uniform, push);
        mean_shader.execute(grp);
        auto data = local_tensor.dataBuffer->downloadBuffer();
        destroy_tensor(local_tensor.name);
        return data[0]; // the first element will contain the mean value
    }

    void save_tensor_to_file(const std::string& tensor_name, const std::string& filename) {
        if (tensors.find(tensor_name) == tensors.end()) {
            throw std::invalid_argument("Tensor " + tensor_name + " does not exist.");
        }
        tensors[tensor_name]->save_to_file(filename);
    }

    void save_all_tensors_to_files(const std::string& directory) {
        for (const auto& [name, tensor] : tensors) {
            std::string filename = directory + "/" + name + ".tnsr";
            tensor->save_to_file(filename);
        }
    }

    void createTensor(const std::string& file_name){
        auto tensor = std::make_unique<Tensor<T>>(file_name, allocator);
        tensors[tensor->name] = std::move(tensor);
    }

    void clear() {
        tensors.clear();
    };

private:
    // tensor operation shaders
    gpuTaskNoDesc<T, tensor_push_const, matadd_inplace_context> inplaceAdditionShader; // computes C = A + B or A = A + B, supports broadcasting
    gpuTaskNoDesc<T, tensor_push_const, matadd_inplace_context> inplaceAdditionShaderBackward;
    gpuTaskNoDesc<T, tensor_push_const, matmul_elementwise_context> elementwiseMultiplicationShader; // computes C = A * B or A = A * B, supports broadcasting
    gpuTaskNoDesc<T, tensor_push_const, matmul_elementwise_context> elementwiseMultiplicationShaderBackward;
    gpuTaskNoDesc<T, tensor_push_const, logvar_to_std_context> logvarToStdShader; // computes std = (0.5 * logvar).exp()
    gpuTaskNoDesc<T, tensor_push_const, logvar_to_std_context> logvarToStdShaderBackward; // computes std = (0.5 * logvar).exp() backward pass
    gpuTaskNoDesc<T, tensor_push_const, logvar_to_std_context> expShader; // computes output = (input).exp()
    gpuTaskNoDesc<T, tensor_push_const, logvar_to_std_context> expShaderBackward; // computes output = (input).exp() backward pass
    gpuTaskNoDesc<T, tensor_push_const, mse_loss_context> mseLossShader; // computes Mean Squared Error (MSE) loss
    gpuTaskNoDesc<T, tensor_push_const, kld_loss_context> kldLossShader; // computes the Kullback Liebler Divergence (KLD) loss
	gpuTaskNoDesc<T, tensor_push_const, tensor_relu_context> ReLUShader; // computes ReLU
    gpuTaskNoDesc<T, tensor_push_const, tensor_relu_context> ReLUShaderBackward;
	gpuTaskNoDesc<T, tensor_push_const, tensor_softmax_context> softmaxShader;  // computes softmax
	gpuTaskNoDesc<T, tensor_push_const, tensor_cross_entropy_context> crossEntropyShader; // computes softmax + cross-entropy loss
    gpuTaskNoDesc<T, tensor_push_const, tensor_cross_entropy_context> crossEntropyShaderBackward; // computes softmax + cross-entropy loss
	gpuTaskNoDesc<T, tensor_push_const, linear_context> linearShader; // computes linear layer
    gpuTaskNoDesc<T, tensor_push_const, linear_context> linearShaderBackward;
    gpuTaskNoDesc<T, tensor_push_const, linear_context> linearReLUShader; // computes linear layer + ReLU
    gpuTaskNoDesc<T, tensor_push_const, linear_context> linearReLUShaderBackward;
    gpuTaskNoDesc<T, tensor_push_const, tensor_batchnorm_context> batchnormShader; // computes batchnorm
    gpuTaskNoDesc<T, tensor_push_const, tensor_batchnorm_context> batchnormShaderBackward;
    gpuTaskNoDesc<T, tensor_push_const, tensor_batchnorm2d_context> batchnorm2dShader;
    gpuTaskNoDesc<T, tensor_push_const, tensor_batchnorm2d_context> batchnorm2dShaderBackward;
    gpuTaskNoDesc<T, tensor_push_const, tensor_layernorm1d_context> layernorm1dShader;
    gpuTaskNoDesc<T, tensor_push_const, tensor_layernorm1d_context> layernorm1dShaderBackward;
    gpuTaskNoDesc<T, tensor_push_const, embedding_lookup_context> embedLookupShader; // computes embedding lookup
    gpuTaskNoDesc<T, tensor_push_const, embedding_lookup_context> embedLookupShaderBackward;
    gpuTaskNoDesc<T, tensor_push_const, tensor_cmp_context> cmpShader; // compares two tensors for equality
    gpuTaskNoDesc<T, tensor_push_const, mean_context> mean_shader; // compute the mean of a tensor
    gpuTaskNoDesc<T, tensor_push_const, tensor_fill_rand_uniform_address<T>> fillRandomShader; // fills tensor with random values
    gpuTaskNoDesc<T, tensor_push_const, tensor_conv2d_3x3_context> conv2dShader3x3; // computes 2D convolution with 3x3 kernel
    gpuTaskNoDesc<T, tensor_push_const, tensor_conv2d_3x3_context> conv2dShader3x3Backward; // computes 2D convolution with 3x3 kernel backward pass
    gpuTaskNoDesc<T, tensor_push_const, tensor_conv2d_context> conv2dShader;    // generalized conv2d shader. Supports any kernel size up to 15x15
    gpuTaskNoDesc<T, tensor_push_const, tensor_conv2d_context> conv2dShaderBackward;    // generalized conv2d backward shader. Supports any kernel size up to 15x15
    gpuTaskNoDesc<T, tensor_push_const, tensor_transposed_conv2d_context> transposedConv2dShader;   // generalized transposed conv2d shader.
    gpuTaskNoDesc<T, tensor_push_const, tensor_transposed_conv2d_context> transposedConv2dShaderBackward;   // generalized transposed conv2d backward shader.
    gpuTaskNoDesc<T, tensor_push_const, tensor_max_pool_context> maxPoolShader;     // computes maxPool2D with any kernel shape
    gpuTaskNoDesc<T, tensor_push_const, tensor_max_pool_context> maxPoolShaderBackward;     // computes maxPool2D's backward pass with any kernel shape
};

template<typename T>
Tensor<T>& Tensor<T>::operator*(Tensor<T>& other) {
    auto &output = pool->createTensor(this->shape, name + other.name + "-elementwise_multiply_output");
    output.back = [this, &other, &output](){
        this->pool->tensor_multiply_elementwise(other.name, this->name, output.name, 1);
        this->backward();
        other.backward();
    };
    pool->tensor_multiply_elementwise(other.name, name, output.name);
    return output;
}

template<typename T>
Tensor<T>& Tensor<T>::operator*(T other){
    auto &output = pool->createTensor(this->shape, name + "-elementwise_multiply_output_for_" + std::to_string(other));
    auto &to_mul = pool->createTensor(std::vector<uint32_t>(this->shape.size(), 1), "const_mul_tensor_" + name + "_" + std::to_string(other));
    // set scalar value
    to_mul.dataBuffer->set(0, other);

    output.back = [this, &output, &to_mul](){
        this->pool->tensor_multiply_elementwise(to_mul.name, this->name, output.name, 1);
        this->backward();
    };
    pool->tensor_multiply_elementwise(to_mul.name, this->name, output.name);
    return output;
}

template<typename T>
Tensor<T>& operator*(T lhs, Tensor<T>& rhs){
    // create output with same shape as rhs
    auto &output = rhs.pool->createTensor(rhs.shape, std::to_string(lhs) + "_" + rhs.name + "-elementwise_multiply_output");
    // create scalar tensor to hold lhs
    auto &to_mul = rhs.pool->createTensor(std::vector<uint32_t>(rhs.shape.size(), 1), "const_mul_tensor_" + std::to_string(lhs) + "_" + rhs.name);
    to_mul.dataBuffer->set(0, lhs);

    output.back = [&rhs, &output, &to_mul](){
        rhs.pool->tensor_multiply_elementwise(to_mul.name, rhs.name, output.name, 1);
        rhs.backward();
    };

    rhs.pool->tensor_multiply_elementwise(to_mul.name, rhs.name, output.name);
    return output;
}

template<typename T>
Tensor<T>& Tensor<T>::operator+(Tensor<T>& other) {
    auto &output = pool->createTensor(this->shape, name + other.name + "-elementwise_addition_output");
    output.back = [this, &other, &output](){
        this->pool->tensor_add_inplace(other.name, this->name, output.name, 1);
        this->backward();
        other.backward();
    };
    pool->tensor_add_inplace(other.name, name, output.name);
    return output;
}

template<typename T>
Tensor<T>& Tensor<T>::operator+(T other) {
    auto &output = pool->createTensor(this->shape, name + "_" + std::to_string(other) + "-elementwise_addition_output");
    auto &to_add = pool->createTensor(std::vector<uint32_t>(this->shape.size(), 1), "const_add_tensor_" + name + "_" + std::to_string(other));

    output.back = [this, &output, &to_add](){
        this->pool->tensor_add_inplace(to_add.name, this->name, output.name, 1);
    };
    pool->tensor_add_inplace(to_add.name, name, output.name);
    return output;
}

template<typename T>
Tensor<T>& operator+(T lhs, Tensor<T>& rhs){
    // create output with same shape as rhs
    auto &output = rhs.pool->createTensor(rhs.shape, std::to_string(lhs) + "_" + rhs.name + "-elementwise_addition_output");
    // create scalar tensor to hold lhs
    auto &to_add = rhs.pool->createTensor(std::vector<uint32_t>(rhs.shape.size(), 1), "const_add_tensor_" + std::to_string(lhs) + "_" + rhs.name);
    to_add.dataBuffer->set(0, lhs);

    output.back = [&rhs, &output, &to_add](){
        rhs.pool->tensor_add_inplace(to_add.name, rhs.name, output.name, 1);
        rhs.backward();
    };

    rhs.pool->tensor_add_inplace(to_add.name, rhs.name, output.name);
    return output;
}

template<typename T>
Tensor<T>& Tensor<T>::exp(){
    
    auto &output = pool->tensor_exp(name);

    output.back = [&output, this](){
        this->pool->tensor_exp(this->name, output.name, 1);
        this->backward();
    };

    return output;
}

template<typename T>
Tensor<T>& Tensor<T>::matmul(Tensor<T>& other){
    auto &shape_a = this->shape;
    auto &shape_b = other.shape;

    // Extract dimensions
    // shape_a: [..., M, K]
    // shape_b: [..., N, K] (transposed, so K is last)
    // output: [..., M, N]

    std::vector<uint32_t> output_shape;

    // Copy batch dimensions (all dimensions except last 2)
    // Use the larger of the two batch dimension sets
    size_t batch_dims_a = shape_a.size() - 2;
    size_t batch_dims_b = shape_b.size() - 2;
    size_t max_batch_dims = std::max(batch_dims_a, batch_dims_b);

    // Handle broadcasting for batch dimensions
    for (size_t i = 0; i < max_batch_dims; ++i) {
        size_t idx_a = i + batch_dims_a - max_batch_dims;
        size_t idx_b = i + batch_dims_b - max_batch_dims;
        
        uint32_t dim_a = (i < max_batch_dims - batch_dims_a) ? 1 : shape_a[idx_a];
        uint32_t dim_b = (i < max_batch_dims - batch_dims_b) ? 1 : shape_b[idx_b];
        
        output_shape.push_back(std::max(dim_a, dim_b));
    }

    // Add matrix dimensions: M from A, N from B
    uint32_t M = shape_a[shape_a.size() - 2];
    uint32_t N = shape_b[shape_b.size() - 2];  // N is at position -2 since B is transposed

    output_shape.push_back(M);
    output_shape.push_back(N);

    auto &output = pool->createTensor(output_shape, name + other.name + "-matmul_output");
    pool->tensor_linear(output.name, name, other.name);
    output.back = [this, &other, &output](){
        pool->tensor_linear(output.name, this->name, other.name, "", 1);
        this->backward();
        other.backward();
    };
    
    return output;
}

template<typename T>
void Tensor<T>::elementwise_add(Tensor<T>& other) {
    pool->tensor_add_inplace(other.name, name);
}

template<typename T>
void Tensor<T>::elementwise_multiply(Tensor<T>& other) {
    pool->tensor_multiply_elementwise(other.name, name);
}