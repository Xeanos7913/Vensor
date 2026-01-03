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
    uint32_t kernel_type;   // 0 = dInput and dBias, 1 = dWeight
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

struct tanh_context {
    tensor_impl input_tensor;   
    tensor_impl output_tensor;
    uint batch_size;
    uint m, n;
    uint accumulate_grad; // 0: overwrite, 1: += for grads
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
    tensor_impl target_tensor;   // [B, C, H, W] shape
    tensor_impl predicted_tensor;// [B, C, H, W] shape
    tensor_impl loss_tensor;     // [B] shape
    uint batch_size;            // B
    uint channels;              // C
    uint height;                // H
    uint width;                 // W
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
    tensor_impl input_tensor;   // [B, C_in, H_in, W_in]
    tensor_impl weight_tensor;  // [C_out, C_in, KH, KW]
    tensor_impl bias_tensor;    // [C_out]
    tensor_impl out_tensor;     // [B, C_out, H_out, W_out]
    
    uint batch_size;
    uint in_channels;
    uint out_channels;
    uint in_height;
    uint in_width;
    uint out_height;
    uint out_width;
    uint kernel_h;
    uint kernel_w;
    uint stride_h;
    uint stride_w;
    uint padding_h;
    uint padding_w;
    uint dilation_h;
    uint dilation_w;
    
    uint use_bias;
    uint accumulate_grad;
    uint kernel_type;
    
    // GEMM dimensions
    // M = out_height * out_width
    // N = out_channels
    // K = in_channels * kernel_h * kernel_w

    uint m, n, k;
};

struct tensor_transposed_conv2d_context {
    tensor_impl input_tensor;   // [B, C_in, H_in, W_in]
    tensor_impl weight_tensor;  // [C_in, C_out, KH, KW]
    tensor_impl bias_tensor;    // [C_out]
    tensor_impl out_tensor;     // [B, C_out, H_out, W_out]
    
    uint batch_size;
    uint in_channels;
    uint out_channels;
    uint in_height;
    uint in_width;
    uint out_height;
    uint out_width;
    uint kernel_h;
    uint kernel_w;
    uint stride_h;
    uint stride_w;
    uint padding_h;
    uint padding_w;
    uint dilation_h;
    uint dilation_w;
    uint output_padding_h;
    uint output_padding_w;
    
    uint use_bias;
    uint accumulate_grad;
    uint kernel_type;
    
    // GEMM dimensions for TransposedConv2d
    // M = in_height * in_width (input spatial)
    // N = in_channels
    // K = out_channels * kernel_h * kernel_w
    uint m, n, k;
};

struct tensor_flash_attention_fwd_ctx{
    tensor_impl Q;
    tensor_impl K;
    tensor_impl V;
    VkDeviceAddress L;   // Softmax demonimators
    VkDeviceAddress M;  // Softmax max values
    tensor_impl Out;
    uint N_CTX;
    uint Z;  // batch size
    uint H;  // number of heads
    float sm_scale;  // softmax scale (typically 1/sqrt(d_k))
    int stride_qz, stride_qh, stride_qm, stride_qk;
    int stride_kz, stride_kh, stride_kn, stride_kk;
    int stride_vz, stride_vh, stride_vk, stride_vn;
    int stride_oz, stride_oh, stride_om, stride_on;
};

struct tensor_flash_attention_bwd_preprocess_ctx{
    tensor_impl Out;      // Output from forward pass (Out.data.data)
    VkDeviceAddress L;    // Softmax denominator from forward pass
    VkDeviceAddress Delta;// Delta values (sum of o * do)
    uint N_CTX;
    uint Z;               // batch size
    uint H;               // number of heads
    int stride_oz, stride_oh, stride_om, stride_on;
};

struct tensor_flash_attention_bwd_ctx {
    tensor_impl Q;        // Q.data.data for forward values, Q.grad.grad for gradients (output)
    tensor_impl K;        // K.data.data for forward values, K.grad.grad for gradients (output)
    tensor_impl V;        // V.data.data for forward values, V.grad.grad for gradients (output)
    VkDeviceAddress M;    // Softmax max values
    tensor_impl Out;      // Normalized gradient from preprocess lives in the out tensor
    VkDeviceAddress Delta;// Delta values from preprocess
    uint N_CTX;
    uint Z;              // batch size
    uint H;              // number of heads
    float sm_scale;      // softmax scale
    uint num_block;      // number of blocks
    int stride_qz, stride_qh, stride_qm, stride_qk;
    int stride_kz, stride_kh, stride_kn, stride_kk;
    int stride_vz, stride_vh, stride_vk, stride_vn;
};

struct tensor_upsample_context {
    tensor_impl input_tensor;   // [B, C, H_in, W_in] shape
    tensor_impl output_tensor;  // [B, C, H_out, W_out] shape
    uint batch_size;           // B
    uint channels;             // C
    uint height_in;            // H_in
    uint width_in;             // W_in
    uint height_out;           // H_out
    uint width_out;            // W_out
};

struct tensor_sample_context {
    tensor_impl input_tensor;  // going to be a vector tensor shaped (B, 1, N) [the prob distribution]
    tensor_impl output_tensor; // output tensor (B)
    uint32_t M, N;
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
    uint init_type;     // 0=Uniform, 1=Normal, 2=Xavier/Glorot, 3=He/Kaiming, 4=Constant
    uint fan_in;        // Number of input connections (for Xavier/He)
    uint fan_out;       // Number of output connections (for Xavier)
    float param1;       // mean for Normal, constant value, or min for Uniform
    float param2;       // stddev for Normal, unused, or max for Uniform
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

   std::vector<std::function<void()>> back;

    bool claimable = false;
    bool isClaimed = false;

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

    void backward() {
        for (size_t i = back.size(); i-- > 0; ) {
            DEBUG_PRINT("Calling backward " << i << " for tensor: " << name);
            back[i]();
        }
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

    void view(const std::vector<uint32_t>& dims) {
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
	   std::cout << name << " Tensor shape: " << shapeToString(shape) << "\n";
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
            flashAttention(
                readShaderBytecode("compiled_shaders/FlashAttention.comp.spv"), alloc, nullptr),
            flashAttentionBackwardPreprocess(
                readShaderBytecode("compiled_shaders/FlashAttentionBackwardPreprocess.comp.spv"), alloc, nullptr),
            flashAttentionBackward(
                readShaderBytecode("compiled_shaders/FlashAttentionBackwardPass.comp.spv"), alloc, nullptr),
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
            tanhShader(
                readShaderBytecode("compiled_shaders/TanH.comp.spv"), alloc, nullptr),
            tanhShaderBackward(
                readShaderBytecode("compiled_shaders/TanH_backward.comp.spv"), alloc, nullptr),
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
            sampleShader(
                readShaderBytecode("compiled_shaders/sample_from_dist.comp.spv"), alloc, nullptr),
            softmaxShader(
                readShaderBytecode("compiled_shaders/Softmax.comp.spv"), alloc, nullptr),
            embedLookupShader(
                readShaderBytecode("compiled_shaders/Embedding_table.comp.spv"), alloc, nullptr),
            embedLookupShaderBackward(
                readShaderBytecode("compiled_shaders/Embedding_table_backward.comp.spv"), alloc, nullptr),
            upsampleShader(
                readShaderBytecode("compiled_shaders/Upsample.comp.spv"), alloc, nullptr),
            upsampleShaderBackward(
                readShaderBytecode("compiled_shaders/Upsample_backward.comp.spv"), alloc, nullptr),
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

    Tensor<T>& createTensor(const std::vector<uint32_t>& dims, const std::string& name, bool claimable = false) {
        // if tensor already exists, return it (user-owned tensors)
        auto it = tensors.find(name);
        if (it != tensors.end() && !claimable) {
            DEBUG_PRINT("tensor with name: " << name << " already exists. Returning that tensor.");
            return *(it->second);
        }

        // look for a reusable intermediate tensor
        for (auto it = tensors.begin(); it != tensors.end(); ++it) {
            auto& tensor = it->second;
            if (tensor->shape == dims && tensor->claimable && !tensor->isClaimed) {
                tensor->isClaimed = true;

                // save old pointer
                auto ptr = std::move(it->second);

                // erase old key
                tensors.erase(it);

                // set new name and insert under new key
                ptr->name = name;
                tensors[name] = std::move(ptr);

                return *tensors[name];
            }
        }

        // create new tensor
        auto& new_tensor = *(tensors[name] = std::make_unique<Tensor<T>>(dims, allocator));
        new_tensor.pool = this;
        new_tensor.name = name;
        new_tensor.isClaimed = true;
        new_tensor.claimable = claimable;
        return new_tensor;
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
            tensor->back.clear();
            if(tensor->claimable){
                tensor->isClaimed = false;
                //tensor->dataBuffer->clearBuffer();
            }
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

    void tensor_tanh(const std::string& input_tensor, const std::string& output_tensor, uint32_t mode = 0){
        tanh_context uniform;
        uniform.batch_size = tensors[input_tensor]->shape[0];
        uniform.input_tensor = tensors[input_tensor]->getTensorImpl();
        uniform.output_tensor = tensors[output_tensor]->getTensorImpl();

        uint32_t numel = uniform.input_tensor.num_elements;

        auto cielDiv = [](uint32_t a, uint32_t b) { return (a + b - 1) / b; };

        tensor_push_const push;
        uint32_t grpx = cielDiv(numel, 256u);
        uint32_t wrkgrp[3] = {grpx, 1, 1};
        
        if(mode == 0){
            push.uniformAddress = tanhShader.uniformBuffer->getBufferAddress();
            push.mode = mode;
            tanhShader.loadUniform(uniform, push);
            tanhShader.execute(wrkgrp);
        }else{
            push.uniformAddress = tanhShaderBackward.uniformBuffer->getBufferAddress();
            push.mode = mode;
            tanhShaderBackward.loadUniform(uniform, push);
            tanhShaderBackward.execute(wrkgrp);
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

    // general 2d convolution
    void tensor_conv2d(const std::string& output_tensor,
                   const std::string& input_tensor,
                   const std::string& weight_tensor,
                   const std::string& bias_tensor,
                   uint32_t kernel_h,
                   uint32_t kernel_w,
                   uint32_t mode = 0,
                   uint32_t stride_w = 1,
                   uint32_t stride_h = 1,
                   uint32_t pad_h = 0,
                   uint32_t pad_w = 0,
                   uint32_t dilation_h = 1,
                   uint32_t dilation_w = 1) {
    
        tensor_conv2d_context uniform{};

        constexpr uint32_t BM = 128;
        constexpr uint32_t BN = 128;
        constexpr uint32_t BK = 8;
        constexpr uint32_t NUM_THREADS = 256;

        auto ceilDiv = [](uint32_t a, uint32_t b) { return (a + b - 1) / b; };

        // ---------------------------------------------------------------------
        // Common setup
        // ---------------------------------------------------------------------
        const auto& input_shape = tensors[input_tensor]->shape;
        const auto& weight_shape = tensors[weight_tensor]->shape;
        const auto& output_shape = tensors[output_tensor]->shape;

        // Validate shapes
        if (input_shape.size() != 4)
            throw std::invalid_argument("Input must be 4D [B, C_in, H_in, W_in]");
        if (weight_shape.size() != 4)
            throw std::invalid_argument("Weight must be 4D [C_out, C_in, KH, KW]");
        if (output_shape.size() != 4)
            throw std::invalid_argument("Output must be 4D [B, C_out, H_out, W_out]");

        uint32_t batch_size = input_shape[0];
        uint32_t in_channels = input_shape[1];
        uint32_t in_height = input_shape[2];
        uint32_t in_width = input_shape[3];

        uint32_t out_channels = weight_shape[0];
        uint32_t weight_in_channels = weight_shape[1];
        uint32_t weight_kh = weight_shape[2];
        uint32_t weight_kw = weight_shape[3];

        uint32_t out_batch = output_shape[0];
        uint32_t out_c = output_shape[1];
        uint32_t out_height = output_shape[2];
        uint32_t out_width = output_shape[3];

        // Validate dimensions
        if (weight_in_channels != in_channels)
            throw std::invalid_argument("Weight input channels must match input channels");
        if (weight_kh != kernel_h || weight_kw != kernel_w)
            throw std::invalid_argument("Weight kernel size must match provided kernel size");
        if (out_channels != out_c)
            throw std::invalid_argument("Output channels must match weight output channels");
        if (batch_size != out_batch)
            throw std::invalid_argument("Output batch size must match input batch size");

        // Calculate expected output dimensions
        uint32_t expected_out_h = (in_height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
        uint32_t expected_out_w = (in_width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

        if (expected_out_h != out_height || expected_out_w != out_width)
            throw std::invalid_argument("Output spatial dimensions don't match expected values");

        // Fill uniform context
        uniform.input_tensor = tensors[input_tensor]->getTensorImpl();
        uniform.weight_tensor = tensors[weight_tensor]->getTensorImpl();
        uniform.out_tensor = tensors[output_tensor]->getTensorImpl();

        if (!bias_tensor.empty()) {
            const auto& bias_shape = tensors[bias_tensor]->shape;
            if (bias_shape.size() != 1 || bias_shape[0] != out_channels)
                throw std::invalid_argument("Bias must be 1D [C_out]");
            uniform.bias_tensor = tensors[bias_tensor]->getTensorImpl();
            uniform.use_bias = 1;
        } else {
            uniform.use_bias = 0;
        }

        uniform.batch_size = batch_size;
        uniform.in_channels = in_channels;
        uniform.out_channels = out_channels;
        uniform.in_height = in_height;
        uniform.in_width = in_width;
        uniform.out_height = out_height;
        uniform.out_width = out_width;
        uniform.kernel_h = kernel_h;
        uniform.kernel_w = kernel_w;
        uniform.stride_h = stride_h;
        uniform.stride_w = stride_w;
        uniform.padding_h = pad_h;
        uniform.padding_w = pad_w;
        uniform.dilation_h = dilation_h;
        uniform.dilation_w = dilation_w;
        uniform.accumulate_grad = 1;

        // GEMM dimensions
        // M = out_height * out_width (output spatial dimensions flattened)
        // N = out_channels
        // K = in_channels * kernel_h * kernel_w
        uint32_t M = out_height * out_width;
        uint32_t N = out_channels;
        uint32_t K = in_channels * kernel_h * kernel_w;

        uniform.m = M;
        uniform.n = N;
        uniform.k = K;

        DEBUG_PRINT("Conv2d tensor " << input_tensor << " of shape "
            << shapeToString(input_shape) << " with transposed weights " << weight_tensor
            << " of shape " << shapeToString(weight_shape));
        
        // ---------------------------------------------------------------------
        // FORWARD
        // ---------------------------------------------------------------------
        if (mode == 0) {
            // Grid dimensions for forward pass
            // GEMM: Output[B, M, N] = Im2Col(Input)[B, M, K] @ Weight[N, K]^T
            uint32_t groupX = ceilDiv(N, BN);  // Output channels dimension
            uint32_t groupY = ceilDiv(M, BM);  // Spatial dimension (H_out * W_out)
            uint32_t groupZ = batch_size;      // Batch dimension

            uint32_t workgroup[3] = { groupX, groupY, groupZ };

            tensor_push_const pc{};
            pc.grid_size = { groupX, groupY, groupZ };
            pc.mode = 0;
            pc.uniformAddress = conv2dShader.uniformBuffer->getBufferAddress();

            conv2dShader.loadUniform(uniform, pc);
            conv2dShader.execute(workgroup);

            return;
        }

        // ---------------------------------------------------------------------
        // BACKWARD
        // ---------------------------------------------------------------------
        
        // ---- Kernel 1: dInput + dBias ----
        // GEMM for dInput: dInput_col[B, M, K] = dOutput[B, M, N] @ Weight[N, K]
        // Then scatter dInput_col back to dInput spatial positions
        {
            uint32_t groupX = ceilDiv(K, BN);  // Filter dimension (C_in * KH * KW)
            uint32_t groupY = ceilDiv(M, BM);  // Spatial dimension (H_out * W_out)
            uint32_t groupZ = batch_size;      // Batch dimension

            uint32_t workgroup[3] = { groupX, groupY, groupZ };

            tensor_push_const pc{};
            pc.grid_size = { groupX, groupY, groupZ };
            uniform.kernel_type = 0;  // KERNEL_INPUT_GRAD
            pc.mode = 1;
            pc.uniformAddress = conv2dShaderBackward.uniformBuffer->getBufferAddress();

            conv2dShaderBackward.loadUniform(uniform, pc);
            conv2dShaderBackward.execute(workgroup);
        }

        // ---- Kernel 2: dWeight ----
        // GEMM: dWeight[N, K] = dOutput^T[N, M] @ Im2Col(Input)[M, K]
        // Accumulated across batches
        {
            uint32_t groupX = ceilDiv(K, BN);  // Filter dimension (C_in * KH * KW)
            uint32_t groupY = ceilDiv(N, BM);  // Output channels dimension
            uint32_t groupZ = batch_size;      // Batch dimension (accumulate across)

            uint32_t workgroup[3] = { groupX, groupY, groupZ };

            tensor_push_const pc{};
            pc.grid_size = { groupX, groupY, groupZ };
            uniform.kernel_type = 1;  // KERNEL_WEIGHT_GRAD
            pc.mode = 1;
            pc.uniformAddress = conv2dShaderBackward.uniformBuffer->getBufferAddress();

            conv2dShaderBackward.loadUniform(uniform, pc);
            conv2dShaderBackward.execute(workgroup);
        }
    };

    // general transposed (de)conv2d supporting dynamic kernel sizes
    void tensor_transposed_conv2d(const std::string& output_tensor,
                              const std::string& input_tensor,
                              const std::string& weight_tensor,
                              const std::string& bias_tensor,
                              uint32_t kernel_h,
                              uint32_t kernel_w,
                              uint32_t mode = 0,
                              uint32_t stride_w = 1,
                              uint32_t stride_h = 1,
                              uint32_t pad_h = 0,
                              uint32_t pad_w = 0,
                              uint32_t dilation_h = 1,
                              uint32_t dilation_w = 1,
                              uint32_t output_pad_h = 0,
                              uint32_t output_pad_w = 0) {
    
        tensor_transposed_conv2d_context uniform{};

        constexpr uint32_t BM = 128;
        constexpr uint32_t BN = 128;
        constexpr uint32_t BK = 8;
        constexpr uint32_t NUM_THREADS = 256;

        auto ceilDiv = [](uint32_t a, uint32_t b) { return (a + b - 1) / b; };

        // ---------------------------------------------------------------------
        // Common setup
        // ---------------------------------------------------------------------
        const auto& input_shape = tensors[input_tensor]->shape;
        const auto& weight_shape = tensors[weight_tensor]->shape;
        const auto& output_shape = tensors[output_tensor]->shape;

        // Validate shapes
        if (input_shape.size() != 4)
            throw std::invalid_argument("Input must be 4D [B, C_in, H_in, W_in]");
        if (weight_shape.size() != 4)
            throw std::invalid_argument("Weight must be 4D [C_in, C_out, KH, KW]");  // Note: C_in first!
        if (output_shape.size() != 4)
            throw std::invalid_argument("Output must be 4D [B, C_out, H_out, W_out]");

        uint32_t batch_size = input_shape[0];
        uint32_t in_channels = input_shape[1];
        uint32_t in_height = input_shape[2];
        uint32_t in_width = input_shape[3];

        uint32_t weight_in_channels = weight_shape[0];  // First dimension is C_in
        uint32_t out_channels = weight_shape[1];        // Second dimension is C_out
        uint32_t weight_kh = weight_shape[2];
        uint32_t weight_kw = weight_shape[3];

        uint32_t out_batch = output_shape[0];
        uint32_t out_c = output_shape[1];
        uint32_t out_height = output_shape[2];
        uint32_t out_width = output_shape[3];

        // Validalte dimensions
        if (weight_in_channels != in_channels)
            throw std::invalid_argument("Weight input channels must match input channels");
        if (weight_kh != kernel_h || weight_kw != kernel_w)
            throw std::invalid_argument("Weight kernel size must match provided kernel size");
        if (out_channels != out_c)
            throw std::invalid_argument("Output channels must match weight output channels");
        if (batch_size != out_batch)
            throw std::invalid_argument("Output batch size must match input batch size");

        // Calculate expected output dimensions for transposed convolution
        // Formula: out = (in - 1) * stride - 2 * padding + dilation * (kernel - 1) + output_padding + 1
        uint32_t expected_out_h = (in_height - 1) * stride_h - 2 * pad_h + dilation_h * (kernel_h - 1) + output_pad_h + 1;
        uint32_t expected_out_w = (in_width - 1) * stride_w - 2 * pad_w + dilation_w * (kernel_w - 1) + output_pad_w + 1;

        if (expected_out_h != out_height || expected_out_w != out_width)
            throw std::invalid_argument("Output spatial dimensions don't match expected values. Expected: " +
                                        std::to_string(expected_out_h) + "x" + std::to_string(expected_out_w) +
                                        ", Got: " + std::to_string(out_height) + "x" + std::to_string(out_width));

        // Fill uniform context
        uniform.input_tensor = tensors[input_tensor]->getTensorImpl();
        uniform.weight_tensor = tensors[weight_tensor]->getTensorImpl();
        uniform.out_tensor = tensors[output_tensor]->getTensorImpl();

        if (!bias_tensor.empty()) {
            const auto& bias_shape = tensors[bias_tensor]->shape;
            if (bias_shape.size() != 1 || bias_shape[0] != out_channels)
                throw std::invalid_argument("Bias must be 1D [C_out]");
            uniform.bias_tensor = tensors[bias_tensor]->getTensorImpl();
            uniform.use_bias = 1;
        } else {
            uniform.use_bias = 0;
        }

        uniform.batch_size = batch_size;
        uniform.in_channels = in_channels;
        uniform.out_channels = out_channels;
        uniform.in_height = in_height;
        uniform.in_width = in_width;
        uniform.out_height = out_height;
        uniform.out_width = out_width;
        uniform.kernel_h = kernel_h;
        uniform.kernel_w = kernel_w;
        uniform.stride_h = stride_h;
        uniform.stride_w = stride_w;
        uniform.padding_h = pad_h;
        uniform.padding_w = pad_w;
        uniform.dilation_h = dilation_h;
        uniform.dilation_w = dilation_w;
        uniform.output_padding_h = output_pad_h;
        uniform.output_padding_w = output_pad_w;
        uniform.accumulate_grad = 1;

        // GEMM dimensions for TransposedConv2d
        // M = out_height * out_width (output spatial dimensions flattened)
        // N = out_channels
        // K = in_channels * in_height * in_width (flattened input)
        uint32_t M = out_height * out_width;
        uint32_t N = out_channels;
        uint32_t K = in_channels * in_height * in_width;

        uniform.m = M;
        uniform.n = N;
        uniform.k = K;

        DEBUG_PRINT("TransposedConv2d tensor " << input_tensor << " of shape "
            << shapeToString(input_shape) << " with transposed weights " << weight_tensor
            << " of shape " << shapeToString(weight_shape));

        // ---------------------------------------------------------------------
        // FORWARD
        // ---------------------------------------------------------------------
        if (mode == 0) {
            // Grid dimensions for forward pass
            // GEMM: Output[B, M, N] = Input_reshaped[B, M, K] @ Weight_transformed[K, N]
            // where the weight transformation implicitly happens via the kernel summation
            uint32_t groupX = ceilDiv(N, BN);  // Output channels dimension
            uint32_t groupY = ceilDiv(M, BM);  // Output spatial dimension (H_out * W_out)
            uint32_t groupZ = batch_size;      // Batch dimension

            uint32_t workgroup[3] = { groupX, groupY, groupZ };

            tensor_push_const pc{};
            pc.grid_size = { groupX, groupY, groupZ };
            pc.mode = 0;
            pc.uniformAddress = transposedConv2dShader.uniformBuffer->getBufferAddress();

            transposedConv2dShader.loadUniform(uniform, pc);
            transposedConv2dShader.execute(workgroup);

            return;
        }

        // ---------------------------------------------------------------------
        // BACKWARD
        // ---------------------------------------------------------------------
        
        // For TransposedConv2d backward:
        // M = in_height * in_width (input spatial)
        // N = in_channels
        // K = out_channels * kernel_h * kernel_w
        uint32_t M_back = in_height * in_width;
        uint32_t N_back = in_channels;
        uint32_t K_back = out_channels * kernel_h * kernel_w;
        
        uniform.m = M_back;
        uniform.n = N_back;
        uniform.k = K_back;
        
        // ---- Kernel 1: dInput + dBias ----
        // For TransposedConv2d backward, computing dInput is like doing regular convolution
        // GEMM: dInput[B, M, N] = dOutput_col[B, M, K] @ Weight[K, N]
        // where M = in_h * in_w, N = in_channels, K = out_channels * kh * kw
        {
            uint32_t groupX = ceilDiv(N_back, BN);  // Input channels dimension
            uint32_t groupY = ceilDiv(M_back, BM);  // Input spatial dimension (H_in * W_in)
            uint32_t groupZ = batch_size;           // Batch dimension

            uint32_t workgroup[3] = { groupX, groupY, groupZ };

            tensor_push_const pc{};
            pc.grid_size = { groupX, groupY, groupZ };
            uniform.kernel_type = 0;  // KERNEL_INPUT_GRAD
            pc.mode = 1;
            pc.uniformAddress = transposedConv2dShaderBackward.uniformBuffer->getBufferAddress();

            transposedConv2dShaderBackward.loadUniform(uniform, pc);
            transposedConv2dShaderBackward.execute(workgroup);
        }

        // ---- Kernel 2: dWeight ----
        // GEMM: dWeight[N, K] = Input^T[N, M] @ dOutput_col[M, K]
        // where N = in_channels, M = in_h * in_w, K = out_channels * kh * kw
        // Result stored as [C_in, C_out, KH, KW]
        // Accumulated across batches
        {
            uint32_t groupX = ceilDiv(K_back, BN);  // Output filter dimension (C_out * KH * KW)
            uint32_t groupY = ceilDiv(N_back, BM);  // Input channels dimension
            uint32_t groupZ = batch_size;           // Batch dimension (accumulate across)

            uint32_t workgroup[3] = { groupX, groupY, groupZ };

            tensor_push_const pc{};
            pc.grid_size = { groupX, groupY, groupZ };
            uniform.kernel_type = 1;  // KERNEL_WEIGHT_GRAD
            pc.mode = 1;
            pc.uniformAddress = transposedConv2dShaderBackward.uniformBuffer->getBufferAddress();

            transposedConv2dShaderBackward.loadUniform(uniform, pc);
            transposedConv2dShaderBackward.execute(workgroup);
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
        uint32_t batch_size = 0; //prod(shapeB, 0, shapeB.size() - 2);
        
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
            output = &createTensor(tensors[logvar_tensor]->shape, logvar_tensor + "-std_dev_tensor", true);
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
    
    void tensor_upsample(const std::string& input_tensor, const std::string& output_tensor, 
                        uint32_t height_in, uint32_t width_in, 
                        uint32_t height_out, uint32_t width_out, 
                        uint32_t mode = 0){
        auto inp = tensors[input_tensor].get();
        auto ou = tensors[output_tensor].get();
        
        const auto original_shape = inp->shape;
        const uint32_t total_elements = inp->get_num_elements();
        
        if(original_shape.size() < 1){
            throw std::invalid_argument("Input tensor must have at least 1 dimension");
        }
        
        uint32_t batch_size = original_shape[0];
        
        // Calculate required channels based on input elements and dimensions
        uint32_t required_elements_per_batch = height_in * width_in;
        if(total_elements % batch_size != 0){
            throw std::invalid_argument("Total elements must be divisible by batch size");
        }
        
        uint32_t elements_per_batch = total_elements / batch_size;
        if(elements_per_batch % required_elements_per_batch != 0){
            throw std::invalid_argument("Elements per batch incompatible with height_in * width_in");
        }
        
        uint32_t channels = elements_per_batch / required_elements_per_batch;
        
        // Create the required 4D view [B, C, H, W]
        std::vector<uint32_t> temp_shape = {batch_size, channels, height_in, width_in};
        
        // Verify the view is valid
        uint64_t expected = 1;
        for (auto d : temp_shape) expected *= d;
        
        if (expected != total_elements){
            throw std::invalid_argument("Cannot create valid [B, C, H, W] view with given dimensions");
        }
        
        bool reshaped = (temp_shape != original_shape);
        
        if(reshaped){
            tensors[input_tensor]->view(temp_shape);
        }
        
        tensor_upsample_context uniform;
        uniform.input_tensor = inp->getTensorImpl();
        uniform.output_tensor = ou->getTensorImpl();
        uniform.batch_size = batch_size;
        uniform.channels = channels;
        uniform.height_in = height_in;
        uniform.width_in = width_in;
        uniform.height_out = height_out;
        uniform.width_out = width_out;
        
        uint32_t numElements = uniform.output_tensor.num_elements;
        auto ceilDiv = [](uint32_t a, uint32_t b) { return (a + b - 1) / b; };
        uint32_t grpx = ceilDiv(numElements, 256u);
        uint32_t wrkgrp[3] = {grpx, 1, 1};
        
        tensor_push_const push;
        
        try {
            if(mode == 0){
                push.uniformAddress = upsampleShader.uniformBuffer->getBufferAddress();
                upsampleShader.loadUniform(uniform, push);
                upsampleShader.execute(wrkgrp);
            }
            else{
                push.uniformAddress = upsampleShaderBackward.uniformBuffer->getBufferAddress();
                upsampleShaderBackward.loadUniform(uniform, push);
                upsampleShaderBackward.execute(wrkgrp);
            }
        } catch (...) {
            if (reshaped) tensors[input_tensor]->view(original_shape);
            throw;
        }
        
        if (reshaped)
            tensors[input_tensor]->view(original_shape);
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
            output = &createTensor(tensors[input_tensor]->shape, input_tensor + "-exponentiated_", true);
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

        DEBUG_PRINT("Dispatch: ");
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
        else {
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

        auto prod = [](const std::vector<uint32_t>& v, size_t l, size_t r)->uint32_t {
            uint64_t p = 1;
            for (size_t i = l; i < r; ++i) p *= v[i];
            if (p > UINT32_MAX) throw std::overflow_error("Batch too large");
            return static_cast<uint32_t>(p);
        };

        // ---------------------------------------------------------------------
        // Common setup
        // ---------------------------------------------------------------------
        uniform.weight_tensor = tensors[weight_tensor]->getTensorImpl();

        if (!bias_tensor.empty()) {
            uniform.bias_tensor = tensors[bias_tensor]->getTensorImpl();
            uniform.use_bias = 1;
        } else {
            uniform.use_bias = 0;
        }

        uniform.output_tensor = tensors[output_tensor]->getTensorImpl();
        uniform.mode = mode;
        uniform.accumulate_grad = 1;

        constexpr uint32_t BM = 128;
        constexpr uint32_t BN = 128;
        constexpr uint32_t BK = 8;
        constexpr uint32_t NUM_THREADS = 256;

        auto ceilDiv = [](uint32_t a, uint32_t b) { return (a + b - 1) / b; };

        // ---------------------------------------------------------------------
        // FORWARD
        // ---------------------------------------------------------------------
        if (mode == 0) {
            const auto original_shape = tensors[input_tensor]->shape;
            const uint32_t total_elements = tensors[input_tensor]->get_num_elements();

            if (original_shape.size() < 2)
                throw std::invalid_argument("Input rank must be >= 2");

            uint32_t orig_M = original_shape[original_shape.size() - 2];
            uint32_t orig_K = original_shape.back();

            const auto& weight_shape = tensors[weight_tensor]->shape;
            if (weight_shape.size() < 2)
                throw std::invalid_argument("Weight rank must be >= 2");

            uint32_t KB = weight_shape.back();
            uint32_t N  = weight_shape[weight_shape.size() - 2];

            bool reshaped = false;
            std::vector<uint32_t> temp_shape;

            uint32_t M = orig_M;
            uint32_t K = orig_K;
            uint32_t batch_size = prod(original_shape, 0, original_shape.size() - 2);

            if (KB != orig_K) {
                const auto& output_shape = tensors[output_tensor]->shape;
                if (output_shape.size() < 2)
                    throw std::invalid_argument("Output rank must be >= 2");

                temp_shape = output_shape;
                temp_shape[temp_shape.size() - 1] = KB;

                M = temp_shape[temp_shape.size() - 2];
                K = KB;

                uint64_t expected = 1;
                for (auto d : temp_shape) expected *= d;

                if (expected != total_elements)
                    throw std::invalid_argument("Inner-dim mismatch cannot be reconciled by view");

                tensors[input_tensor]->view(temp_shape);
                reshaped = true;
                batch_size = prod(temp_shape, 0, temp_shape.size() - 2);
            }

            uniform.input_tensor = tensors[input_tensor]->getTensorImpl();
            uniform.m = M;
            uniform.n = N;
            uniform.k = K;
            uniform.batch_size = batch_size;

            const auto& output_shape = tensors[output_tensor]->shape;
            uint32_t dispatchM = output_shape[output_shape.size() - 2];
            uint32_t dispatchN = output_shape.back();
            uint32_t dispatchB = prod(output_shape, 0, output_shape.size() - 2);

            uint32_t groupX = ceilDiv(dispatchN, BN);
            uint32_t groupY = ceilDiv(dispatchM, BM);
            uint32_t groupZ = dispatchB;

            uint32_t workgroup[3] = { groupX, groupY, groupZ };

            tensor_push_const pc{};
            pc.grid_size = { groupX, groupY, groupZ };
            pc.mode = 0;
            pc.uniformAddress = linearReLUShader.uniformBuffer->getBufferAddress();

            try {
                linearReLUShader.loadUniform(uniform, pc);
                linearReLUShader.execute(workgroup);
            } catch (...) {
                if (reshaped) tensors[input_tensor]->view(original_shape);
                throw;
            }

            if (reshaped)
                tensors[input_tensor]->view(original_shape);

            return;
        }

        // ---------------------------------------------------------------------
        // BACKWARD
        // ---------------------------------------------------------------------
        const auto original_shape = tensors[input_tensor]->shape;
        const uint32_t total_elements = tensors[input_tensor]->get_num_elements();

        if (original_shape.size() < 2)
            throw std::invalid_argument("Backward: Input rank must be >= 2");

        uint32_t orig_M = original_shape[original_shape.size() - 2];
        uint32_t orig_K = original_shape.back();

        const auto& weight_shape = tensors[weight_tensor]->shape;
        if (weight_shape.size() < 2)
            throw std::invalid_argument("Backward: Weight rank must be >= 2");

        uint32_t KB = weight_shape.back();
        uint32_t N  = weight_shape[weight_shape.size() - 2];

        bool reshaped = false;
        std::vector<uint32_t> temp_shape;

        uint32_t M = orig_M;
        uint32_t K = orig_K;
        uint32_t batch_size = prod(original_shape, 0, original_shape.size() - 2);

        if (KB != orig_K) {
            const auto& output_shape = tensors[output_tensor]->shape;
            if (output_shape.size() < 2)
                throw std::invalid_argument("Backward: Output rank must be >= 2");

            temp_shape = output_shape;
            temp_shape[temp_shape.size() - 1] = KB;

            M = temp_shape[temp_shape.size() - 2];
            K = KB;

            uint64_t expected = 1;
            for (auto d : temp_shape) expected *= d;

            if (expected != total_elements)
                throw std::invalid_argument("Backward: Inner-dim mismatch cannot be reconciled by view");

            tensors[input_tensor]->view(temp_shape);
            reshaped = true;
            batch_size = prod(temp_shape, 0, temp_shape.size() - 2);
        }

        uniform.input_tensor = tensors[input_tensor]->getTensorImpl();
        uniform.m = M;
        uniform.n = N;
        uniform.k = K;
        uniform.batch_size = batch_size;

        // ---- Kernel 1: dInput + dBias ----
        try {
            {
                uint32_t groupX = ceilDiv(K, BN);
                uint32_t groupY = ceilDiv(M, BM);
                uint32_t groupZ = batch_size;

                uint32_t workgroup[3] = { groupX, groupY, groupZ };

                tensor_push_const pc{};
                pc.grid_size = { groupX, groupY, groupZ };
                uniform.kernel_type = 0;
                pc.mode = 1;
                pc.uniformAddress = linearReLUShaderBackward.uniformBuffer->getBufferAddress();

                linearReLUShaderBackward.loadUniform(uniform, pc);
                linearReLUShaderBackward.execute(workgroup);
            }

            // ---- Kernel 2: dWeight ----
            {
                uint32_t groupX = ceilDiv(K, BN);
                uint32_t groupY = ceilDiv(N, BM);
                uint32_t groupZ = batch_size;

                uint32_t workgroup[3] = { groupX, groupY, groupZ };

                tensor_push_const pc{};
                pc.grid_size = { groupX, groupY, groupZ };
                uniform.kernel_type = 1;
                pc.mode = 1;
                pc.uniformAddress = linearReLUShaderBackward.uniformBuffer->getBufferAddress();

                linearReLUShaderBackward.loadUniform(uniform, pc);
                linearReLUShaderBackward.execute(workgroup);
            }
        } catch (...) {
            if (reshaped) tensors[input_tensor]->view(original_shape);
            throw;
        }

        if (reshaped)
            tensors[input_tensor]->view(original_shape);
    }

    void tensor_linear(const std::string& output_tensor,
                   const std::string& input_tensor,
                   const std::string& weight_tensor,
                   const std::string& bias_tensor = "",
                   uint32_t mode = 0) {
        linear_context uniform{};

        auto prod = [](const std::vector<uint32_t>& v, size_t l, size_t r)->uint32_t {
            uint64_t p = 1;
            for (size_t i = l; i < r; ++i) p *= v[i];
            if (p > UINT32_MAX) throw std::overflow_error("Batch too large");
            return static_cast<uint32_t>(p);
        };

        // ---------------------------------------------------------------------
        // Common setup
        // ---------------------------------------------------------------------
        uniform.weight_tensor = tensors[weight_tensor]->getTensorImpl();

        if (!bias_tensor.empty()) {
            uniform.bias_tensor = tensors[bias_tensor]->getTensorImpl();
            uniform.use_bias = 1;
        } else {
            uniform.use_bias = 0;
        }

        uniform.output_tensor = tensors[output_tensor]->getTensorImpl();
        uniform.mode = mode;
        uniform.accumulate_grad = 1;

        constexpr uint32_t BM = 128;
        constexpr uint32_t BN = 128;
        constexpr uint32_t BK = 8;
        constexpr uint32_t NUM_THREADS = 256;

        auto ceilDiv = [](uint32_t a, uint32_t b) { return (a + b - 1) / b; };

        // ---------------------------------------------------------------------
        // FORWARD
        // ---------------------------------------------------------------------
        if (mode == 0) {
            const auto original_shape = tensors[input_tensor]->shape;
            const uint32_t total_elements = tensors[input_tensor]->get_num_elements();

            if (original_shape.size() < 2)
                throw std::invalid_argument("Input rank must be >= 2");

            uint32_t orig_M = original_shape[original_shape.size() - 2];
            uint32_t orig_K = original_shape.back();

            const auto& weight_shape = tensors[weight_tensor]->shape;
            if (weight_shape.size() < 2)
                throw std::invalid_argument("Weight rank must be >= 2");

            uint32_t KB = weight_shape.back();
            uint32_t N  = weight_shape[weight_shape.size() - 2];

            bool reshaped = false;
            std::vector<uint32_t> temp_shape;

            uint32_t M = orig_M;
            uint32_t K = orig_K;
            uint32_t batch_size = prod(original_shape, 0, original_shape.size() - 2);

            if (KB != orig_K) {
                const auto& output_shape = tensors[output_tensor]->shape;
                if (output_shape.size() < 2)
                    throw std::invalid_argument("Output rank must be >= 2");

                temp_shape = output_shape;
                temp_shape[temp_shape.size() - 1] = KB;

                M = temp_shape[temp_shape.size() - 2];
                K = KB;

                uint64_t expected = 1;
                for (auto d : temp_shape) expected *= d;

                if (expected != total_elements)
                    throw std::invalid_argument("Inner-dim mismatch cannot be reconciled by view");

                tensors[input_tensor]->view(temp_shape);
                reshaped = true;
                batch_size = prod(temp_shape, 0, temp_shape.size() - 2);
            }

            uniform.input_tensor = tensors[input_tensor]->getTensorImpl();
            uniform.m = M;
            uniform.n = N;
            uniform.k = K;
            uniform.batch_size = batch_size;

            const auto& output_shape = tensors[output_tensor]->shape;
            uint32_t dispatchM = output_shape[output_shape.size() - 2];
            uint32_t dispatchN = output_shape.back();
            uint32_t dispatchB = prod(output_shape, 0, output_shape.size() - 2);

            uint32_t groupX = ceilDiv(dispatchN, BN);
            uint32_t groupY = ceilDiv(dispatchM, BM);
            uint32_t groupZ = dispatchB;

            uint32_t workgroup[3] = { groupX, groupY, groupZ };

            tensor_push_const pc{};
            pc.grid_size = { groupX, groupY, groupZ };
            pc.mode = 0;
            pc.uniformAddress = linearShader.uniformBuffer->getBufferAddress();

            try {
                linearShader.loadUniform(uniform, pc);
                linearShader.execute(workgroup);
            } catch (...) {
                if (reshaped) tensors[input_tensor]->view(original_shape);
                throw;
            }

            if (reshaped)
                tensors[input_tensor]->view(original_shape);

            return;
        }

        // ---------------------------------------------------------------------
        // BACKWARD
        // ---------------------------------------------------------------------
        const auto original_shape = tensors[input_tensor]->shape;
        const uint32_t total_elements = tensors[input_tensor]->get_num_elements();

        if (original_shape.size() < 2)
            throw std::invalid_argument("Backward: Input rank must be >= 2");

        uint32_t orig_M = original_shape[original_shape.size() - 2];
        uint32_t orig_K = original_shape.back();

        const auto& weight_shape = tensors[weight_tensor]->shape;
        if (weight_shape.size() < 2)
            throw std::invalid_argument("Backward: Weight rank must be >= 2");

        uint32_t KB = weight_shape.back();
        uint32_t N  = weight_shape[weight_shape.size() - 2];

        bool reshaped = false;
        std::vector<uint32_t> temp_shape;

        uint32_t M = orig_M;
        uint32_t K = orig_K;
        uint32_t batch_size = prod(original_shape, 0, original_shape.size() - 2);

        if (KB != orig_K) {
            const auto& output_shape = tensors[output_tensor]->shape;
            if (output_shape.size() < 2)
                throw std::invalid_argument("Backward: Output rank must be >= 2");

            temp_shape = output_shape;
            temp_shape[temp_shape.size() - 1] = KB;

            M = temp_shape[temp_shape.size() - 2];
            K = KB;

            uint64_t expected = 1;
            for (auto d : temp_shape) expected *= d;

            if (expected != total_elements)
                throw std::invalid_argument("Backward: Inner-dim mismatch cannot be reconciled by view");

            tensors[input_tensor]->view(temp_shape);
            reshaped = true;
            batch_size = prod(temp_shape, 0, temp_shape.size() - 2);
        }

        uniform.input_tensor = tensors[input_tensor]->getTensorImpl();
        uniform.m = M;
        uniform.n = N;
        uniform.k = K;
        uniform.batch_size = batch_size;

        // ---- Kernel 1: dInput + dBias ----
        try {
            {
                uint32_t groupX = ceilDiv(K, BN);
                uint32_t groupY = ceilDiv(M, BM);
                uint32_t groupZ = batch_size;

                uint32_t workgroup[3] = { groupX, groupY, groupZ };

                tensor_push_const pc{};
                pc.grid_size = { groupX, groupY, groupZ };
                uniform.kernel_type = 0;
                pc.mode = 1;
                pc.uniformAddress = linearShaderBackward.uniformBuffer->getBufferAddress();

                linearShaderBackward.loadUniform(uniform, pc);
                linearShaderBackward.execute(workgroup);
            }

            // ---- Kernel 2: dWeight ----
            {
                uint32_t groupX = ceilDiv(K, BN);
                uint32_t groupY = ceilDiv(N, BM);
                uint32_t groupZ = batch_size;

                uint32_t workgroup[3] = { groupX, groupY, groupZ };

                tensor_push_const pc{};
                pc.grid_size = { groupX, groupY, groupZ };
                uniform.kernel_type = 1;
                pc.mode = 1;
                pc.uniformAddress = linearShaderBackward.uniformBuffer->getBufferAddress();

                linearShaderBackward.loadUniform(uniform, pc);
                linearShaderBackward.execute(workgroup);
            }
        } catch (...) {
            if (reshaped) tensors[input_tensor]->view(original_shape);
            throw;
        }

        if (reshaped)
            tensors[input_tensor]->view(original_shape);
    }

    // input is [Batch, sequence, embed_dim]
    // So, weights must be [embed_dim, 3 * n_heads * d_model] to generate all the qkv tensors for all the sequence vectors for all batches.
    void tensor_flash_attention(const std::string& input_seq, const std::string& W_qkv, const std::string& tmp_qkv, const std::string& output, VkDeviceAddress L_buffer, VkDeviceAddress M_buffer, uint32_t d_model, uint32_t max_seq_len, uint32_t n_heads){
        
        tensor_flash_attention_fwd_ctx uniform;
    
        // Get input tensor to determine batch size
        Tensor<T>* input = tensors[input_seq].get();
        uint32_t batch_size = input->shape[0];
        uint32_t seq_len = input->shape[1];  // actual sequence length
        
        // tmp_qkv tensor: [B, T, 3 * n_heads * d_model]
        // Generate Q, K, V through linear projection
        tensor_linear(tmp_qkv, input_seq, W_qkv);
        
        // Calculate head dimension
        uint32_t d_k = d_model / n_heads;  // head dimension
        
        // Memory layout strides for the combined QKV buffer
        // Shape is [B, T, 3 * n_heads * d_k]
        uint32_t stride_qkv = 3 * n_heads * d_k;  // stride along the feature dimension
        uint32_t stride_seq = stride_qkv;          // stride along sequence dimension
        uint32_t stride_batch = max_seq_len * stride_seq;  // stride along batch dimension
        
        // Base address of the combined QKV buffer
        VkDeviceAddress base = tensors[tmp_qkv]->dataBuffer->getBufferAddress();
        VkDeviceAddress gradientBase = tensors[tmp_qkv]->gradientBuffer->getBufferAddress();
        
        // Q, K, V are interleaved in memory as [Q_all_heads, K_all_heads, V_all_heads]
        // Q starts at offset 0
        // K starts at offset n_heads * d_k
        // V starts at offset 2 * n_heads * d_k
        
        // Configure tensor_impl structures for Q, K, V
        // These point to the same buffer but with different offsets
        
        // Q tensor: shape [B, n_heads, T, d_k]
        uniform.Q.data = base;
        uniform.Q.grad = gradientBase;  // Q starts at the beginning
        uniform.Q.num_dims = 3;
        uniform.Q.requires_gradient = 1;
        
        // K tensor: shape [B, n_heads, T, d_k]
        uniform.K.data = base + sizeof(float) * n_heads * d_k;
        uniform.K.grad = gradientBase + sizeof(float) * n_heads * d_k;  // K offset
        uniform.K.num_dims = 3;
        uniform.K.requires_gradient = 1;
        
        // V tensor: shape [B, n_heads, T, d_k]
        uniform.V.data = base + sizeof(float) * 2 * n_heads * d_k;
        uniform.V.grad = gradientBase + sizeof(float) * 2 * n_heads * d_k;  // V offset
        uniform.V.num_dims = 3;
        uniform.V.requires_gradient = 1;

        // Create output tensor: [B, n_heads, T, d_k]
        uniform.Out = tensors[output]->getTensorImpl();
        
        // Create L and M buffers for softmax statistics
        // Shape: [B * n_heads, T]
        uniform.L = L_buffer;
        uniform.M = M_buffer;
        
        // Set dimensions
        uniform.N_CTX = max_seq_len;  // context length (max sequence length)
        uniform.Z = batch_size;       // batch size
        uniform.H = n_heads;          // number of heads
        
        // Softmax scale: 1/sqrt(d_k)
        uniform.sm_scale = 1.0f / std::sqrt(static_cast<float>(d_k));
        
        // Calculate strides for Q, K, V, O tensors
        // The layout in memory is [B, T, 3*n_heads*d_k] but we view it as [B, n_heads, T, d_k]
        
        // Q strides: [B, n_heads, T, d_k]
        uniform.stride_qk = 1;                           // stride along d_k dimension
        uniform.stride_qm = stride_seq;                  // stride along T (sequence) dimension
        uniform.stride_qh = d_k;                         // stride along n_heads dimension
        uniform.stride_qz = stride_batch;                // stride along B (batch) dimension
        
        // K strides: [B, n_heads, T, d_k] (same layout as Q)
        uniform.stride_kk = 1;                           // stride along d_k dimension
        uniform.stride_kn = stride_seq;                  // stride along T dimension
        uniform.stride_kh = d_k;                         // stride along n_heads dimension
        uniform.stride_kz = stride_batch;                // stride along B dimension
        
        // V strides: [B, n_heads, d_k, T] - note different ordering
        uniform.stride_vn = 1;                           // stride along T dimension (innermost)
        uniform.stride_vk = stride_seq;                  // stride along d_k dimension
        uniform.stride_vh = d_k;                         // stride along n_heads dimension
        uniform.stride_vz = stride_batch;                // stride along B dimension
        
        // O strides: [B, n_heads, T, d_k] - standard output layout
        uniform.stride_on = 1;                           // stride along d_k dimension
        uniform.stride_om = d_k;                         // stride along T dimension
        uniform.stride_oh = max_seq_len * d_k;           // stride along n_heads dimension
        uniform.stride_oz = n_heads * max_seq_len * d_k; // stride along B dimension
        
        // Calculate dispatch dimensions
        // Based on Triton: grid = (triton.cdiv(q.shape[2], BLOCK), q.shape[0] * q.shape[1])
        // q.shape = [B, n_heads, T, d_k]
        // So grid = (cdiv(T, BLOCK), B * n_heads)
        
        const uint32_t BLOCK = 128;  // BLOCK_M = BLOCK_N = 128
        
        
        // Dispatch will be: (grid_x, grid_y, grid_z)
        // Each workgroup processes a BLOCK x BLOCK tile of the attention matrix
        // for a specific (batch, head) combination
        
        uint32_t grid_x = (max_seq_len + BLOCK - 1) / BLOCK;  // ceiling division
        uint32_t grid_y = batch_size * n_heads;
        uint32_t grid_z = 1;
        uint32_t wrkgrp[3] = {grid_x, grid_y, grid_z};
        tensor_push_const p;

        // forward pass
        p.uniformAddress = flashAttention.uniformBuffer->getBufferAddress();
        flashAttention.loadUniform(uniform, p);
        flashAttention.execute(wrkgrp);
    };

    // Backward preprocess kernel - computes delta and normalizes dO in-place
    void tensor_flash_attention_bwd_preprocess(
        const std::string& output_name,      // O tensor from forward pass
        VkDeviceAddress L_buffer,            // L from forward pass
        VkDeviceAddress Delta_buffer,        // Delta output buffer
        uint32_t d_model,
        uint32_t max_seq_len,
        uint32_t n_heads)
    {
        tensor_flash_attention_bwd_preprocess_ctx uniform;
        
        Tensor<T>* output = tensors[output_name].get();
        uint32_t batch_size = output->shape[0];
        uint32_t d_k = d_model / n_heads;
        
        // Get O tensor (we use O.data for forward values, O.grad for dO which gets normalized in-place)
        uniform.Out = output->getTensorImpl();
        
        // L buffer from forward pass
        uniform.L = L_buffer;
        
        // Delta buffer (will be written to)
        uniform.Delta = Delta_buffer;
        
        // Set dimensions
        uniform.N_CTX = max_seq_len;
        uniform.Z = batch_size;
        uniform.H = n_heads;
        
        // O strides: [B, n_heads, T, d_k]
        uniform.stride_on = 1;                           // stride along d_k dimension
        uniform.stride_om = d_k;                         // stride along T dimension
        uniform.stride_oh = max_seq_len * d_k;           // stride along n_heads dimension
        uniform.stride_oz = n_heads * max_seq_len * d_k; // stride along B dimension
        
        // Calculate dispatch dimensions
        // Grid from forward: (cdiv(T, BLOCK), B * n_heads)
        const uint32_t BLOCK = 128;
        uint32_t grid_x = (max_seq_len + BLOCK - 1) / BLOCK;
        uint32_t grid_y = batch_size * n_heads;
        
        // Preprocess uses grid[0] * grid[1] workgroups
        uint32_t grid_total = grid_x * grid_y;
        uint32_t wrkgrp[3] = {grid_total, 1, 1};
        
        tensor_push_const p;
        p.uniformAddress = flashAttentionBackwardPreprocess.uniformBuffer->getBufferAddress();
        flashAttentionBackwardPreprocess.loadUniform(uniform, p);
        flashAttentionBackwardPreprocess.execute(wrkgrp);
    }

    // Backward kernel - computes dQ, dK, dV gradients
    void tensor_flash_attention_bwd(
        const std::string& qkv_combined_name, // Combined QKV buffer from forward
        const std::string& output_name,       // O tensor (normalized dO is in O.grad after preprocess)
        VkDeviceAddress M_buffer,             // M from forward pass
        VkDeviceAddress Delta_buffer,         // Delta from preprocess
        uint32_t d_model,
        uint32_t max_seq_len,
        uint32_t n_heads)
    {
        tensor_flash_attention_bwd_ctx uniform;
        
        Tensor<T>* qkv_tensor = tensors[qkv_combined_name].get();
        Tensor<T>* output = tensors[output_name].get();
        uint32_t batch_size = qkv_tensor->shape[0];
        uint32_t d_k = d_model / n_heads;
        
        // Memory layout strides for the combined QKV buffer
        // Shape is [B, T, 3 * n_heads * d_k]
        uint32_t stride_qkv = 3 * n_heads * d_k;
        uint32_t stride_seq = stride_qkv;
        uint32_t stride_batch = max_seq_len * stride_seq;
        
        // Base addresses
        VkDeviceAddress base = qkv_tensor->dataBuffer->getBufferAddress();
        VkDeviceAddress gradientBase = qkv_tensor->gradientBuffer->getBufferAddress();
        
        // Q tensor: forward values in Q.data, gradients written to Q.grad
        uniform.Q.data = base;
        uniform.Q.grad = gradientBase;
        uniform.Q.num_dims = 3;
        uniform.Q.requires_gradient = 1;
        
        // K tensor: forward values in K.data, gradients written to K.grad
        uniform.K.data = base + sizeof(float) * n_heads * d_k;
        uniform.K.grad = gradientBase + sizeof(float) * n_heads * d_k;
        uniform.K.num_dims = 3;
        uniform.K.requires_gradient = 1;
        
        // V tensor: forward values in V.data, gradients written to V.grad
        uniform.V.data = base + sizeof(float) * 2 * n_heads * d_k;
        uniform.V.grad = gradientBase + sizeof(float) * 2 * n_heads * d_k;
        uniform.V.num_dims = 3;
        uniform.V.requires_gradient = 1;
        
        // Out tensor: normalized dO gradient lives in Out.grad after preprocess
        uniform.Out = output->getTensorImpl();
        
        // M and Delta buffers
        uniform.M = M_buffer;
        uniform.Delta = Delta_buffer;
        
        // Set dimensions and scale
        float sm_scale = 1.0f / std::sqrt(static_cast<float>(d_k));
        uniform.sm_scale = sm_scale;
        uniform.N_CTX = max_seq_len;
        uniform.Z = batch_size;
        uniform.H = n_heads;
        
        // Q strides: [B, n_heads, T, d_k] - viewing the [B, T, 3*n_heads*d_k] buffer
        uniform.stride_qk = 1;                    // stride along d_k dimension
        uniform.stride_qm = stride_seq;           // stride along T (sequence) dimension
        uniform.stride_qh = d_k;                  // stride along n_heads dimension
        uniform.stride_qz = stride_batch;         // stride along B (batch) dimension
        
        // K strides: [B, n_heads, T, d_k] (same layout as Q)
        uniform.stride_kk = 1;                    // stride along d_k dimension
        uniform.stride_kn = stride_seq;           // stride along T dimension
        uniform.stride_kh = d_k;                  // stride along n_heads dimension
        uniform.stride_kz = stride_batch;         // stride along B dimension
        
        // V strides: [B, n_heads, T, d_k] (same layout as Q and K)
        uniform.stride_vn = 1;                    // stride along d_k dimension (innermost)
        uniform.stride_vk = stride_seq;           // stride along T dimension
        uniform.stride_vh = d_k;                  // stride along n_heads dimension
        uniform.stride_vz = stride_batch;         // stride along B dimension
        
        // Calculate dispatch dimensions
        const uint32_t BLOCK = 128;
        uint32_t grid_x = (max_seq_len + BLOCK - 1) / BLOCK;
        uint32_t grid_y = batch_size * n_heads;
        
        // Store num_block (grid[0] from forward pass)
        uniform.num_block = grid_x;
        
        // Backward kernel dispatches grid_y workgroups (one per batch * head)
        uint32_t wrkgrp[3] = {grid_y, 1, 1};
        
        tensor_push_const p;
        p.uniformAddress = flashAttentionBackward.uniformBuffer->getBufferAddress();
        flashAttentionBackward.loadUniform(uniform, p);
        flashAttentionBackward.execute(wrkgrp);
    }

    // Combined backward pass function
    void tensor_flash_attention_backward(
        const std::string& qkv_combined_name, // Combined QKV buffer from forward
        const std::string& output_name,       // O tensor from forward pass (dO is in O.grad)
        VkDeviceAddress L_buffer,             // L from forward pass
        VkDeviceAddress M_buffer,             // M from forward pass
        VkDeviceAddress Delta_buffer,         // Delta buffer (allocated by caller)
        uint32_t d_model,
        uint32_t max_seq_len,
        uint32_t n_heads)
    {
        // Step 1: Backward preprocess
        // This normalizes dO in-place (stored in Out.grad) and computes Delta
        tensor_flash_attention_bwd_preprocess(
            output_name,
            L_buffer,
            Delta_buffer,
            d_model,
            max_seq_len,
            n_heads
        );
        
        // Step 2: Backward kernel
        // This computes dQ, dK, dV using the normalized dO from Out.grad
        tensor_flash_attention_bwd(
            qkv_combined_name,
            output_name,
            M_buffer,
            Delta_buffer,
            d_model,
            max_seq_len,
            n_heads
        );
        
        // Gradients are now computed:
        // - dQ is at gradientBase + 0
        // - dK is at gradientBase + sizeof(float) * n_heads * d_k
        // - dV is at gradientBase + sizeof(float) * 2 * n_heads * d_k
        // All in the qkv_tensor->gradientBuffer
    }


    // Automatically populates predicted tensor's gradient buffer with a single call. No need to call .backward()
    void mse_loss(const std::string& predicted, const std::string& target, const std::string& loss) {
        mse_loss_context uniform;
        
        // Get tensor implementations
        uniform.loss_tensor = tensors[loss]->getTensorImpl();
        uniform.predicted_tensor = tensors[predicted]->getTensorImpl();
        uniform.target_tensor = tensors[target]->getTensorImpl();
        
        // Extract shape dimensions and pad missing dims with 1s so we always have [B, C, H, W]
        auto shape = tensors[target]->shape;
        if (shape.empty()) {
            throw std::invalid_argument("Target tensor shape cannot be empty for MSE loss");
        }

        // Normalize common cases:
        // - rank == 4: [B, C, H, W] (no change)
        // - rank == 3: assume missing channel -> [B, 1, H, W]
        // - rank == 2: assume [H, W] -> [1, 1, H, W]
        // - rank == 1: assume [W] -> [1, 1, 1, W]
        // This preserves spatial (last) dims and sets unspecified dims to 1.
        if (shape.size() < 4) {
            std::vector<uint32_t> padded(4, 1);
            size_t orig = shape.size();
            for (size_t i = 0; i < orig; ++i) {
                // copy original dims into the trailing positions
                padded[4 - orig + i] = shape[i];
            }
            // Special-case rank==3: prefer interpreting as [B, C, H] vs [B, H, W].
            // Heuristic: if original size == 3, treat as [B, C, H/W] -> map to [B, C, H, W] by inserting channel=1 only if ambiguous.
            if (orig == 3) {
                // Try to detect whether the first dimension looks like a batch (>=1).
                // If the first value seems like batch (>=1) and third is likely spatial, assume [B, C, H] -> insert channel=1 in middle:
                // Interpret original as either [B, H, W] or [B, C, H]. To preserve common case [B, H, W], map to [B, 1, H, W].
                // That corresponds to padded = {shape[0], 1, shape[1], shape[2]}
                padded = { shape[0], 1u, shape[1], shape[2] };
            }
            shape = std::move(padded);
        }

        // Now use the (possibly) padded shape
        uniform.batch_size = shape[0];  // B
        uniform.channels = shape[1];    // C
        uniform.height = shape[2];      // H
        uniform.width = shape[3];       // W
        
        // Calculate elements per batch: C * H * W
        uint32_t elements_per_batch = uniform.channels * uniform.height * uniform.width;
        
        // Dispatch configuration
        constexpr uint32_t VEC_TILE_SIZE = 4;      // As defined in shader
        constexpr uint32_t ELEMENTS_PER_VEC4 = 4;
        constexpr uint32_t TILE = VEC_TILE_SIZE * ELEMENTS_PER_VEC4;  // 16 elements per thread
        constexpr uint32_t GRP = 256;              // Workgroup size
        
        // Calculate workgroups needed per batch element
        // Each workgroup has GRP threads, each processing TILE elements
        uint32_t num_workgroups_x = (elements_per_batch + (GRP * TILE) - 1) / (GRP * TILE);
        
        // Dispatch: (workgroups_per_batch, 1, batch_size)
        // Each Z slice handles one batch element
        uint32_t workgroup[3] = { num_workgroups_x, 1, uniform.batch_size };
        
        DEBUG_PRINT("MSE loss on tensor: " << predicted 
                    << " Using targets: " << target 
                    << " Shape: [" << uniform.batch_size << ", " << uniform.channels 
                    << ", " << uniform.height << ", " << uniform.width << "]"
                    << " Dispatch: (" << num_workgroups_x << ", 1, " << uniform.batch_size << ")");
        
        // Setup push constants and execute
        tensor_push_const push;
        push.uniformAddress = mseLossShader.uniformBuffer->getBufferAddress();
        push.grid_size = { workgroup[0], workgroup[1], workgroup[2] };
        push.mode = 0;
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
        uniform.beta = 0.001f;

        uint32_t grpx = (uniform.elements_per_batch + (256 * 4) - 1)/(256 * 4);
        uint32_t wrkgrp[3] = {grpx, 1, uniform.batch_size};

        DEBUG_PRINT("KLD loss on tensor: " << mu_tensor << ", " << logvar_tensor << " with dispatch: (" << grpx << ", " << "1, " << uniform.batch_size << ")");

        tensor_push_const push;
        push.mode = 0;
        push.uniformAddress = kldLossShader.uniformBuffer->getBufferAddress();
        kldLossShader.loadUniform(uniform, push);
        kldLossShader.execute(wrkgrp);
    }

    void tensor_fill_random(const std::string& tensor, uint32_t init_type, uint32_t fan_in, uint32_t fan_out, T param1, T param2) {
        tensor_fill_rand_uniform_address<T> uniform{};
        
        uniform.tensor = tensors[tensor]->getTensorImpl();
        
        // Set initialization type and parameters
        uniform.init_type = init_type;  // 0=Uniform, 1=Normal, 2=Xavier, 3=He, 4=Constant
        uniform.fan_in = fan_in;
        uniform.fan_out = fan_out;
        uniform.param1 = param1;
        uniform.param2 = param2;
        
        // Random seed
        std::random_device rd;
        uniform.seed = static_cast<uint>(rd());
        
        tensor_push_const p;
        p.uniformAddress = fillRandomShader.uniformBuffer->getBufferAddress();
        
        uint32_t total_elements = std::accumulate(tensors[tensor]->shape.begin(), tensors[tensor]->shape.end(), 1, std::multiplies<uint32_t>());
        
        // Compute dispatch grid to cover the tensor
        auto ceilDiv = [](uint32_t a, uint32_t b) { return (a + b - 1) / b; };
        uint32_t groupX = ceilDiv(total_elements, 256u);
        uint32_t groupY = 1u;
        uint32_t groupZ = 1u;
        
        p.grid_size = glm::uvec3(groupX, groupY, groupZ);
        
        fillRandomShader.loadUniform(uniform, p);
        
        std::array<uint32_t, 3> workgroup = { groupX, groupY, groupZ };
        DEBUG_PRINT("workgroup: " << workgroup[0] << " " << workgroup[1] << " " << workgroup[2] << " for tensor: " << tensor);
        
        fillRandomShader.execute(workgroup.data());
    }

    // He initialization for Conv2d/Linear with ReLU
    void init_he(const std::string& tensor, uint32_t fan_in) {
        tensor_fill_random(tensor, 3, fan_in, 0, 0.0f, 0.0f);
    }

    // Xavier initialization for layers with TanH/Sigmoid
    void init_xavier(const std::string& tensor, uint32_t fan_in, uint32_t fan_out) {
        tensor_fill_random(tensor, 2, fan_in, fan_out, 0.0f, 0.0f);
    }

    // Constant initialization (for biases, BatchNorm params)
    void init_constant(const std::string& tensor, float value) {
        tensor_fill_random(tensor, 4, 0, 0, value, 0.0f);
    }

    // Uniform initialization (if you still need it)
    void init_uniform(const std::string& tensor, float min, float max) {
        tensor_fill_random(tensor, 0, 0, 0, min, max);
    }

    // Normal/Gaussian initialization with custom mean and stddev
    void init_normal(const std::string& tensor, float mean, float stddev) {
        tensor_fill_random(tensor, 1, 0, 0, mean, stddev);
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

    std::vector<T> get_highest_classes_from_dist(const std::string& dist){
        auto dist_tensor = tensors[dist].get();
        auto& local_tensor = createTensor({dist_tensor->shape[0]}, "temp_local_tensor");
        tensor_sample_context ctx;
        ctx.input_tensor = dist_tensor->getTensorImpl();
        ctx.output_tensor = local_tensor.getTensorImpl();
        ctx.M = dist_tensor->shape[1];
        ctx.N = dist_tensor->shape[dist_tensor->shape.size() - 1];
        auto cieldiv = [](uint32_t a, uint32_t b) { return (a + b - 1) / b; };
        uint32_t grpx = cieldiv(ctx.N, 256u);
        uint32_t grpz = dist_tensor->shape[0];
        uint32_t wrkgrp[3] = {grpx, 1, grpz};
        tensor_push_const p;
        p.uniformAddress = sampleShader.uniformBuffer->getBufferAddress();
        sampleShader.loadUniform(ctx, p);
        sampleShader.execute(wrkgrp);
        auto data = local_tensor.dataBuffer->downloadBuffer();
        destroy_tensor(local_tensor.name);
        return data;
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
    gpuTaskNoDesc<T, tensor_push_const, tanh_context> tanhShader; // computes tanh(input)
    gpuTaskNoDesc<T, tensor_push_const, tanh_context> tanhShaderBackward;
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
    gpuTaskNoDesc<T, tensor_push_const, tensor_flash_attention_fwd_ctx> flashAttention; // computes flash attention forward pass
    gpuTaskNoDesc<T, tensor_push_const, tensor_flash_attention_bwd_preprocess_ctx> flashAttentionBackwardPreprocess; // computes flash attention backward preprocess pass
    gpuTaskNoDesc<T, tensor_push_const, tensor_flash_attention_bwd_ctx> flashAttentionBackward; // computes flash attention backward pass
    gpuTaskNoDesc<T, tensor_push_const, embedding_lookup_context> embedLookupShader; // computes embedding lookup
    gpuTaskNoDesc<T, tensor_push_const, embedding_lookup_context> embedLookupShaderBackward;
    gpuTaskNoDesc<T, tensor_push_const, tensor_upsample_context> upsampleShader; // upsamples image tensors [B, C, H, W] using bilinear interpolation
    gpuTaskNoDesc<T, tensor_push_const, tensor_upsample_context> upsampleShaderBackward;
    gpuTaskNoDesc<T, tensor_push_const, tensor_cmp_context> cmpShader; // compares two tensors for equality
    gpuTaskNoDesc<T, tensor_push_const, mean_context> mean_shader; // compute the mean of a tensor
    gpuTaskNoDesc<T, tensor_push_const, tensor_sample_context> sampleShader; // returns class index of highest probability
    gpuTaskNoDesc<T, tensor_push_const, tensor_fill_rand_uniform_address<T>> fillRandomShader; // fills tensor with random values
    gpuTaskNoDesc<T, tensor_push_const, tensor_conv2d_context> conv2dShader;    // generalized conv2d shader. Supports any kernel size up to 15x15
    gpuTaskNoDesc<T, tensor_push_const, tensor_conv2d_context> conv2dShaderBackward;    // generalized conv2d backward shader. Supports any kernel size up to 15x15
    gpuTaskNoDesc<T, tensor_push_const, tensor_transposed_conv2d_context> transposedConv2dShader;   // generalized transposed conv2d shader.
    gpuTaskNoDesc<T, tensor_push_const, tensor_transposed_conv2d_context> transposedConv2dShaderBackward;   // generalized transposed conv2d backward shader.
    gpuTaskNoDesc<T, tensor_push_const, tensor_max_pool_context> maxPoolShader;     // computes maxPool2D with any kernel shape
    gpuTaskNoDesc<T, tensor_push_const, tensor_max_pool_context> maxPoolShaderBackward;     // computes maxPool2D's backward pass with any kernel shape
};

template<typename T>
Tensor<T>& Tensor<T>::operator*(Tensor<T>& other) {
    auto &output = pool->createTensor(this->shape,
        name + other.name + "-elementwise_multiply_output", true);

    Tensor<T>* self_ptr = this;
    Tensor<T>* oth_ptr  = &other;
    Tensor<T>* out_ptr  = &output;
    auto pool_ptr       = this->pool;

    output.back.push_back([self_ptr, oth_ptr, out_ptr, pool_ptr]() {
        pool_ptr->tensor_multiply_elementwise(oth_ptr->name, self_ptr->name, out_ptr->name, 1);
        self_ptr->backward();
        oth_ptr->backward();
        out_ptr->isClaimed = false;
    });

    pool->tensor_multiply_elementwise(other.name, name, output.name);
    return output;
}

template<typename T>
Tensor<T>& Tensor<T>::operator*(T other){
    auto &output = pool->createTensor(this->shape,
        name + "-elementwise_multiply_output_for_" + std::to_string(other), true);

    auto &to_mul = pool->createTensor(
        std::vector<uint32_t>(this->shape.size(), 1),
        "const_mul_tensor_" + name + "_" + std::to_string(other), true
    );
    to_mul.dataBuffer->set(0, other);

    Tensor<T>* self_ptr = this;
    Tensor<T>* out_ptr  = &output;
    Tensor<T>* mul_ptr  = &to_mul;
    auto pool_ptr       = this->pool;

    output.back.push_back([self_ptr, out_ptr, mul_ptr, pool_ptr]() {
        pool_ptr->tensor_multiply_elementwise(mul_ptr->name, self_ptr->name, out_ptr->name, 1);
        self_ptr->backward();
        out_ptr->isClaimed = false;
        mul_ptr->isClaimed = false;
    });

    pool->tensor_multiply_elementwise(to_mul.name, this->name, output.name);
    return output;
}

template<typename T>
Tensor<T>& operator*(T lhs, Tensor<T>& rhs) {
    auto &output = rhs.pool->createTensor(
        rhs.shape,
        std::to_string(lhs) + "_" + rhs.name + "-elementwise_multiply_output", true
    );

    auto &to_mul = rhs.pool->createTensor(
        std::vector<uint32_t>(rhs.shape.size(), 1),
        "const_mul_tensor_" + std::to_string(lhs) + "_" + rhs.name, true
    );
    to_mul.dataBuffer->set(0, lhs);

    Tensor<T>* rhs_ptr = &rhs;
    Tensor<T>* out_ptr = &output;
    Tensor<T>* mul_ptr = &to_mul;
    auto pool_ptr      = rhs.pool;

    output.back.push_back([rhs_ptr, out_ptr, mul_ptr, pool_ptr]() {
        pool_ptr->tensor_multiply_elementwise(mul_ptr->name, rhs_ptr->name, out_ptr->name, 1);
        rhs_ptr->backward();
        mul_ptr->isClaimed = false;
        out_ptr->isClaimed = false;
    });

    rhs.pool->tensor_multiply_elementwise(to_mul.name, rhs.name, output.name);
    return output;
}

template<typename T>
Tensor<T>& Tensor<T>::operator+(Tensor<T>& other) {
    auto &output = pool->createTensor(
        this->shape,
        name + other.name + "-elementwise_addition_output", true
    );

    Tensor<T>* self_ptr = this;
    Tensor<T>* oth_ptr  = &other;
    Tensor<T>* out_ptr  = &output;
    auto pool_ptr       = this->pool;

    output.back.push_back([self_ptr, oth_ptr, out_ptr, pool_ptr]() {
        pool_ptr->tensor_add_inplace(oth_ptr->name, self_ptr->name, out_ptr->name, 1);
        self_ptr->backward();
        oth_ptr->backward();
        out_ptr->isClaimed = false;
    });

    pool->tensor_add_inplace(other.name, name, output.name);
    return output;
}

template<typename T>
Tensor<T>& Tensor<T>::operator+(T other) {
    auto &output = pool->createTensor(this->shape,
        name + "_" + std::to_string(other) + "-elementwise_addition_output", true);

    auto &to_add = pool->createTensor(
        std::vector<uint32_t>(this->shape.size(), 1),
        "const_add_tensor_" + name + "_" + std::to_string(other), true
    );
    to_add.dataBuffer->set(0, other);

    Tensor<T>* self_ptr = this;
    Tensor<T>* out_ptr  = &output;
    Tensor<T>* add_ptr  = &to_add;
    auto pool_ptr       = this->pool;

    output.back.push_back([self_ptr, out_ptr, add_ptr, pool_ptr]() {
        pool_ptr->tensor_add_inplace(add_ptr->name, self_ptr->name, out_ptr->name, 1);
        out_ptr->isClaimed = false;
        add_ptr->isClaimed = false;
    });

    pool->tensor_add_inplace(to_add.name, name, output.name);
    return output;
}

template<typename T>
Tensor<T>& operator+(T lhs, Tensor<T>& rhs){
    auto &output = rhs.pool->createTensor(
        rhs.shape,
        std::to_string(lhs) + "_" + rhs.name + "-elementwise_addition_output", true
    );

    auto &to_add = rhs.pool->createTensor(
        std::vector<uint32_t>(rhs.shape.size(), 1),
        "const_add_tensor_" + std::to_string(lhs) + "_" + rhs.name, true
    );
    to_add.dataBuffer->set(0, lhs);

    Tensor<T>* rhs_ptr = &rhs;
    Tensor<T>* out_ptr = &output;
    Tensor<T>* add_ptr = &to_add;
    auto pool_ptr      = rhs.pool;

    output.back.push_back([rhs_ptr, out_ptr, add_ptr, pool_ptr]() {
        pool_ptr->tensor_add_inplace(add_ptr->name, rhs_ptr->name, out_ptr->name, 1);
        rhs_ptr->backward();
        out_ptr->isClaimed = false;
        add_ptr->isClaimed = false;
    });

    rhs.pool->tensor_add_inplace(to_add.name, rhs.name, output.name);
    return output;
}

template<typename T>
Tensor<T>& Tensor<T>::exp() {
    auto &output = pool->tensor_exp(name);

    Tensor<T>* self_ptr = this;
    Tensor<T>* out_ptr  = &output;
    auto pool_ptr       = this->pool;

    output.back.push_back([self_ptr, out_ptr, pool_ptr]() {
        pool_ptr->tensor_exp(self_ptr->name, out_ptr->name, 1);
        self_ptr->backward();
        out_ptr->isClaimed = false;
    });

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

    {
        std::string out_name  = output.name;
        std::string self_name = this->name;
        std::string oth_name  = other.name;

        Tensor *self  = this;
        Tensor *oth   = &other;

        output.back.push_back([self, oth, out_name, self_name, oth_name]() {
            self->pool->tensor_linear(out_name, self_name, oth_name, "", 1);
            self->backward();
            oth->backward();
        });
    }

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