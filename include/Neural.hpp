#pragma once

#include <vector>
#include <string>
#include "VkMemAlloc.hpp"
#include "Tensor.hpp"

template<typename T>
struct Module {
	Module<T>* next;
	Module<T>* prev;
	Tensor<T>* output;
	virtual Tensor<T>* forward(Tensor<T>* input) = 0;
	virtual void backward(Tensor<T>* input) = 0;
	Tensor<T>* operator()(Tensor<T>* input) {
		return forward(input);
	}
	virtual std::vector<Tensor<T>*> getTrainableTensors() {return {};}
	virtual std::vector<Tensor<T>*> getIntermediateTensors() {return {};}
	virtual size_t getTrainableParamCount() {return 0;}
	virtual ~Module() = default;
	bool requires_target = false; // Whether this module needs target labels (e.g., for loss functions)
};

template<typename T>
struct Optimiser {
	virtual void step() = 0;
	virtual ~Optimiser() = default;
};

template<typename T>
struct LinearReLU : public Module<T>{

	Tensor<T>* weights; // Weight matrix
	Tensor<T>* bias;    // Bias vector (not used rn)
	//Tensor<T>* output;
	TensorPool<T>* tensorPool;
	std::string name;
	std::string weights_name;
	std::string bias_name;
	std::string output_name;

	LinearReLU(TensorPool<T>* pool, uint32_t in_features, uint32_t out_features, uint32_t batch_size, const std::string& name) : tensorPool(pool), name(name) {
		// Initialize weights and bias tensors
		std::vector<uint32_t> weight_shape = { 1, out_features, in_features };
		std::vector<uint32_t> output_shape = { batch_size, 1, out_features };
		weights_name = name + "-weights";
		bias_name = name + "-bias";
		output_name = name + "-output";

		// store the transposed shape for weights
		weights = &tensorPool->createTensor({weight_shape[0], weight_shape[1], weight_shape[2]}, weights_name);
		bias = &tensorPool->createTensor(output_shape, bias_name);
		this->output = &tensorPool->createTensor(output_shape, output_name);
		tensorPool->tensor_fill_random(weights_name, -0.1f, 0.1f);
		tensorPool->tensor_fill_random(bias_name, -0.1f, 0.1f);
	}

	LinearReLU(TensorPool<T>* pool, const std::vector<uint32_t>& weight_dims, const std::vector<uint32_t>& output_dims, const std::string& name) : tensorPool(pool), name(name) {
		// Initialize weights and bias tensors

		weights_name = name + "-weights";
		bias_name = name + "-bias";
		output_name = name + "-output";

		weights = &tensorPool->createTensor(weight_dims, weights_name);
		bias = &tensorPool->createTensor(output_dims, bias_name);
		this->output = &tensorPool->createTensor(output_dims, output_name);
		tensorPool->tensor_fill_random(weights_name, -0.1f, 0.1f);
		tensorPool->tensor_fill_random(bias_name, -0.1f, 0.1f);
	}

	LinearReLU(){}; // default empty ctor

	std::vector<Tensor<T>*> getTrainableTensors() override {
		return {weights, bias};
	}
	
	Tensor<T>* forward(Tensor<T>* input) override {
		// insert extra dim to satisfy kernel API
		auto o = input->shape;
		if(input->shape.size() == 2){
			input->view({input->shape[0], 1, input->shape[1]});
		}
		tensorPool->tensor_linear_ReLU(output_name, input->name, weights_name, bias_name, 0);
		input->view(o);
		return this->output;
	}
	void backward(Tensor<T>* input) override {
		auto o = input->shape;
		if(input->shape.size() == 2){
			input->view({input->shape[0], 1, input->shape[1]});
		}
		tensorPool->tensor_linear_ReLU(output_name, input->name, weights_name, bias_name, 1);
		input->view(o);
	}
};

template<typename T>
struct Linear : public Module<T> {
	Tensor<T>* weights; // Weight matrix
	Tensor<T>* bias;    // Bias vector
	//Tensor<T>* output;
	TensorPool<T>* tensorPool;
	std::string name;
	std::string weights_name;
	std::string bias_name;
	std::string output_name;

	Linear(TensorPool<T>* pool, uint32_t in_features, uint32_t out_features, uint32_t batch_size, const std::string& name) : tensorPool(pool), name(name) {
		// Initialize weights and bias tensors

		std::vector<uint32_t> weight_shape = { 1, out_features, in_features };
		std::vector<uint32_t> output_shape = { batch_size, 1, out_features };
		weights_name = name + "-weights";
		bias_name = name + "-bias";
		output_name = name + "-output";

		weights = &tensorPool->createTensor(weight_shape, weights_name);
		bias = &tensorPool->createTensor(output_shape, bias_name);
		this->output = &tensorPool->createTensor(output_shape, output_name);
		tensorPool->tensor_fill_random(weights_name, -0.1f, 0.1f);
		tensorPool->tensor_fill_random(bias_name, -0.1f, 0.1f);
	}

	Linear(TensorPool<T>* pool, const std::vector<uint32_t>& weight_dims, const std::vector<uint32_t>& output_dims, const std::string& name) : tensorPool(pool), name(name) {
		// Initialize weights and bias tensors

		weights_name = name + "-weights";
		bias_name = name + "-bias";
		output_name = name + "-output";

		weights = &tensorPool->createTensor(weight_dims, weights_name);
		bias = &tensorPool->createTensor({1, output_dims[1], output_dims[2]}, bias_name); // exclude the batch dim
		this->output = &tensorPool->createTensor(output_dims, output_name);
		tensorPool->tensor_fill_random(weights_name, -0.1f, 0.1f);
		tensorPool->tensor_fill_random(bias_name, -0.1f, 0.1f);
	}

	Linear(){}; // default empty ctor

	std::vector<Tensor<T>*> getTrainableTensors() override {
		return {weights, bias};
	}

	Tensor<T>* forward(Tensor<T>* input) override {
		// insert extra dim to satisfy kernel API
		auto o = input->shape;
		if(input->shape.size() == 2){
			input->view({input->shape[0], 1, input->shape[1]});
		}
		tensorPool->tensor_linear(output_name, input->name, weights_name, bias_name, 0);
		input->view(o); // return the tensor's original dims
		return this->output;
	}
	void backward(Tensor<T>* input) override {
		auto o = input->shape;
		if(input->shape.size() == 2){
			input->view({input->shape[0], 1, input->shape[1]});
		}
		tensorPool->tensor_linear(output_name, input->name, weights_name, bias_name, 1);
		input->view(o);
	}
};

template<typename T>
struct ReLU : public Module<T> {
	//Tensor<T>* output;
	TensorPool<T>* tensorPool;
	std::string name;
	std::string output_name;
	ReLU(TensorPool<T>* pool, uint32_t features, uint32_t batch_size, const std::string& name) : tensorPool(pool), name(name) {
		output_name = name + "-output";
		std::vector<uint32_t> shape = { batch_size, 1, features };
		tensorPool->createTensor(shape, output_name);
		this->output = &tensorPool->getTensor(output_name, { shape.begin(), shape.end() });
	}
	ReLU(TensorPool<T>* pool, const std::vector<uint32_t> dims, const std::string& name) : tensorPool(pool), name(name) {
		output_name = name + "-output";
		tensorPool->createTensor(dims, output_name);
		this->output = &tensorPool->getTensor(output_name, { dims.begin(), dims.end() });
	}
	ReLU(){}; // default empty ctor
	Tensor<T>* forward(Tensor<T>* input) override {
		// Ensure input shape is compatible
		if (input->shape != this->output->shape) {
			throw std::invalid_argument("Input tensor shape is not compatible with ReLU output");
		}
		// Apply ReLU activation function
		tensorPool->tensor_ReLU(output_name, input->name);
		return this->output;
	}
	void backward(Tensor<T>* input) override {
		tensorPool->tensor_ReLU(output_name, input->name, 1);
	}
};

template<typename T>
struct Dropout : public Module<T> {
	//Tensor<T>* output;
	TensorPool<T>* tensorPool;
	std::string name;
	std::string output_name;
	float dropout_rate;
	Dropout(TensorPool<T>* pool, std::vector<uint32_t> shape, float rate, const std::string& name) : tensorPool(pool), name(name), dropout_rate(rate) {
		output_name = name + "-output";
		tensorPool->createTensor(shape, output_name);
		this->output = &tensorPool->getTensor(output_name, { shape.begin(), shape.end() });
	}
	Dropout(){}; // default empty ctor
	Tensor<T>* forward(Tensor<T>* input) override {
		// Ensure input shape is compatible
		if (input->shape != this->output->shape) {
			throw std::invalid_argument("Input tensor shape is not compatible with Dropout output");
		}
		// Apply Dropout (not implemented yet)
		std::cout << "Dropout not implemented yet\n";
		return this->output;
	}
	void backward(Tensor<T>* input) override {
		// Backward pass for Dropout (not implemented yet)
		std::cout << "Dropout backward not implemented yet\n";
	}
};

// Softmax followed by Cross-Entropy Loss. Provide one-hot encoded labels as input.
template<typename T>
struct SoftmaxCrossEntropy : public Module<T> {
	//Tensor<T>* output;
	TensorPool<T>* tensorPool;
	std::string name;
	std::string output_name;
	Tensor<T>* target = nullptr; // One-hot encoded target labels
	Tensor<T>* softmax_output = nullptr; // To store softmax probabilities
	
	SoftmaxCrossEntropy(TensorPool<T>* pool, uint32_t features, uint32_t channels, uint32_t batch_size, const std::string& name) : tensorPool(pool), name(name) {
		output_name = name + "-output";
		std::vector<uint32_t> shape = { batch_size, channels, features };
		this->output = &tensorPool->createTensor({ batch_size, channels }, output_name); // the loss tensor
		this->requires_target = true;
		softmax_output = &tensorPool->createTensor(shape, name + "-softmax"); // (B, M, N)
	}

	SoftmaxCrossEntropy(TensorPool<T>* pool, const std::vector<uint32_t>& dims, const std::string& name) : tensorPool(pool), name(name) {
		output_name = name + "-output";
		tensorPool->createTensor({dims[0], dims[1] }, output_name);
		this->output = &tensorPool->getTensor(output_name, { dims[0], dims[1] }); // the loss tensor
		softmax_output = &tensorPool->createTensor(dims, name + "-softmax"); // (B, M, N)
	}

	SoftmaxCrossEntropy(){}; //default empty ctor

	Tensor<T>* forward(Tensor<T>* input) override {
		tensorPool->tensor_cross_entropy(output_name, input->name, target->name, softmax_output->name, 0);
		return this->output;
	}

	void backward(Tensor<T>* input) override {
		if (target == nullptr) {
			throw std::runtime_error("Target tensor not set for SoftmaxCrossEntropy");
		}
		tensorPool->tensor_cross_entropy(output_name, input->name, target->name, softmax_output->name, 1);
	}
};

template<typename T>
struct BatchNorm1d : public Module<T> {
	//Tensor<T>* output;
	Tensor<T>* input_tensor;  // [B, M, N] - input feature vector
	Tensor<T>* weight_tensor; // [M] - weight (gamma)
	Tensor<T>* bias_tensor;   // [M] - bias (beta)
	Tensor<T>* running_mean;  // [M] - running mean (for inference)
	Tensor<T>* running_var;   // [M] - running variance (for inference)
	Tensor<T>* save_mean;     // [M] - saved mean (for backward)
	Tensor<T>* save_var;      // [M] - saved variance (for backward)
	uint mode;                // 0 = train, 1 = eval
	uint batch_size;
	uint accumulate_grad;     // 0: overwrite, 1: += for grads
	float momentum;         // momentum for running stats
	float eps;              // epsilon for numerical stability
	TensorPool<T>* tensorPool;
	std::string name;
	std::string output_name;
	BatchNorm1d(TensorPool<T>* pool, uint32_t features, uint32_t channels, uint32_t batch_size, const std::string& name) : tensorPool(pool), name(name) {
		output_name = name + "-output";
		std::vector<uint32_t> shape = { batch_size, channels, features };
		this->output = &tensorPool->createTensor(shape, output_name);
		
		weight_tensor = &tensorPool->createTensor({ channels, features }, name + "-weight");
		bias_tensor = &tensorPool->createTensor({ channels, features }, name + "-bias");
		running_mean = &tensorPool->createTensor({ channels, features }, name + "-running_mean");
		running_var = &tensorPool->createTensor({ channels, features }, name + "-running_var");
		save_mean = &tensorPool->createTensor({ channels, features }, name + "-save_mean");
		save_var = &tensorPool->createTensor({ channels, features }, name + "-save_var");

		tensorPool->tensor_fill_random(weight_tensor->name, 1.0f, 1.0f);
		// keep bias at zero at initialization
	}

	BatchNorm1d(TensorPool<T>* pool, const std::vector<uint32_t>& shape, const std::string& name) : tensorPool(pool), name(name) {
		output_name = name + "-output";
		this->output = &tensorPool->createTensor(shape, output_name);
		
		auto channels = shape[1];
		auto features = shape[2];

		weight_tensor = &tensorPool->createTensor({ channels, features }, name + "-weight");
		bias_tensor = &tensorPool->createTensor({ channels, features }, name + "-bias");
		running_mean = &tensorPool->createTensor({ channels, features }, name + "-running_mean");
		running_var = &tensorPool->createTensor({ channels, features }, name + "-running_var");
		save_mean = &tensorPool->createTensor({ channels, features }, name + "-save_mean");
		save_var = &tensorPool->createTensor({ channels, features }, name + "-save_var");

		tensorPool->tensor_fill_random(weight_tensor->name, 1.0f, 1.0f);
		// keep bias at zero at initialization
	}

	BatchNorm1d(){}; // default empty ctor

	std::vector<Tensor<T>*> getTrainableTensors() override {
		return {weight_tensor, bias_tensor};
	}

	Tensor<T>* forward(Tensor<T>* input) override {
		tensorPool->tensor_batchnorm_1d(input->name, weight_tensor->name, bias_tensor->name, running_mean->name, running_var->name, this->output->name, save_mean->name, save_var->name, 0);
		return this->output;
	}
	void backward(Tensor<T>* input) override {
		tensorPool->tensor_batchnorm_1d(input->name, weight_tensor->name, bias_tensor->name, running_mean->name, running_var->name, this->output->name, save_mean->name, save_var->name, 1);
	}
};

template<typename T>
struct BatchNorm2d : public Module<T> {
	Tensor<T>* input_tensor;    // [B, C, H, W]
    Tensor<T>* weight_tensor;   // [C]
    Tensor<T>* bias_tensor;     // [C]
    Tensor<T>* running_mean;    // [C]
    Tensor<T>* running_var;     // [C]
    //Tensor<T>* out_tensor;    // [B, C, H, W]
    Tensor<T>* save_mean;       // [C]
    Tensor<T>* save_var;        // [C]
	uint mode;                
	uint batch_size;
	uint accumulate_grad;     // 0: overwrite, 1: += for grads
	float momentum;         // momentum for running stats
	float eps;              // epsilon for numerical stability
	TensorPool<T>* tensorPool;
	std::string name;
	std::string output_name;

	BatchNorm2d(TensorPool<T>* tensorPool, uint32_t channels, uint32_t width, uint32_t height, uint32_t batch_size, const std::string& name): tensorPool(tensorPool), name(name) {
		output_name = name + "-output";
		std::vector<uint32_t> shape = { batch_size, channels, height, width };
		this->output = &tensorPool->createTensor(shape, output_name);
		
		weight_tensor = &tensorPool->createTensor({ channels }, name + "-weight");
		bias_tensor = &tensorPool->createTensor({ channels }, name + "-bias");
		running_mean = &tensorPool->createTensor({ channels }, name + "-running_mean");
		running_var = &tensorPool->createTensor({ channels }, name + "-running_var");
		save_mean = &tensorPool->createTensor({ channels }, name + "-save_mean");
		save_var = &tensorPool->createTensor({ channels }, name + "-save_var");

		tensorPool->tensor_fill_random(weight_tensor->name, 1.0f, 1.0f);
	}

	BatchNorm2d(TensorPool<T>* pool, const std::vector<uint32_t>& shape, const std::string& name) : tensorPool(pool), name(name) {
		output_name = name + "-output";
		this->output = &tensorPool->createTensor(shape, output_name);
		auto batch = shape[0];
		auto channels = shape[1];
		auto height = shape[2];
		auto width = shape[3];

		weight_tensor = &tensorPool->createTensor({ channels }, name + "-weight");
		bias_tensor = &tensorPool->createTensor({ channels }, name + "-bias");
		running_mean = &tensorPool->createTensor({ channels }, name + "-running_mean");
		running_var = &tensorPool->createTensor({ channels }, name + "-running_var");
		save_mean = &tensorPool->createTensor({ channels }, name + "-save_mean");
		save_var = &tensorPool->createTensor({ channels }, name + "-save_var");

		tensorPool->tensor_fill_random(weight_tensor->name, 1.0f, 1.0f);
		// keep bias at zero at initialization
	}

	BatchNorm2d(){}; // default empty ctor

	std::vector<Tensor<T>*> getTrainableTensors() override {
		return {weight_tensor, bias_tensor};
	}

	Tensor<T>* forward(Tensor<T>* input) override {
		tensorPool->tensor_batchnorm_2d(input->name, weight_tensor->name, bias_tensor->name, running_mean->name, running_var->name, this->output->name, save_mean->name, save_var->name, 0);
		return this->output;
	}
	void backward(Tensor<T>* input) override {
		tensorPool->tensor_batchnorm_2d(input->name, weight_tensor->name, bias_tensor->name, running_mean->name, running_var->name, this->output->name, save_mean->name, save_var->name, 1);
	}
};

template<typename T>
struct Layernorm1d : public Module<T> {
	//Tensor<T>* output;
	Tensor<T>* input_tensor;    // [B, M, N]
    Tensor<T>* weight_tensor;   // [N] - normalized shape
    Tensor<T>* bias_tensor;     // [N] - normalized shape
    Tensor<T>* save_mean;       // [B, M] - mean for each sample
    Tensor<T>* save_rstd;       // [B, M] - reciprocal std for each sample
	TensorPool<T>* tensorPool;
	std::string name;
	std::string output_name;
	Layernorm1d(TensorPool<T>* pool, uint32_t features, uint32_t channels, uint32_t batch_size, const std::string& name) : tensorPool(pool), name(name) {
		output_name = name + "-output";
		std::vector<uint32_t> shape = { batch_size, channels, features };
		this->output = &tensorPool->createTensor(shape, output_name);
		
		weight_tensor = &tensorPool->createTensor({ features }, name + "-weight");
		bias_tensor = &tensorPool->createTensor({ features }, name + "-bias");
		save_mean = &tensorPool->createTensor({ batch_size, channels }, name + "-save_mean");
		save_rstd = &tensorPool->createTensor({ batch_size, channels }, name + "-save_rstd");

		tensorPool->tensor_fill_random(weight_tensor->name, 1.0f, 1.0f);
		// keep bias at zero at initialization
	}

	Layernorm1d(TensorPool<T>* pool, const std::vector<uint32_t>& shape, const std::string& name) : tensorPool(pool), name(name) {
		output_name = name + "-output";
		this->output = &tensorPool->createTensor(shape, output_name);
		
		auto batch_size = shape[0];
		auto channels = shape[1];
		auto features = shape[2];

		weight_tensor = &tensorPool->createTensor({ features }, name + "-weight");
		bias_tensor = &tensorPool->createTensor({ features }, name + "-bias");
		save_mean = &tensorPool->createTensor({ batch_size, channels }, name + "-save_mean");
		save_rstd = &tensorPool->createTensor({ batch_size, channels }, name + "-save_rstd");

		tensorPool->tensor_fill_random(weight_tensor->name, 1.0f, 1.0f);
		// keep bias at zero at initialization
	}

	Layernorm1d(){}; // default empty ctor

	std::vector<Tensor<T>*> getTrainableTensors() override {
		return {weight_tensor, bias_tensor};
	}

	Tensor<T>* forward(Tensor<T>* input) override {
		tensorPool->tensor_layernorm_1d(input->name, weight_tensor->name, bias_tensor->name, this->output->name, save_mean->name, save_rstd->name, 0);
		return this->output;
	}
	void backward(Tensor<T>* input) override {
		tensorPool->tensor_layernorm_1d(input->name, weight_tensor->name, bias_tensor->name, this->output->name, save_mean->name, save_rstd->name, 1);
	}
};

template<typename T>
struct Conv2d3x3 : public Module<T> {
	//Tensor<T>* output;
	Tensor<T>* weight_tensor; // [C_out, C_in, K_h, K_w]
	Tensor<T>* bias_tensor;   // [C_out]
	TensorPool<T>* tensorPool;
	std::string name;
	std::string output_name;

	uint32_t stride_w = 1;
	uint32_t stride_h = 1;
	uint32_t pad_h = 1;
	uint32_t pad_w = 1;
	uint32_t dilation_h = 1;
	uint32_t dilation_w = 1;
	uint32_t groups = 1;

	uint32_t output_height;
	uint32_t output_width;

	Conv2d3x3(TensorPool<T>* pool, uint32_t in_channels, uint32_t out_channels, uint32_t batch_size, uint32_t height, uint32_t width, const std::string& name,
	uint32_t stride_w = 1, uint32_t stride_h = 1, uint32_t pad_h = 1, uint32_t pad_w = 1, uint32_t dilation_h = 1, uint32_t dilation_w = 1, uint32_t groups = 1) 
	: tensorPool(pool), name(name) {
		// Validate parameters
		if (pad_h > 2 || pad_w > 2) {
			throw std::invalid_argument("Padding must be 0, 1, or 2");
		}
		if (stride_h == 0 || stride_w == 0) {
			throw std::invalid_argument("Stride must be greater than 0");
		}
		if (dilation_h > 2 || dilation_w > 2) {
			throw std::invalid_argument("Dilation must be 1 or 2");
		}
		
		// Calculate effective kernel size with dilation
		uint32_t effective_kernel_h = 3 + (3 - 1) * (dilation_h - 1);
		uint32_t effective_kernel_w = 3 + (3 - 1) * (dilation_w - 1);
		
		// Calculate output dimensions
		this->output_height = ((height + 2 * pad_h - effective_kernel_h) / stride_h) + 1;
		this->output_width = ((width + 2 * pad_w - effective_kernel_w) / stride_w) + 1;
		
		// Validate output dimensions
		if (this->output_height < 1 || this->output_width < 1) {
			throw std::invalid_argument("Invalid combination of input size, padding, stride and dilation");
		}

		this->stride_w = stride_w;
		this->stride_h = stride_h;
		this->pad_w = pad_w;
		this->pad_h = pad_h;
		this->dilation_h = dilation_h;
		this->dilation_w = dilation_w;
		this->groups = groups;
		
		output_name = name + "-output";
		this->output = &tensorPool->createTensor({ batch_size, out_channels, output_height, output_width }, output_name);
		
		weight_tensor = &tensorPool->createTensor({ out_channels, in_channels, 3, 3 }, name + "-weight");
		bias_tensor = &tensorPool->createTensor({ out_channels }, name + "-bias");
		
		tensorPool->tensor_fill_random(weight_tensor->name, -1.0f, 1.0f);
		tensorPool->tensor_fill_random(bias_tensor->name, -1.0f, 1.0f);
	}

	Conv2d3x3(TensorPool<T>* pool, const std::vector<uint32_t>& weight_dims, const std::vector<uint32_t>& output_dims, const std::string& name) 
	: tensorPool(pool), name(name) {
		if (weight_dims[2] != 3 || weight_dims[3] != 3) {
			throw std::invalid_argument("Weight dimensions must be [out_channels, in_channels, 3, 3]");
		}
		
		output_name = name + "-output";
		this->output = &tensorPool->createTensor(output_dims, output_name);
		
		weight_tensor = &tensorPool->createTensor(weight_dims, name + "-weight");
		bias_tensor = &tensorPool->createTensor({ weight_dims[0] }, name + "-bias");
		
		tensorPool->tensor_fill_random(weight_tensor->name, -1.0f, 1.0f);
		tensorPool->tensor_fill_random(bias_tensor->name, -1.0f, 1.0f);
	}

	Conv2d3x3(){}; // default empty ctor

	std::vector<Tensor<T>*> getTrainableTensors() override {
		return {weight_tensor, bias_tensor};
	}

	// input tensor shape: [N, C_in, H_in, W_in]
	Tensor<T>* forward(Tensor<T>* input) override {
		// Ensure input tensor shape matches expected dimensions
		if (input->shape.size() != 4) {
			throw std::invalid_argument("Input tensor must have 4 dimensions: [N, C_in, H_in, W_in]");
		}
		if (input->shape[1] != weight_tensor->shape[1]) {
			throw std::invalid_argument("Input channels (C_in) must match weight tensor's input channels");
		}
		if (input->shape[2] < 3 || input->shape[3] < 3) {
			throw std::invalid_argument("Input height and width must be at least 3 to apply a 3x3 convolution");
		}

		tensorPool->tensor_conv2d_3x3(output_name, input->name, weight_tensor->name, bias_tensor->name, 0, stride_w, stride_h, pad_h, pad_w, dilation_h, dilation_w, groups);
		return this->output;
	}

	void backward(Tensor<T>* input) override {
		// Ensure input tensor shape matches expected dimensions
		if (input->shape.size() != 4) {
			throw std::invalid_argument("Input tensor must have 4 dimensions: [N, C_in, H_in, W_in]");
		}
		if (input->shape[1] != weight_tensor->shape[1]) {
			throw std::invalid_argument("Input channels (C_in) must match weight tensor's input channels");
		}
		if (input->shape[2] < 3 || input->shape[3] < 3) {
			throw std::invalid_argument("Input height and width must be at least 3 to apply a 3x3 convolution");
		}

		tensorPool->tensor_conv2d_3x3(output_name, input->name, weight_tensor->name, bias_tensor->name, 1, stride_w, stride_h, pad_h, pad_w, dilation_h, dilation_w, groups);
	}
};

template<typename T>
struct EmbeddingTable : public Module<T> {
	std::string name;
	uint32_t vocab_size;
	uint32_t embedding_dim;
	Tensor<T>* embedding_tensor;
	TensorPool<T>* tensorPool;
	EmbeddingTable(TensorPool<T>* pool, const std::string& name, uint32_t vocab_size, uint32_t embedding_dim, uint32_t num_tokens, uint32_t batch_size)
		: name(name), vocab_size(vocab_size), embedding_dim(embedding_dim), tensorPool(pool) {
		embedding_tensor = &pool->createTensor({ vocab_size, embedding_dim }, name + "-embeddings");
		pool->tensor_fill_random(embedding_tensor->name, -0.1f, 0.1f);
		this->output = &pool->createTensor({batch_size, num_tokens, embedding_dim}, name + "-output");
	}

	EmbeddingTable(){}; // empty default ctor
	
	// input tensor contains token indices (B, token_count)
	Tensor<T>* forward(Tensor<T>* input) override {
		tensorPool->tensor_embed_lookup(this->output->name, embedding_tensor->name, input->name, 0);
		return this->output;
	}
	void backward(Tensor<T>* input) override {
		tensorPool->tensor_embed_lookup(this->output->name, embedding_tensor->name, input->name, 1);
	}
};

template<typename T>
struct FlattenTo : public Module<T> {
	TensorPool<T>* tensorPool;
	std::string name;
	std::string outputName;
	std::vector<uint32_t> original_shape;
	std::vector<uint32_t> new_shape;

	FlattenTo(TensorPool<T>* pool, const std::vector<uint32_t>& flatten_to, const std::string& name) : tensorPool(pool), new_shape(flatten_to), name(name){
		outputName = name + "-output";
	}
	FlattenTo(){};

	Tensor<T>* forward(Tensor<T>* input) override {
		original_shape = input->shape;
		input->view(new_shape);
		this->output = input;
		return this->output;
	}
	
	void backward(Tensor<T>* input) override {
		input->view(original_shape);
	}
};

// flatten a tensor to 1d vector
template<typename T>
struct FlattenTo1d : public Module<T> {
	TensorPool<T>* tensorPool;
	std::string name;
	std::string output_name;
	std::vector<uint32_t> original_shape;
	FlattenTo1d(TensorPool<T>* pool, const std::string& name) : tensorPool(pool), name(name) {
		output_name = name + "-output";
	}

	FlattenTo1d(){}; // default empty ctor

	Tensor<T>* forward(Tensor<T>* input) override {
		original_shape = input->shape;
		input->view({ input->shape[0], static_cast<uint32_t>(std::accumulate(input->shape.begin() + 1, input->shape.end(), 1, std::multiplies<uint32_t>())) });
		this->output = input;
		return this->output;
	}

	void backward(Tensor<T>* input) override {
		input->view(original_shape);
	}
};

template<typename T>
struct FlattenTo2d : public Module<T> {
	TensorPool<T>* tensorPool;
	std::string name;
	std::string output_name;
	std::vector<uint32_t> original_shape;
	FlattenTo2d(TensorPool<T>* pool, const std::string& name) : tensorPool(pool), name(name){
		output_name = name + "-output";
	}

	FlattenTo2d(){}; // default empty ctor

	Tensor<T>* forward(Tensor<T>* input) override {
		original_shape = input->shape;
		input->view({ input->shape[0], input->shape[1], static_cast<uint32_t>(std::accumulate(input->shape.begin()+2, input->shape.end(), 1, std::multiplies<uint32_t>())) });
		this->output = input;
		return this->output;
	}
	void backward(Tensor<T>* input) override {
		input->view(original_shape);
	}
};

template<typename T>
struct Sequential : public Module<T> {
	std::vector<std::unique_ptr<Module<T>>> layers;
	TensorPool<T>* tensorPool;
	Tensor<T>* output;
	std::string name;
	std::string output_name;
	Sequential(TensorPool<T>* pool, const std::string& name) : tensorPool(pool), name(name) {
		output_name = name + "-output";
	}

	Sequential(){}; // empty ctor

	void addLayer(std::unique_ptr<Module<T>> layer) {
		if (dynamic_cast<SoftmaxCrossEntropy<T>*>(layer.get())) {
			throw std::runtime_error("SoftmaxCrossEntropy cannot be added as an intermediate layer");
		}

		if(layers.empty()) {
			layer->prev = nullptr;
		}
		else {
			layer->prev = layers.back().get();
			layers.back()->next = layer.get();
		}
		layers.push_back(std::move(layer));
	}
	Tensor<T>* forward(Tensor<T>* input) override {
		for (auto& layer : layers) {
			layer->forward(input);
			input = layer->output;
		}
		this->output = input;
		return this->output;
	}

	std::vector<Tensor<T>*> getTrainableTensors() override {
		std::vector<Tensor<T>*> ts;
		for (auto& m : layers) {
			auto layer_ts = m->getTrainableTensors();
			ts.insert(ts.end(), layer_ts.begin(), layer_ts.end());
		}
		return ts;
	}

	void backward(Tensor<T>* input = nullptr) {
		for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
			auto& layer = *it;
			if(layer->prev) {
				layer->backward(layer->prev->output);
			}
			else {
				// we're at first layer
				layer->backward(input);
			}
		}
	}
};



template<typename T>
struct SDGoptim : public Optimiser<T>{

	struct uniform{
		tensor_impl tensor;
		float lr;
		uint32_t batch_size;
	};

	struct push {
		VkDeviceAddress uniformAddress;
		static uint32_t getPushConstantRange() {
			return sizeof(tensor_push_const);
		}
	};

	gpuTaskNoDesc<T, push, uniform> optimiser_shader;
	std::vector<Tensor<T>*> tensors;
	uint32_t batch_size;
	float lr = 1e-03;

	SDGoptim(Allocator* allocator) : optimiser_shader(readShaderBytecode("compiled_shaders/SDG.comp.spv"), allocator, nullptr){}

	SDGoptim(){}

	void load_tensors(Sequential<T>& sequence){
		auto s = sequence.getTrainableTensors();
		tensors.insert(tensors.end(), s.begin(), s.end());
	}

	void step() override {
		
		uniform u;
		u.batch_size = batch_size;
		u.lr = lr;
		
		push p;
		
		for (uint32_t i = 0; i < tensors.size(); i++){
			uint32_t num_elements = std::accumulate(tensors[i]->shape.begin(), tensors[i]->shape.end(), 1, std::multiplies<uint32_t>());

			uint32_t x = (num_elements + 256u - 1) / 256u;
			uint32_t workgroup[3] = {x, 1, 1};
			//std::cout << "optimising tensor: " << tensors[i]->name << " of shape: " << shapeToString(tensors[i]->shape) <<"\n";

			u.tensor = tensors[i]->getTensorImpl();
			p.uniformAddress = optimiser_shader.uniformBuffer->getBufferAddress();
			optimiser_shader.loadUniform(u, p);
			optimiser_shader.execute(workgroup);
		}
	}
};