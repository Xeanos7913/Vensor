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
	Tensor<T>* operator()(Tensor<T>* input) {
		return forward(input);
	}
	virtual std::vector<Tensor<T>*> getTrainableTensors() {return {};}
	virtual std::vector<Tensor<T>*> getIntermediateTensors() {return {};}
	virtual void serializeTrainableTensors(std::ofstream& filestream){}
	virtual void loadFromSerializedTensor(std::ifstream& filestream){}
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

	std::vector<Tensor<T>*> getIntermediateTensors() override {
		return {this->output};
	}

	void serializeTrainableTensors(std::ofstream& filestream) override {
		weights->save_to_stream(filestream);
		bias->save_to_stream(filestream);
	}

	void loadFromSerializedTensor(std::ifstream& filestream) override {
		weights->load_from_stream(filestream);
		bias->load_from_stream(filestream);
	}
	
	Tensor<T>* forward(Tensor<T>* input) override {
		// insert extra dim to satisfy kernel API
		auto o = input->shape;
		if(input->shape.size() == 2){
			input->view({input->shape[0], 1, input->shape[1]});
		}
		tensorPool->tensor_linear_ReLU(output_name, input->name, weights_name, bias_name, 0);
		input->view(o);
		this->output->back.push_back([this, input](){
			this->tensorPool->tensor_linear_ReLU(this->output_name, input->name, this->weights_name, this->bias_name, 1);
			input->backward();
		});
		return this->output;
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
		bias = &tensorPool->createTensor({1, 1, out_features}, bias_name);
		this->output = &tensorPool->createTensor(output_shape, output_name);
		
		// He initialization for weights (assuming ReLU follows)
		uint32_t fan_in = in_features;
		tensorPool->tensor_fill_random(weights_name, 3, fan_in, 0, 0.0f, 0.0f); // He init
		tensorPool->tensor_fill_random(bias_name, 4, 0, 0, 0.0f, 0.0f); // Zero bias
	}

	Linear(TensorPool<T>* pool, const std::vector<uint32_t>& weight_dims, const std::vector<uint32_t>& output_dims, const std::string& name) : tensorPool(pool), name(name) {
		weights_name = name + "-weights";
		bias_name = name + "-bias";
		output_name = name + "-output";
		
		weights = &tensorPool->createTensor(weight_dims, weights_name);
		bias = &tensorPool->createTensor({1, output_dims[1], output_dims[2]}, bias_name);
		this->output = &tensorPool->createTensor(output_dims, output_name);
		
		// He initialization
		uint32_t fan_in = weight_dims[2]; // in_features
		tensorPool->tensor_fill_random(weights_name, 3, fan_in, 0, 0.0f, 0.0f);
		tensorPool->tensor_fill_random(bias_name, 4, 0, 0, 0.0f, 0.0f);
	}

	Linear(){}; // default empty ctor

	std::vector<Tensor<T>*> getTrainableTensors() override {
		return {weights, bias};
	}

	void serializeTrainableTensors(std::ofstream& filestream) override {
		weights->save_to_stream(filestream);
		bias->save_to_stream(filestream);
	}

	void loadFromSerializedTensor(std::ifstream& filestream) override {
		weights->load_from_stream(filestream);
		bias->load_from_stream(filestream);
	}

	Tensor<T>* forward(Tensor<T>* input) override {
		// insert extra dim to satisfy kernel API
		auto o = input->shape;
		if(input->shape.size() == 2){
			input->view({input->shape[0], 1, input->shape[1]});
		}
		tensorPool->tensor_linear(output_name, input->name, weights_name, bias_name, 0);
		input->view(o); // return the tensor's original dims
		this->output->back.push_back([this, input](){
			this->tensorPool->tensor_linear(this->output_name, input->name, this->weights_name, this->bias_name, 1);
			input->backward();
		});
		return this->output;
	}
};

template<typename T>
struct ReLU : public Module<T> {
	//Tensor<T>* output;
	TensorPool<T>* tensorPool;
	std::string name;
	std::string output_name;
	ReLU(TensorPool<T>* pool, const std::string& name) : tensorPool(pool), name(name) {
		output_name = name + "-output";
		// do not create output here; create on first forward call
	}
	ReLU(TensorPool<T>* pool, const std::vector<uint32_t>& dims, const std::string& name) : tensorPool(pool), name(name) {
		output_name = name + "-output";
		// do not create output here; create on first forward call
	}
	ReLU(){}; // default empty ctor
	Tensor<T>* forward(Tensor<T>* input) override {
		// create output on first use if it doesn't exist, matching input shape
		if (this->output == nullptr) {
			this->output = &tensorPool->createTensor(input->shape, output_name);
		}

		// If shapes differ, allow it only when total element counts are equal (views).
		if (input->shape != this->output->shape) {
			auto prod = [](const std::vector<uint32_t>& s) -> uint64_t {
				uint64_t p = 1;
				for (auto v : s) p *= v;
				return p;
			};
			uint64_t in_elems = prod(input->shape);
			uint64_t out_elems = prod(this->output->shape);

			if (in_elems != out_elems) {
				throw std::invalid_argument("Input tensor shape is not compatible with ReLU output");
			}

			// Same number of elements: temporarily view input to match output dims for the kernel call,
			// then restore original shape (as done in other layers).
			auto orig_shape = input->shape;
			input->view(this->output->shape);
			tensorPool->tensor_ReLU(output_name, input->name);
			input->view(orig_shape);

			this->output->back.push_back([this, input](){
				this->tensorPool->tensor_ReLU(this->output_name, input->name, 1);
				input->backward();
			});
			return this->output;
		}

		// Shapes equal: normal path
		tensorPool->tensor_ReLU(output_name, input->name);
		this->output->back.push_back([this, input](){
			this->tensorPool->tensor_ReLU(this->output_name, input->name, 1);
			input->backward();
		});
		return this->output;
	}
};

template<typename T>
struct TanH : public Module<T> {
	TensorPool<T>* tensorPool;
	std::string name;
	std::string output_name;

	TanH(TensorPool<T>* pool, const std::string& name) : tensorPool(pool), name(name) {
		output_name = name + "-output";
		// do not create output here; create on first forward call
	}
	TanH(TensorPool<T>* pool, const std::vector<uint32_t>& dims, const std::string& name) : tensorPool(pool), name(name) {
		output_name = name + "-output";
		// do not create output here; create on first forward call
	}
	TanH(){}; // default empty ctor

	// Forward left empty for user to implement
	Tensor<T>* forward(Tensor<T>* input) override {
		// create output on first use if it doesn't exist, matching input shape
		if (this->output == nullptr) {
			this->output = &tensorPool->createTensor(input->shape, output_name);
		}
		// Ensure input shape is compatible
		if (input->shape != this->output->shape) {
			throw std::invalid_argument("Input tensor shape is not compatible with TanH output");
		}
		
		// Apply tanh activation function
		tensorPool->tensor_tanh(input->name, this->output->name);
		this->output->back.push_back([this, input](){
			this->tensorPool->tensor_tanh(input->name, this->output->name, 1);
			input->backward();
		});

		return this->output;
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
		this->output->back.push_back([this, input](){
			this->tensorPool->tensor_cross_entropy(this->output_name, input->name, this->target->name, this->softmax_output->name, 1);
			input->backward();
		});
		return this->output;
	}
};

template<typename T>
struct MSEloss : public Module<T> {

	TensorPool<T>* tensorPool;
	Tensor<T>* target; 		// shape = [B, ...]
	// Tensor<T>* output; 	// shape = [B] (this is the loss tensor. Calculates loss for each example in the batch tensor)
	std::string name;

	MSEloss(TensorPool<T>* pool, uint32_t batch_size, const std::string& name) : tensorPool(pool), name(name){
		this->output = &tensorPool->createTensor({batch_size}, name + "-loss");
	}

	MSEloss(){};

	// forward already handles gradient computation directly
	Tensor<T>* forward(Tensor<T>* input) override {
		if(target == nullptr) throw std::runtime_error("Target tensor for MSE loss not set!");

		// If shapes differ, allow it only when total element counts are equal (views).
		if (input->shape != target->shape) {
			auto prod = [](const std::vector<uint32_t>& s) -> uint64_t {
				uint64_t p = 1;
				for (auto v : s) p *= v;
				return p;
			};
			uint64_t in_elems = prod(input->shape);
			uint64_t target_elems = prod(target->shape);
			if (in_elems != target_elems) {
				throw std::invalid_argument("Input tensor shape is not compatible with target for MSE loss");
			}

			// Temporarily view input to match target shape for kernel call, then restore original shape.
			auto orig_shape = input->shape;
			input->view(target->shape);
			tensorPool->mse_loss(input->name, target->name, this->output->name);
			input->view(orig_shape);
			this->output->back.push_back([input](){
				input->backward();
			});

			return this->output;
		}

		// Shapes equal: normal path
		tensorPool->mse_loss(input->name, target->name, this->output->name);

		this->output->back.push_back([input](){
			input->backward();
		});

		return this->output;
	}
};

template<typename T>
struct KLDloss : public Module<T> {
	TensorPool<T>* tensorPool;
	Tensor<T>* mu_tensor;
	Tensor<T>* logvar_tensor;
	std::string name;

	KLDloss(TensorPool<T>* pool, uint32_t batch_size, const std::string& name) : tensorPool(pool), name(name) {
		this->output = &tensorPool->createTensor({batch_size}, name + "-loss");
	}

	KLDloss() {} // default empty ctor

	// forward already handles gradient computation directly
	Tensor<T>* forward(Tensor<T>* input) override {
		if(logvar_tensor == nullptr || mu_tensor == nullptr) throw std::runtime_error("logvar and mu tensors need to be set for KLDloss to work!");
		tensorPool->kld_loss(mu_tensor->name, logvar_tensor->name, this->output->name);
		this->output->back.push_back([this](){
			mu_tensor->backward();
			logvar_tensor->backward();
		});
		return this->output;
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
	uint32_t mode;                // 0 = train, 1 = eval
	uint32_t batch_size;
	uint32_t accumulate_grad;     // 0: overwrite, 1: += for grads
	float momentum;         // momentum for running stats
	float eps;              // epsilon for numerical stability
	TensorPool<T>* tensorPool;
	std::string name;
	std::string output_name;
	BatchNorm1d(TensorPool<T>* pool, uint32_t features, uint32_t channels, uint32_t batch_size, const std::string& name, uint32_t mode = 0) : tensorPool(pool), name(name), mode(mode) {
		output_name = name + "-output";
		std::vector<uint32_t> shape = { batch_size, channels, features };
		this->output = &tensorPool->createTensor(shape, output_name);
		
		weight_tensor = &tensorPool->createTensor({ channels, features }, name + "-weight");
		bias_tensor = &tensorPool->createTensor({ channels, features }, name + "-bias");
		running_mean = &tensorPool->createTensor({ channels, features }, name + "-running_mean");
		running_var = &tensorPool->createTensor({ channels, features }, name + "-running_var");

		// no need for save_mean or save_var for eval mode
		if(mode == 0){
			save_mean = &tensorPool->createTensor({ channels, features }, name + "-save_mean");
			save_var = &tensorPool->createTensor({ channels, features }, name + "-save_var");
		}
		
		tensorPool->tensor_fill_random(weight_tensor->name, 4, 0.0f, 0.0f, 1.0f, 1.0f);
		// keep bias at zero at initialization
	}

	BatchNorm1d(TensorPool<T>* pool, const std::vector<uint32_t>& shape, const std::string& name, uint32_t mode = 0) : tensorPool(pool), name(name), mode(mode) {
		output_name = name + "-output";
		this->output = &tensorPool->createTensor(shape, output_name);
		
		auto channels = shape[1];
		auto features = shape[2];

		weight_tensor = &tensorPool->createTensor({ channels, features }, name + "-weight");
		bias_tensor = &tensorPool->createTensor({ channels, features }, name + "-bias");
		running_mean = &tensorPool->createTensor({ channels, features }, name + "-running_mean");
		running_var = &tensorPool->createTensor({ channels, features }, name + "-running_var");

		// no need for save_mean or save_var for eval mode
		if(mode == 0){
			save_mean = &tensorPool->createTensor({ channels, features }, name + "-save_mean");
			save_var = &tensorPool->createTensor({ channels, features }, name + "-save_var");
		}

		tensorPool->tensor_fill_random(weight_tensor->name, 1.0f, 1.0f);
		// keep bias at zero at initialization
	}

	BatchNorm1d(){}; // default empty ctor

	std::vector<Tensor<T>*> getTrainableTensors() override {
		return {weight_tensor, bias_tensor};
	}

	void serializeTrainableTensors(std::ofstream& filestream) override {
		weight_tensor->save_to_stream(filestream);
		bias_tensor->save_to_stream(filestream);
		running_mean->save_to_stream(filestream);
		running_var->save_to_stream(filestream);
	}

	void loadFromSerializedTensor(std::ifstream& filestream) override {
		weight_tensor->load_from_stream(filestream);
		bias_tensor->load_from_stream(filestream);
		running_mean->load_from_stream(filestream);
		running_var->load_from_stream(filestream);
	}

	Tensor<T>* forward(Tensor<T>* input) override {
		if (mode == 0){
			tensorPool->tensor_batchnorm_1d(input->name, weight_tensor->name, bias_tensor->name, running_mean->name, running_var->name, this->output->name, save_mean->name, save_var->name, 0);
			this->output->back.push_back([this, input](){
				this->tensorPool->tensor_batchnorm_1d(input->name, this->weight_tensor->name, this->bias_tensor->name, this->running_mean->name, this->running_var->name, this->output_name, this->save_mean->name, this->save_var->name, 1);
				input->backward();
			});
		}else {
			tensorPool->tensor_batchnorm_1d(input->name, weight_tensor->name, bias_tensor->name, running_mean->name, running_var->name, this->output->name, "", "", 0);
		}
		return this->output;
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
	uint32_t mode;                
	uint32_t batch_size;
	uint32_t accumulate_grad;     // 0: overwrite, 1: += for grads
	float momentum;         // momentum for running stats
	float eps;              // epsilon for numerical stability
	TensorPool<T>* tensorPool;
	std::string name;
	std::string output_name;

	BatchNorm2d(TensorPool<T>* tensorPool, uint32_t channels, uint32_t width, uint32_t height, uint32_t batch_size, const std::string& name, uint32_t mode = 0): tensorPool(tensorPool), name(name), mode(mode) {
		output_name = name + "-output";
		std::vector<uint32_t> shape = { batch_size, channels, height, width };
		this->output = &tensorPool->createTensor(shape, output_name);
		
		weight_tensor = &tensorPool->createTensor({ channels }, name + "-weight");
		bias_tensor = &tensorPool->createTensor({ channels }, name + "-bias");
		running_mean = &tensorPool->createTensor({ channels }, name + "-running_mean");
		running_var = &tensorPool->createTensor({ channels }, name + "-running_var");

		// no need for save_mean or save_var for eval mode
		if(mode == 0){
			save_mean = &tensorPool->createTensor({ channels }, name + "-save_mean");
			save_var = &tensorPool->createTensor({ channels }, name + "-save_var");
		}

		tensorPool->tensor_fill_random(weight_tensor->name, 4, 0.0f, 0.0f, 1.0f, 0.0f);
	}

	BatchNorm2d(TensorPool<T>* pool, const std::vector<uint32_t>& shape, const std::string& name, uint32_t mode = 0) : tensorPool(pool), name(name), mode(mode) {
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

		// no need for save_mean or save_var for eval mode
		if(mode == 0){
			save_mean = &tensorPool->createTensor({ channels }, name + "-save_mean");
			save_var = &tensorPool->createTensor({ channels }, name + "-save_var");
		}

		tensorPool->tensor_fill_random(weight_tensor->name, 1.0f, 1.0f);
		// keep bias at zero at initialization
	}

	BatchNorm2d(){}; // default empty ctor

	std::vector<Tensor<T>*> getTrainableTensors() override {
		return {weight_tensor, bias_tensor};
	}

	void serializeTrainableTensors(std::ofstream& filestream) override {
		weight_tensor->save_to_stream(filestream);
		bias_tensor->save_to_stream(filestream);
		running_mean->save_to_stream(filestream);
		running_var->save_to_stream(filestream);
	}

	void loadFromSerializedTensor(std::ifstream& filestream) override {
		weight_tensor->load_from_stream(filestream);
		bias_tensor->load_from_stream(filestream);
		running_mean->load_from_stream(filestream);
		running_var->load_from_stream(filestream);
	}

	Tensor<T>* forward(Tensor<T>* input) override {
		if(mode == 0){
			tensorPool->tensor_batchnorm_2d(input->name, weight_tensor->name, bias_tensor->name, running_mean->name, running_var->name, this->output->name, save_mean->name, save_var->name, 0);
			this->output->back.push_back([this, input](){
				this->tensorPool->tensor_batchnorm_2d(input->name, this->weight_tensor->name, this->bias_tensor->name, this->running_mean->name, this->running_var->name, this->output_name, this->save_mean->name, this->save_var->name, 1);
				input->backward();
			});
		} else {
			tensorPool->tensor_batchnorm_2d(input->name, weight_tensor->name, bias_tensor->name, running_mean->name, running_var->name, this->output->name, "", "", 0);
		}
		return this->output;
	}
};

template<typename T>
struct ResidualConnect : public Module<T>{

	// input_b = input_b + input_a
	Tensor<T>* input_a;		// [any dims]
	Tensor<T>* input_b;		// [same dim as input_a]

	TensorPool<T>* tensorPool;
	std::string name;

	// input_module is the module whose output will be used to add to the output of the layer that comes before this ResidualConnect layer.
	ResidualConnect(TensorPool<T>* tensorPool, Module<T>& input_module, const std::string& name) : tensorPool(tensorPool), name(name) {
		input_a = input_module.output;
		if(this->prev == nullptr) std::runtime_error("a residual connection needs to have a layer above it!");
		input_b = this->prev->output;
	}

	Tensor<T>* forward(Tensor<T>* input) override {
		return input_a->operator+(input_b);
	}
};

template<typename T>
struct Layernorm : public Module<T> {
	//Tensor<T>* output;
    Tensor<T>* weight_tensor;   // [N] - normalized shape
    Tensor<T>* bias_tensor;     // [N] - normalized shape
    Tensor<T>* save_mean;       // [B, M] - mean for each sample
    Tensor<T>* save_rstd;       // [B, M] - reciprocal std for each sample
	TensorPool<T>* tensorPool;
	std::string name;
	std::string output_name;
	uint32_t mode; // 0 = train, 1 = eval

	Layernorm(TensorPool<T>* pool, const std::vector<uint32_t>& shape, const std::vector<uint32_t>& norm_over, const std::string& name, uint32_t mode = 0) : tensorPool(pool), name(name), mode(mode) {
		output_name = name + "-output";
		this->output = &tensorPool->createTensor(shape, output_name);

		// weight/bias correspond to the normalized-over dims (the tail of `shape`)
		weight_tensor = &tensorPool->createTensor(norm_over, name + "-weight");
		bias_tensor = &tensorPool->createTensor(norm_over, name + "-bias");

		// save_mean and save_rstd shapes are the prefix of `shape` up to (but not including) the normalized dims.
		// e.g. if shape = {B, D1, D2, ..., Dn, Dn+1, ...} and norm_over = {Dn, Dn+1, ...},
		// then save shapes = {B, D1, D2, ..., Dn-1, Dn} -> which is shape[0 .. prefix_len-1]
		if (mode == 0) {
			if (norm_over.size() > shape.size() - 1) {
				throw std::invalid_argument("norm_over has more dimensions than input shape (excluding batch)");
			}
			size_t prefix_len = shape.size() - norm_over.size();
			std::vector<uint32_t> save_shape;
			save_shape.reserve(prefix_len);
			for (size_t i = 0; i < prefix_len; ++i) save_shape.push_back(shape[i]);
			save_mean = &tensorPool->createTensor(save_shape, name + "-save_mean");
			save_rstd = &tensorPool->createTensor(save_shape, name + "-save_rstd");
		}

		tensorPool->tensor_fill_random(weight_tensor->name, 4, 0.0f, 0.0f ,1.0f, 0.0f);
		// keep bias at zero at initialization
	}

	Layernorm(){}; // default empty ctor

	std::vector<Tensor<T>*> getTrainableTensors() override {
		return {weight_tensor, bias_tensor};
	}

	void serializeTrainableTensors(std::ofstream& filestream) override {
		weight_tensor->save_to_stream(filestream);
		bias_tensor->save_to_stream(filestream);
	}

	void loadFromSerializedTensor(std::ifstream& filestream) override {
		weight_tensor->load_from_stream(filestream);
		bias_tensor->load_from_stream(filestream);
	}

	Tensor<T>* forward(Tensor<T>* input) override {
		if(mode == 0){
			tensorPool->tensor_layernorm(input->name, weight_tensor->name, bias_tensor->name, this->output->name, save_mean->name, save_rstd->name, 0);
			this->output->back.push_back([this, input](){
				this->tensorPool->tensor_layernorm(input->name, this->weight_tensor->name, this->bias_tensor->name, this->output_name, this->save_mean->name, this->save_rstd->name, 1);
				input->backward();
			});
		}
		else {
			tensorPool->tensor_layernorm(input->name, weight_tensor->name, bias_tensor->name, this->output->name, "", "", 0);
		}
		return this->output;
	}
};

template<typename T>
struct FlashAttention : public Module<T> {

	std::string& name;
	TensorPool<T>* tensorPool;
	Tensor<T>* W_qkv;
	Tensor<T>* Out;
	Tensor<T>* tmp_qkv;
	std::unique_ptr<StandaloneBuffer<T>> delta;
	std::unique_ptr<StandaloneBuffer<T>> L;
	std::unique_ptr<StandaloneBuffer<T>> M;

	uint32_t d_model, n_heads, seq_len, in_features, batch_size;

	FlashAttention(TensorPool<T>* pool, uint32_t d_model, uint32_t in_features, uint32_t n_heads, uint32_t seq_len, uint32_t batch_size, const std::string& name) :
		tensorPool(pool), d_model(d_model), n_heads(n_heads), seq_len(seq_len), batch_size(batch_size), name(name), in_features(in_features) {

		W_qkv = &pool->createTensor({in_features, 3 * n_heads * d_model}, name + "W_qkv");
		tmp_qkv = &pool->createTensor({batch_size, seq_len, 3 * n_heads * d_model}, name + "-tmp_qkv");
		Out = &pool->createTensor({batch_size, n_heads, seq_len, d_model / n_heads}, name + "-output");

		L = std::make_unique<StandaloneBuffer<T>>(batch_size * n_heads * seq_len, &pool->allocator);
		M = std::make_unique<StandaloneBuffer<T>>(batch_size * n_heads * seq_len, &pool->allocator);
		delta = std::make_unique<StandaloneBuffer<T>>(528 ,&pool->allocator); // need to find exact allocation size later
	}

	Tensor<T>* forward(Tensor<T>* input) override {
		
		tensorPool->tensor_flash_attention(input->name, W_qkv->name, tmp_qkv->name, Out->name, L->getBufferAddress(), M->getBufferAddress(), d_model, seq_len, n_heads);

		Out->back.push_back([this, input](){
			this->tensorPool->tensor_flash_attention_bwd(tmp_qkv->name, Out->name, M->getDeviceAddress(), delta->getDeviceAddress(), d_model, seq_len, n_heads);
			input->backward();
		});

		return Out;
	}
};

template<typename T>
struct Conv2d : public Module<T> {
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
	uint32_t kernel_h, kernel_w;

	uint32_t output_height;
	uint32_t output_width;

	Conv2d(TensorPool<T>* pool, uint32_t in_channels, uint32_t out_channels, uint32_t batch_size, uint32_t height, uint32_t width, uint32_t kernel_h, uint32_t kernel_w, const std::string& name,
    uint32_t stride_w = 1, uint32_t stride_h = 1, uint32_t pad_h = 1, uint32_t pad_w = 1, uint32_t dilation_h = 1, uint32_t dilation_w = 1) 
    : tensorPool(pool), name(name), kernel_h(kernel_h), kernel_w(kernel_w) {
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
		if (kernel_w > 10 || kernel_h > 10) {
			throw std::invalid_argument("Kernel sizes above 10 are not supported!");
		}
		
		// Calculate effective kernel size with dilation
		uint32_t effective_kernel_h = kernel_h + (kernel_h - 1) * (dilation_h - 1);
		uint32_t effective_kernel_w = kernel_w + (kernel_w - 1) * (dilation_w - 1);
		
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
		
		output_name = name + "-output";
		this->output = &tensorPool->createTensor({ batch_size, out_channels, this->output_height, this->output_width }, output_name);
		weight_tensor = &tensorPool->createTensor({ 
			out_channels, 
			in_channels, 
			kernel_h, 
			kernel_w 
		}, name + "-weight");
		bias_tensor = &tensorPool->createTensor({ out_channels }, name + "-bias");
		
		// He initialization for Conv2d (assuming ReLU follows)
		uint32_t fan_in = in_channels * kernel_h * kernel_w;
		tensorPool->tensor_fill_random(weight_tensor->name, 3, fan_in, 0, 0.0f, 0.0f);
		tensorPool->tensor_fill_random(bias_tensor->name, 4, 0, 0, 0.0f, 0.0f);
	}

	Conv2d(TensorPool<T>* pool, const std::vector<uint32_t>& weight_dims, const std::vector<uint32_t>& output_dims, const std::string& name) 
		: tensorPool(pool), name(name) {
		if (weight_dims[2] != 3 || weight_dims[3] != 3) {
			throw std::invalid_argument("Weight dimensions must be [out_channels, in_channels, 3, 3]");
		}
		
		output_name = name + "-output";
		this->output = &tensorPool->createTensor(output_dims, output_name);
		weight_tensor = &tensorPool->createTensor(weight_dims, name + "-weight");
		bias_tensor = &tensorPool->createTensor({ weight_dims[0] }, name + "-bias");
		
		// He initialization
		uint32_t fan_in = weight_dims[1] * weight_dims[2] * weight_dims[3]; // in_channels * kernel_h * kernel_w
		tensorPool->tensor_fill_random(weight_tensor->name, 3, fan_in, 0, 0.0f, 0.0f);
		tensorPool->tensor_fill_random(bias_tensor->name, 4, 0, 0, 0.0f, 0.0f);
	}
	Conv2d(){}; // default empty ctor

	std::vector<Tensor<T>*> getTrainableTensors() override {
		return {weight_tensor, bias_tensor};
	}

	void serializeTrainableTensors(std::ofstream& filestream) override {
		weight_tensor->save_to_stream(filestream);
		bias_tensor->save_to_stream(filestream);
	}

	void loadFromSerializedTensor(std::ifstream& filestream) override {
		weight_tensor->load_from_stream(filestream);
		bias_tensor->load_from_stream(filestream);
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

		tensorPool->tensor_conv2d(this->output->name, input->name, weight_tensor->name, bias_tensor->name, kernel_h, kernel_w, 0, stride_w, stride_h, pad_h, pad_w, dilation_h, dilation_w);
		this->output->back.push_back([this, input](){
			this->tensorPool->tensor_conv2d(this->output->name, input->name, weight_tensor->name, bias_tensor->name, kernel_h, kernel_w, 1, stride_w, stride_h, pad_h, pad_w, dilation_h, dilation_w);
			input->backward();
		});
		
		return this->output;
	}
};

// Uses Bilinear Interpolation to upsample input to output
template<typename T>
struct Upsample : public Module<T>{
	TensorPool<float>* tensorPool;
	uint32_t height_out, height_in;
	uint32_t width_out, width_in;
	std::string name;
	Upsample(TensorPool<float>* tensorPool, uint32_t height_in, uint32_t width_in, uint32_t height_out, uint32_t width_out, const std::string& name) 
	: tensorPool(tensorPool), height_out(height_out), width_out(width_out), height_in(height_in), width_in(width_in), name(name) {}
	Upsample(){}

	Tensor<T>* forward(Tensor<T>* input) override {
		if(this->output == nullptr){
			uint32_t batch_size = input->shape[0];
			const uint32_t total_elements = input->get_num_elements();
			uint32_t required_elements_per_batch = height_in * width_in;
			if(total_elements % batch_size != 0){
				throw std::invalid_argument("Total elements must be divisible by batch size");
			}
			
			uint32_t elements_per_batch = total_elements / batch_size;
			if(elements_per_batch % required_elements_per_batch != 0){
				throw std::invalid_argument("Elements per batch incompatible with height_in * width_in");
			}
			
			uint32_t channels = elements_per_batch / required_elements_per_batch;
			this->output = &tensorPool->createTensor({batch_size, channels, height_out, width_out}, name + "-output");
		}
		tensorPool->tensor_upsample(input->name, this->output->name, height_in, width_in, height_out, width_out);
		this->output->back.push_back([this, input](){
			this->tensorPool->tensor_upsample(input->name, this->output->name, height_in, width_in, height_out, width_out, 1);
			input->backward();
		});

		return this->output;
	}
};

template<typename T>
struct TransposedConv2d : public Module<T> {
    Tensor<T>* weight_tensor; // [C_in, C_out, K_h, K_w]
    Tensor<T>* bias_tensor;   // [C_out]
    TensorPool<T>* tensorPool;
    std::string name;
    std::string output_name;

    uint32_t stride_w = 1;
    uint32_t stride_h = 1;
    uint32_t pad_h = 0;
    uint32_t pad_w = 0;
    uint32_t dilation_h = 1;
    uint32_t dilation_w = 1;
    uint32_t output_pad_h = 0;
    uint32_t output_pad_w = 0;
    uint32_t kernel_h, kernel_w;

    uint32_t output_height;
    uint32_t output_width;

    TransposedConv2d(TensorPool<T>* pool,
                 uint32_t in_channels,
                 uint32_t out_channels,
                 uint32_t batch_size,
                 uint32_t height,
                 uint32_t width,
                 uint32_t kernel_h,
                 uint32_t kernel_w,
                 const std::string& name,
                 uint32_t stride_w = 1,
                 uint32_t stride_h = 1,
                 uint32_t pad_h = 1,
                 uint32_t pad_w = 1,
                 uint32_t dilation_h = 1,
                 uint32_t dilation_w = 1,
                 uint32_t output_pad_h = 0,
                 uint32_t output_pad_w = 0)
    : tensorPool(pool), name(name), kernel_h(kernel_h), kernel_w(kernel_w)
	{
		if (stride_h == 0 || stride_w == 0)
			throw std::invalid_argument("Stride must be greater than 0");
		if (kernel_w > 15 || kernel_h > 15)
			throw std::invalid_argument("Kernel sizes above 15 are not supported");
		if (dilation_h > 2 || dilation_w > 2)
			throw std::invalid_argument("Dilation must be 1 or 2");
		
		output_height = (height - 1) * stride_h - 2 * pad_h + dilation_h * (kernel_h - 1) + output_pad_h + 1;
		output_width  = (width  - 1) * stride_w - 2 * pad_w + dilation_w * (kernel_w - 1) + output_pad_w + 1;
		
		if (output_height < 1 || output_width < 1)
			throw std::invalid_argument("Invalid transposed conv configuration producing negative/zero output size");
		
		this->stride_w = stride_w;
		this->stride_h = stride_h;
		this->pad_w = pad_w;
		this->pad_h = pad_h;
		this->dilation_h = dilation_h;
		this->dilation_w = dilation_w;
		this->output_pad_h = output_pad_h;
		this->output_pad_w = output_pad_w;
		
		output_name = name + "-output";
		this->output = &tensorPool->createTensor({ batch_size, out_channels, output_height, output_width }, output_name);
		
		// Note: weight layout [C_in, C_out, K_h, K_w]
		weight_tensor = &tensorPool->createTensor({ in_channels, out_channels, kernel_h, kernel_w }, name + "-weight");
		bias_tensor   = &tensorPool->createTensor({ out_channels }, name + "-bias");
		
		// He initialization for TransposedConv2d
		// fan_in is based on the input channels and kernel size
		uint32_t fan_in = in_channels * kernel_h * kernel_w;
		tensorPool->tensor_fill_random(weight_tensor->name, 3, fan_in, 0, 0.0f, 0.0f);
		tensorPool->tensor_fill_random(bias_tensor->name, 4, 0, 0, 0.0f, 0.0f);
	}

	TransposedConv2d(TensorPool<T>* pool,
					const std::vector<uint32_t>& weight_dims,
					const std::vector<uint32_t>& output_dims,
					const std::string& name)
		: tensorPool(pool), name(name)
	{
		if (weight_dims.size() != 4 || output_dims.size() != 4)
			throw std::invalid_argument("Weight/output dims must both be 4D");
		
		// weight_dims = [C_in, C_out, K_h, K_w]
		kernel_h = weight_dims[2];
		kernel_w = weight_dims[3];
		
		output_name = name + "-output";
		this->output = &tensorPool->createTensor(output_dims, output_name);
		weight_tensor = &tensorPool->createTensor(weight_dims, name + "-weight");
		bias_tensor   = &tensorPool->createTensor({ weight_dims[1] }, name + "-bias");
		
		// He initialization
		uint32_t fan_in = weight_dims[0] * weight_dims[2] * weight_dims[3]; // in_channels * kernel_h * kernel_w
		tensorPool->tensor_fill_random(weight_tensor->name, 3, fan_in, 0, 0.0f, 0.0f);
		tensorPool->tensor_fill_random(bias_tensor->name, 4, 0, 0, 0.0f, 0.0f);
	}

    TransposedConv2d() {}

    std::vector<Tensor<T>*> getTrainableTensors() override {
        return { weight_tensor, bias_tensor };
    }

    void serializeTrainableTensors(std::ofstream& filestream) override {
        weight_tensor->save_to_stream(filestream);
        bias_tensor->save_to_stream(filestream);
    }

    void loadFromSerializedTensor(std::ifstream& filestream) override {
        weight_tensor->load_from_stream(filestream);
        bias_tensor->load_from_stream(filestream);
    }

    // Forward
    Tensor<T>* forward(Tensor<T>* input) override {
        if (input->shape.size() != 4)
            throw std::invalid_argument("Input tensor must have 4 dims [N, C_in, H_in, W_in]");
        if (input->shape[1] != weight_tensor->shape[0])
            throw std::invalid_argument("Input channels must match weight tensor's input channels (C_in)");
        
        tensorPool->tensor_transposed_conv2d(
            output_name,
            input->name,
            weight_tensor->name,
            bias_tensor->name,
            kernel_h, kernel_w,
            0, // mode = forward
            stride_w, stride_h,
            pad_h, pad_w,
            dilation_h, dilation_w,
            output_pad_h, output_pad_w
        );

		this->output->back.push_back([this, input](){
			this->tensorPool->tensor_transposed_conv2d(
				output_name,
				input->name,
				weight_tensor->name,
				bias_tensor->name,
				kernel_h, kernel_w,
				1, // mode = backward
				stride_w, stride_h,
				pad_h, pad_w,
				dilation_h, dilation_w,
				output_pad_h, output_pad_w
        	);
			input->backward();
		});
        return this->output;
    }
};

template<typename T>
struct MaxPool : public Module<T> {

	std::string name;
	uint32_t kernel_h;
	uint32_t kernel_w;
	uint32_t stride_h;
	uint32_t stride_w;
	TensorPool<T>* tensorPool;

	uint32_t height, width, channels;

	MaxPool(TensorPool<T>* pool, uint32_t batch_size, uint32_t height, uint32_t width, uint32_t channels, const std::string& name, uint32_t kernel_h = 4, uint32_t kernel_w = 4, uint32_t stride_h = 1, uint32_t stride_w = 1)
	: tensorPool(pool), kernel_h(kernel_h), kernel_w(kernel_w), stride_h(stride_h), stride_w(stride_w), name(name), height(height), width(width), channels(channels) {

		// basic sanity checks
		if (kernel_h == 0 || kernel_w == 0) throw std::invalid_argument("MaxPool kernel dimensions must be > 0");
		if (stride_h == 0 || stride_w == 0) throw std::invalid_argument("MaxPool stride dimensions must be > 0");
		if (height == 0 || width == 0) throw std::invalid_argument("Input height/width must be > 0");
		if (channels == 0) throw std::invalid_argument("Channels must be > 0");
		if (height < kernel_h) throw std::invalid_argument("Input height is smaller than kernel height");
		if (width  < kernel_w) throw std::invalid_argument("Input width is smaller than kernel width");

		// compute output spatial dimensions using (H - K) / S + 1 (integer division)
		// flag if there would be a partial/window remainder (could indicate user mistake)
		uint32_t diff_h = height - kernel_h;
		uint32_t diff_w = width  - kernel_w;
		if (diff_h % stride_h != 0) {
			throw std::invalid_argument("MaxPool configuration produces a non-integer number of steps in height ( (height - kernel_h) % stride_h != 0 )");
		}
		if (diff_w % stride_w != 0) {
			throw std::invalid_argument("MaxPool configuration produces a non-integer number of steps in width ( (width - kernel_w) % stride_w != 0 )");
		}

		uint32_t out_height = 1 + diff_h / stride_h;
		uint32_t out_width  = 1 + diff_w / stride_w;

		if (out_height == 0 || out_width == 0) throw std::invalid_argument("Computed output spatial dimensions are zero");

		std::string output_name = name + "-output";
		this->output = &tensorPool->createTensor({ batch_size, channels, out_height, out_width }, output_name);
	}

	Tensor<T>* forward(Tensor<T>* input) override {
		tensorPool->tensor_max_pool(input->name, this->output->name, kernel_h, kernel_w, stride_h, stride_w, 0);

		this->output->back.push_back([this,input](){
			this->tensorPool->tensor_max_pool(input->name, this->output->name, kernel_h, kernel_w, stride_h, stride_w, 1);
			input->backward();
		});
		return this->output;
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
	
	void serializeTrainableTensors(std::ofstream& filestream) override {
		embedding_tensor->save_to_stream(filestream);
	}

	void loadFromSerializedTensor(std::ifstream& filestream) override {
		embedding_tensor->load_from_stream(filestream);
	}

	// input tensor contains token indices (B, token_count)
	Tensor<T>* forward(Tensor<T>* input) override {
		tensorPool->tensor_embed_lookup(this->output->name, embedding_tensor->name, input->name, 0);
		
		this->output->back.push_back([this, input](){
			this->tensorPool->tensor_embed_lookup(this->output->name, this->embedding_tensor->name, input->name, 1);
		});

		return this->output;
	}
};

template<typename T>
struct ShapeTo : public Module<T> {
	TensorPool<T>* tensorPool;
	std::string name;
	std::string outputName;
	std::vector<uint32_t> original_shape;
	std::vector<uint32_t> new_shape;

	ShapeTo(TensorPool<T>* pool, const std::vector<uint32_t>& shape_to, const std::string& name) : tensorPool(pool), new_shape(shape_to), name(name){
		outputName = name + "-output";
	}
	ShapeTo(){};

	Tensor<T>* forward(Tensor<T>* input) override {
		auto saved_shape = input->shape;   // COPY

		input->back.push_back([input, saved_shape]() mutable {
			input->view(saved_shape);
		});
		input->view(new_shape);
		
		this->output = input;
		return this->output;
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
		
		auto saved_shape = original_shape;   // COPY

		input->back.push_back([input, saved_shape]() mutable {
			input->view(saved_shape);
		});
		
		this->output = input;
		return this->output;
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

		input->view({
			input->shape[0],
			input->shape[1],
			static_cast<uint32_t>(
				std::accumulate(input->shape.begin()+2, input->shape.end(), 1, std::multiplies<uint32_t>())
			)
		});

		auto saved_shape = original_shape;   // COPY

		input->back.push_back([input, saved_shape]() {
			input->view(saved_shape);
		});

		this->output = input;
		return this->output;
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

	void serializeTrainableTensors(std::ofstream& filestream) override {
		for (auto& layer : layers){
			layer->serializeTrainableTensors(filestream);
		}
	}

	void loadFromSerializedTensor(std::ifstream& filestream) override {
		for (auto& layer : layers){
			layer->loadFromSerializedTensor(filestream);
		}
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

	void load_tensors(Module<T>& module){
		auto s = module.getTrainableTensors();
		for (auto* t : s) this->tensors.push_back(t);
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

template<typename T>
struct AdamW : public Optimiser<T> {

    struct uniform {
        tensor_impl tensor;          // param + grad
        VkDeviceAddress m;           // first moment
        VkDeviceAddress v;           // second moment
        float lr;
        float beta1;
        float beta2;
        float inv_bias1;
        float inv_bias2;
        float lambda;
        float epsilon;
    };

    float lr      = 1e-3f;
    float beta1   = 0.9f;
    float beta2   = 0.999f;
    float lambda  = 1e-2f;
    float epsilon = 1e-8f;

    uint64_t stepCount = 0;
    float beta1_pow = 1.0f;
    float beta2_pow = 1.0f;

    gpuTaskNoDesc<T, tensor_push_const, uniform> shader;

    std::vector<Tensor<T>*> tensors;
    std::vector<std::unique_ptr<StandaloneBuffer<float>>> m;
    std::vector<std::unique_ptr<StandaloneBuffer<float>>> v;

    Allocator* allocator;

	AdamW(){};

    AdamW(Allocator* alloc)
        : allocator(alloc),
          shader(readShaderBytecode("compiled_shaders/AdamW.comp.spv"),
                 alloc, nullptr) {}

    void load_tensors(Module<T>& module) {
        for (auto* t : module.getTrainableTensors()) {
            tensors.push_back(t);

            m.push_back(std::make_unique<StandaloneBuffer<float>>(
                t->get_num_elements(), allocator));

            v.push_back(std::make_unique<StandaloneBuffer<float>>(
                t->get_num_elements(), allocator));
        }
    }

    void step() override {
        stepCount++;

        beta1_pow *= beta1;
        beta2_pow *= beta2;

        const float inv_bias1 = 1.0f / (1.0f - beta1_pow);
        const float inv_bias2 = 1.0f / (1.0f - beta2_pow);

        uniform u;
        u.lr = lr;
        u.beta1 = beta1;
        u.beta2 = beta2;
        u.lambda = lambda;
        u.epsilon = epsilon;
        u.inv_bias1 = inv_bias1;
        u.inv_bias2 = inv_bias2;

        tensor_push_const pc;
        pc.uniformAddress = shader.uniformBuffer->getBufferAddress();

        for (size_t i = 0; i < tensors.size(); ++i) {
            uint32_t n = tensors[i]->get_num_elements();
            uint32_t wg = (n + 255u) / 256u;

            u.tensor = tensors[i]->getTensorImpl();
            u.m = m[i]->getBufferAddress();
            u.v = v[i]->getBufferAddress();
			uint32_t wrkgrp[3] = {wg, 1, 1};

            shader.loadUniform(u, pc);
            shader.execute(wrkgrp);
        }
    }
};