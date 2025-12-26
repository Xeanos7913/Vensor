/*
This main file contains some testing code (in the commented out parts), an example MNIST handwritten digit recognision neural network training loop, 
and a VAE implementation for MNIST handwritten digit data generation.
*/
#define VOLK_IMPLEMENTATION
#include <iostream>
#include <fstream>
//#define DEBUG
#include "../include/Tensor.hpp"
#include "../include/Neural.hpp"
#include "../include/Dataloader.hpp"
#include "../include/progressbar.hpp"
#include <memory>
#include <initializer_list>

using vec = std::vector<uint32_t>;
// Read shader bytecode from a file
//#ifndef readShaderCode
//#define readShaderCode
//std::vector<char> readShaderBytecode(const std::string& filename) {
//    std::ifstream file(filename, std::ios::ate | std::ios::binary);  // Open file at the end in binary mode
//
//    if (!file.is_open()) {
//        throw std::runtime_error("Failed to open shader file: " + filename);
//    }
//
//    size_t fileSize = file.tellg();  // Get file size
//    std::vector<char> buffer(fileSize);
//
//    file.seekg(0);  // Go back to the beginning
//    file.read(buffer.data(), fileSize);  // Read file into buffer
//    file.close();
//    return buffer;
//}    
//#endif // !readShaderCode

//int main(void) {
// auto vec1 = std::vector<int>{ 3, 5, 1 };
//    // inputs for the single GPU task
//    auto vec2 = std::vector<int>{ 4, 3, 5 };
//    auto vec5 = std::vector<int>{ 1, 1, 1 };
//   
//    auto vec3 = std::vector<int>{ 1, 1, 1 };
//    auto vec4 = std::vector<int>{ 10, 10, 10 };
//    
//    auto input = std::vector<std::vector<int>>{ vec1, vec2};
//
//	// inputs for the sequential GPU task
//    std::vector<float> buffer1Input{2.0f, 2.0f, 2.0f};
//    std::vector<float> buffer2Output{1.0f, 1.0f, 1.0f};
//    std::vector<float> buffer3Output{ 1.0f, 1.0f, 1.0f };
//    std::vector<float> buffer4Output{ 1.0f, 1.0f, 1.0f };
//
//    std::vector<uint32_t> inputIndices{ 0 };
//    std::vector<uint32_t> outputIndices{ 1 };
//
//	std::vector<uint32_t> inputIndices2{ 1 };
//	std::vector<uint32_t> outputIndices2{ 2 };
//
//	std::vector<uint32_t> inputIndices3{ 2 };
//	std::vector<uint32_t> outputIndices3{ 3 };
//
//
//    std::vector<std::vector<float>> buffers{buffer1Input, buffer2Output, buffer3Output, buffer4Output};
//
//    auto singleTaskCode = readShaderBytecode("shader.spv");
//
//    // sequential task shader codes
//    auto code = readShaderBytecode("task1.spv");
//    auto code1 = readShaderBytecode("task2.spv");
//    auto code2 = readShaderBytecode("task3.spv");
//
//    Init init;
//
//    device_initialization(init);
//
//	auto allocator = Allocator(&init);
//
//    // single GPU task (numInputs = number of <T> sending in. numOutputs = number of <T> going out in the single output buffer.)
//    gpuTask<int> task = gpuTask<int>(singleTaskCode, init, 100, 3, &allocator);
//    task.load_inputs(input);
//
//    auto result = task.compute();
//    
//    std::cout << "\Single Result: \n\t";
//    for (auto num : result) { std::cout << num << " "; }
//
//	// sequential GPU task
//    auto task1 = SequentialgpuTask<float>(code, init);
//    auto task2 = SequentialgpuTask<float>(code1, init);
//    auto task3 = SequentialgpuTask<float>(code2, init);
//    
//    task1.load_indices(inputIndices, outputIndices);
//    task2.load_indices(inputIndices2, outputIndices2);
//    task3.load_indices(inputIndices3, outputIndices3);
//    
//    auto sequence = ComputeSequence<float>(1, &allocator);
//
//    sequence.addTask(task1);
//    sequence.addTask(task2);
//    sequence.addTask(task3);
//    sequence.addBuffers(buffers);
//    sequence.initializeCompute();
//    sequence.doTasks(1, 3, 1, 1);
//
//    auto resultSeq = sequence.allBuffers[3];
//
//	std::cout << "\nSequential Result: \n\t";
//	for (auto num : resultSeq) { std::cout << num << " "; }
//
//    //finalTensor.print();
//
//    return 0;
//}


// graphics engine test
// graphics programming is no longer fun. too much work for too little reward.

//int main(void) {
//
//	auto engine = Engine(1920, 1080, "compiled_shaders/triangle.vert.spv", "compiled_shaders/triangle.frag.spv");
//
//	auto entity = new Entity(glm::vec3(1.0f), glm::quat(1.0f, 0.0f, 0.0f, 0.0f), glm::vec3(1.0f), true);
//	entity->mesh.loadModel("cube.obj");
//	entity->mesh.material.loadTexture("bricks.jpg");
//	engine.scene.addEntity(*entity);
//	delete entity;
//
//	auto entity2 = new Entity(glm::vec3(0.0f, -1.0f, 0.0f), glm::quat(1.0f, 0.0f, 0.0f, 0.0f), glm::vec3(10.0f, 10.0f, 10.0f), true);
//	entity2->mesh.loadModel("bunny.obj");
//	entity2->mesh.material.loadTexture("bricks.jpg");
//	engine.scene.addEntity(*entity2);
//	delete entity2;
//
//	auto entity3 = new Entity(glm::vec3(-2.0f, 0.0f, 0.0f), glm::quat(1.0f, 0.0f, 0.0f, 0.0f), glm::vec3(20.0f), true);
//	entity3->mesh.loadModel("bunny.obj");
//	entity3->mesh.material.loadTexture("bricks.jpg");
//	engine.scene.addEntity(*entity3);
//	delete entity3;
//
//	auto entity4 = new Entity(glm::vec3(2.0f, 0.0f, 0.0f), glm::quat(1.0f, 0.0f, 0.0f, 0.0f), glm::vec3(20.0f), true);
//	entity4->mesh.loadModel("bunny.obj");
//	entity4->mesh.material.loadTexture("bricks.jpg");
//	engine.scene.addEntity(*entity4);
//	delete entity4;
//
//	engine.run();
//	return 0;
//}

static int off2or3(const std::vector<uint32_t>& strides, int b, int i, int j)
{
	return b * strides[0] + i * strides[1] + j * strides[2];
}


// basic tensor operations test
/*
int main(void) {

	std::cout << "Starting up...\n";
	Init init;
	device_initialization(init);
	Allocator* allocator = new Allocator(&init);
	TensorPool<float> tensorPool(allocator);
	// Create tensors (... batch, row, column)
	std::cout << "Creating tensors...\n";
	auto& A = tensorPool.createTensor({ 5, 1, 256 }, "A"); // input vector of layer 1
	std::cout << "Tensor A shape: ";
	auto& W1 = tensorPool.createTensor({ 1, 64, 256 }, "W1");	// weights of layer 1
	auto& O1 = tensorPool.createTensor({ 5, 1, 64 }, "O1"); // output of layer 1, input of layer 16
	auto& B1 = tensorPool.createTensor({ 1, 1, 64 }, "B1"); // bias of layer 1

	tensorPool.tensor_fill_random("A", -1.0f, 1.0f);
	tensorPool.tensor_fill_random("W1", -1.0f, 1.0f);
	tensorPool.tensor_fill_random("B1", -1.0f, 1.0f);

	// Second layer
	auto& W2 = tensorPool.createTensor({ 1, 16, 64 }, "W2"); // weights of layer 16
	auto& O2 = tensorPool.createTensor({ 5, 1, 16 }, "O2"); // output of layer 16, input of layer 3
	auto& B2 = tensorPool.createTensor({ 1, 1, 16 }, "B2"); // bias of layer 16
	auto& D = tensorPool.createTensor({ 5, 1, 16 }, "D"); // desired output (for loss calculation)

	tensorPool.tensor_fill_random("W2", -1.0f, 1.0f);
	tensorPool.tensor_fill_random("B2", -1.0f, 1.0f);

	auto& W3 = tensorPool.createTensor({ 1, 16, 16 }, "W3"); // weights of layer 3
	auto& B3 = tensorPool.createTensor({ 1, 1, 16 }, "B3"); // bias of layer 3
	auto& O3 = tensorPool.createTensor({ 5, 1, 16 }, "O3"); // output of layer 3

	tensorPool.tensor_fill_random("W3", -1.0f, 1.0f);
	tensorPool.tensor_fill_random("B3", -1.0f, 1.0f);

	auto& S = tensorPool.createTensor({ 5, 1, 16 }, "S"); // softmax output
	auto& L = tensorPool.createTensor({ 5, 1, 1 }, "L"); // loss output

	auto& T = tensorPool.createTensor({ 5, 1, 16 }, "T"); // target tensor (ground truth one-hot)
	T.dataBuffer->set(1, 1.0f); // set class 1 as ground truth for now
	T.dataBuffer->set(off2or3(T.strides, 1, 0, 1), 1.0f); // set class 1 as ground truth for now
	T.dataBuffer->set(off2or3(T.strides, 2, 0, 1), 1.0f); // set class 1 as ground truth for now
	T.dataBuffer->set(off2or3(T.strides, 3, 0, 1), 1.0f); // set class 1 as ground
	T.dataBuffer->set(off2or3(T.strides, 4, 0, 1), 1.0f); // set class 1 as ground
	//T.dataBuffer->set(off2or3(T.strides, 5, 0, 1), 1.0f); // set class 1 as ground
	//T.dataBuffer->set(off2or3(T.strides, 6, 0, 1), 1.0f); // set class 1 as ground
	//T.dataBuffer->set(off2or3(T.strides, 7, 0, 1), 1.0f); // set class 1 as ground
	//T.dataBuffer->set(off2or3(T.strides, 8, 0, 1), 1.0f); // set class 1 as ground
	//T.dataBuffer->set(off2or3(T.strides, 9, 0, 1), 1.0f); // set class 1 as ground
	//T.dataBuffer->set(off2or3(T.strides, 10, 0, 1), 1.0f); // set class 1 as ground
	//T.dataBuffer->set(off2or3(T.strides, 11, 0, 1), 1.0f); // set class 1 as ground
	//T.dataBuffer->set(off2or3(T.strides, 12, 0, 1), 1.0f); // set class 1 as ground
	//T.dataBuffer->set(off2or3(T.strides, 13, 0, 1), 1.0f); // set class 1 as ground
	//T.dataBuffer->set(off2or3(T.strides, 14, 0, 1), 1.0f); // set class 1 as ground
	//T.dataBuffer->set(off2or3(T.strides, 15, 0, 1), 1.0f); // set class 1 as ground

	// forward pass
	tensorPool.tensor_linear_ReLU("O1", "A", "W1", "B1", 0); // first layer forward pass
	tensorPool.tensor_linear_ReLU("O2", "O1", "W2", "B2", 0); // second layer forward pass
	tensorPool.tensor_cross_entropy("L", "O2", "T", "S", 0); // softmax + cross-entropy loss calculation

	// backward pass
	tensorPool.tensor_cross_entropy("L", "O2", "T", "S", 1); // cross-entropy loss backward pass (computes gradient w.r.t. O2)
	tensorPool.tensor_linear_ReLU("O2", "O1", "W2", "B2", 1); // second layer backward pass (computes gradients w.r.t. W2, B2, O1)
	tensorPool.tensor_linear_ReLU("O1", "A", "W1", "B1", 1); // first layer backward pass (computes gradients w.r.t. W1, B1, A)
	
	// gradients are now in W1.gradientBuffer, B1.gradientBuffer, W2.gradientBuffer, B2.gradientBuffer	
	O2.printGradient();
	W2.printGradient();
	B2.printGradient();

	delete allocator;
	return 0;
}
*/

// batchnorm test
/*
int main(void) {
	Init init;
	device_initialization(init);
	Allocator* allocator = new Allocator(&init);
	TensorPool<float> tensorPool(allocator);

	auto &input = tensorPool.createTensor({16, 32, 32}, "input");
	tensorPool.tensor_fill_random("input", -1.0f, 1.0f);
	auto batchnorm = BatchNorm1d<float>(&tensorPool, 32, 32, 16, "batchnorm");
	auto batchnorm2 = BatchNorm1d<float>(&tensorPool, 32, 32, 16, "batchnorm2");

	batchnorm.forward(&input);
	batchnorm2.forward(batchnorm.output);

	batchnorm2.save_mean->print();
	batchnorm2.save_var->print();
	
	delete allocator;
	return 0;
}
*/
/*
int main(void) {
	Init init;
	device_initialization(init);
	Allocator* allocator = new Allocator(&init);
	TensorPool<float> tensorPool(allocator);

	auto& A = tensorPool.createTensor({ 2, 3, 4 }, "A");
	auto& B = tensorPool.createTensor({ 2, 3, 4 }, "B");

	tensorPool.tensor_fill_random("A", -1.0f, 1.0f);
	tensorPool.tensor_fill_random("B", 1.0f, 1.0f);

	A.print();
	B.print();

	bool equal = tensorPool.are_tensors_equal("A", "B");
	std::cout << "Tensors A and B are equal: " << (equal ? "true" : "false") << "\n";

	tensorPool.destroy_tensor("B");
	tensorPool.destroy_tensor("A");

	delete allocator;
	return 0;
}
	*/

/*
int main(void){
	
	Init init;
	device_initialization(init);
	Allocator* allocator = new Allocator(&init);
	TensorPool<float> tensorPool(allocator);

	tensorPool.createTensor("tensor1.tnsr");
	auto &tensor1 = tensorPool.tensors["tensor1"];
	
	tensor1->print();

	delete allocator;
	return 0;
}
*/


// simple neural network test
/*
int main(void){
	Init init;
	device_initialization(init);
	Allocator* allocator = new Allocator(&init);
	TensorPool<float> tensorPool(allocator);

	auto &input = tensorPool.createTensor({ 5, 5 }, "input"); // token index tensor
	input.dataBuffer->alloc(std::vector<float>{ 10, 20, 30, 40, 50, 10, 20, 30, 40, 50, 10, 20, 30, 40, 50, 10, 20, 30, 40, 50, 10, 20, 30, 40, 50 }); // batch of 5 token indices

	auto embedding = std::make_unique<EmbeddingTable<float>>(&tensorPool, "embedding1", 5000, 1024, 5, 5); // outputs (5, 5, 1024)
	auto linear = std::make_unique<Linear<float>>(&tensorPool, vec{1, 256, 1024}, vec{5, 5, 256}, "linear1"); // outputs (5, 5, 256) [(5, 5, 256) = (5, 5, 1024) @ (1, 1024, 256)]
	auto bn = std::make_unique<BatchNorm1d<float>>(&tensorPool, vec{5, 5, 256}, "batchnorm1"); // outputs (5, 5, 256)
	auto relu1 = std::make_unique<ReLU<float>>(&tensorPool, 256, 5, "relu1"); // outputs (5, 5, 256)
	auto linear2 = std::make_unique<Linear<float>>(&tensorPool, vec{1, 256, 16}, vec{5, 5, 16}, "linear2"); // outputs (5, 5, 16) [(5, 5, 16) = (5, 5, 256) @ (1, 256, 16)]
	auto bn2 = std::make_unique<BatchNorm1d<float>>(&tensorPool, vec{5, 5, 16}, "batchnorm2");
	auto relu2 = std::make_unique<ReLU<float>>(&tensorPool, vec{5, 5, 16}, "relu2"); // logits = last row of (5, 5, 16) -> (5, 1, 16)
	auto softmax_ce = std::make_unique<SoftmaxCrossEntropy<float>>(&tensorPool, vec{5, 5, 16}, "softmax_ce");

	// each token in token_count predicts its next token class (from 0 to num_classes-1)
	// during inference, we only care about the last token's prediction
	auto &target = tensorPool.createTensor({ 5, 5, 16 }, "target"); // target tensor of shape (batch, token_count, num_classes)
	// just some random initialization for testing (setting 1 to be ground truth for all batches
	target.dataBuffer->set(1, 1.0f);
	target.dataBuffer->set(off2or3(target.strides, 1, 0, 1), 1.0f);
	target.dataBuffer->set(off2or3(target.strides, 2, 0, 1), 1.0f);
	target.dataBuffer->set(off2or3(target.strides, 3, 0, 1), 1.0f);
	target.dataBuffer->set(off2or3(target.strides, 4, 0, 1), 1.0f);
	softmax_ce->target = &target;

	auto seq = Sequential<float>(&tensorPool, "model1");
	seq.addLayer(std::move(embedding));
	seq.addLayer(std::move(linear));
	seq.addLayer(std::move(bn));
	seq.addLayer(std::move(relu1));
	seq.addLayer(std::move(linear2));
	seq.addLayer(std::move(bn2));
	seq.addLayer(std::move(relu2));
	seq.addLayer(std::move(softmax_ce));

	seq.forward(&input, &target);

	//if(seq.output == nullptr) {
	//	std::cout << "Output tensor is null!\n";
	//} else {
	//	std::cout << "Output tensor is not null!\n";
	//	seq.output->print();
	//}

	//seq.backward(&input);

	delete allocator;
	
	return 0;
}
*/

// Linear test
/*
int main(void){
	Init init;
	device_initialization(init);
	Allocator* allocator = new Allocator(&init);

	{
		TensorPool<float> pool(allocator);

		//auto l = Linear<float>(&pool, 28 * 28, 512, 16, "l");
		auto l2 = Linear<float>(&pool, 512, 28 * 28, 16, "l2");

		auto m = MSEloss<float>(&pool, 1, "m");

		auto &i = pool.createTensor({16, 1, 512}, "i");
		auto &t = pool.createTensor({16, 1, 28 * 28}, "t");
		pool.tensor_fill_random("i", 0, 0.0f, 0.0f, -1.0f, 1.0f);
		pool.tensor_fill_random("t", 0, 0.0f, 0.0f, -1.0f, 1.0f);
		
		m.target = &t;
		auto o = m(l2((&i)));
		o->backward();
		l2.weights->printGradient();
	}

	delete allocator;
	return 0;
}
*/

// Layernorm test
/*
int main(void) {
	Init init;
	device_initialization(init);
	Allocator* allocator = new Allocator(&init);
	TensorPool<float> tensorPool(allocator);

	auto &input = tensorPool.createTensor({5, 1, 512}, "input");
	tensorPool.tensor_fill_random("input", -1.0f, 1.0f);
	auto linear = Linear<float>(&tensorPool, 512, 512, 5, "linear");
	auto layernorm = Layernorm<float>(&tensorPool, {5, 1, 512}, {1, 512}, "layernorm");
	auto linear1 = Linear<float>(&tensorPool, 512, 10, 5, "linear1");
	auto softmax = SoftmaxCrossEntropy<float>(&tensorPool, 10, 1, 5, "softmax");

	auto& target = tensorPool.createTensor({5, 1, 10}, "targets");

	target.setElement(1.0f, {0, 0, 1});
	target.setElement(1.0f, {1, 0, 1});
	target.setElement(1.0f, {2, 0, 1});
	target.setElement(1.0f, {3, 0, 1});
	target.setElement(1.0f, {4, 0, 1});

	softmax.target = &target;

	softmax(linear1((layernorm(linear(&input)))));
	
	softmax.backward(linear1.output);
	linear1.backward(layernorm.output);
	layernorm.backward(linear.output);
	//linear.backward(&input);

	linear.output->printGradient();

	delete allocator;
	return 0;
}
*/

// Batchnorm2d test
/*
int main(void) {
	Init init;
	device_initialization(init);
	Allocator* allocator = new Allocator(&init);
	TensorPool<float> tensorPool(allocator);

	auto &input = tensorPool.createTensor({5, 3, 32, 32}, "input");
	tensorPool.tensor_fill_random("input", 5.0f, 10.0f);
	auto batchnorm = BatchNorm2d<float>(&tensorPool, 3, 32, 32, 5, "batchnorm");
	auto batchnorm2 = BatchNorm2d<float>(&tensorPool, 3, 32, 32, 5, "batchnorm2");

	batchnorm.forward(&input);
	batchnorm2.forward(batchnorm.output);

	batchnorm2.save_mean->print();
	batchnorm2.save_var->print();
	
	delete allocator;
	return 0;
}
*/

// Max Pooling test
/*
int main(void) {
	Init init;
	device_initialization(init);
	Allocator* allocator = new Allocator(&init);
	TensorPool<float> pool = TensorPool<float>(allocator);

	auto &input = pool.createTensor({16, 3, 64, 64}, "input");

	auto maxPool = MaxPool<float>(&pool, 16, 64, 64, 3, "maxPool");

	auto output = maxPool(&input);
	
	output->printShape();
	return 0;
}
*/

// CNN test
/*
int main(void){
	Init init;
	device_initialization(init);
	Allocator* allocator = new Allocator(&init);

	{
		TensorPool<float> tensorPool(allocator);

		auto &inputImage = tensorPool.createTensor({5, 16, 28, 28}, "inputImage");
		tensorPool.tensor_fill_random("inputImage", 0, 0.0f, 0.0f, -1.0f, 1.0f);

		Conv2d<float> conv(&tensorPool, 16, 16, 5, 28, 28, 3, 3, "conv1", 1, 1, 0, 0);
		BatchNorm2d<float> bn(&tensorPool, 16, conv.output_width, conv.output_height, 5, "batchnorm-2d");
		Linear<float> linear(&tensorPool, 16 * conv.output_height * conv.output_width, 16, 5, "linear"); // output is (5, 1, 16)
		BatchNorm1d<float> batchnorm(&tensorPool, 16, 1, 5, "batchnorm");
		SoftmaxCrossEntropy<float> softmax(&tensorPool, 16, 1, 5, "softmax");

		// set all the labels to be class = 1
		auto &target = tensorPool.createTensor({5, 1, 16}, "targets");
		target.dataBuffer->set(1, 1.0f);
		target.dataBuffer->set(off2or3(target.strides, 1, 0, 1), 1.0f);
		target.dataBuffer->set(off2or3(target.strides, 2, 0, 1), 1.0f);
		target.dataBuffer->set(off2or3(target.strides, 3, 0, 1), 1.0f);
		target.dataBuffer->set(off2or3(target.strides, 4, 0, 1), 1.0f);
		softmax.target = &target;

		//conv.forward(&inputImage)->print();
		softmax((linear((conv(&inputImage)))))->backward();
		inputImage.printGradient();
	}

	delete allocator;
	return 0;
}
*/

// Writes a single image (first batch element by default) from a Tensor<float>* to a PNG file using stb_image_write.
// Supports tensors in NCHW (4D) or NHW / N H W (3D) formats. Creates the output folder if it doesn't exist.

static void write_tensor_image_png(Tensor<float>* tensor,
								   const std::string& filename,
								   size_t img_index = 0,
								   const std::string& out_dir = "output_images")
{
	if (!tensor) return;

	// Try to read shape. Tensor is expected to expose a shape vector<uint32_t>.
	std::vector<uint32_t> shape;
	try {
		shape = tensor->shape; // assumes public member named `shape`
	} catch (...) {
		// If shape not present, bail out.
		std::cerr << "Tensor has no accessible shape member\n";
		return;
	}

	size_t N = 1, C = 1, H = 1, W = 1;
	if (shape.size() != 4) {
		std::cerr << "Expected 4D tensor shape (N,C,H,W). Got " << shape.size() << "D.\n";
		return;
	}
	N = static_cast<size_t>(shape[0]);
	C = static_cast<size_t>(shape[1]);
	H = static_cast<size_t>(shape[2]);
	W = static_cast<size_t>(shape[3]);

	if (img_index >= N) {
		std::cerr << "img_index out of range (N = " << N << ")\n";
		return;
	}

	// raw float data (assumes contiguous NCHW layout like loader provides)
	auto d = tensor->dataBuffer->downloadBuffer();
	float* data = d.data();
	if (!data) {
		std::cerr << "Tensor data pointer is null\n";
		return;
	}

	const size_t img_elems = C * H * W;
	const float* src = data + img_index * img_elems;

	// Prepare output buffer (uint8_t)
	std::vector<uint8_t> out_buf(img_elems);
	for (size_t i = 0; i < img_elems; ++i) {
		float v = src[i];
		// assume input is normalized in [0,1], clamp; if in [-1,1] user can preprocess externally
		int iv = static_cast<int>(std::lround(std::clamp(v, 0.0f, 1.0f) * 255.0f));
		out_buf[i] = static_cast<uint8_t>(std::clamp(iv, 0, 255));
	}

	// Ensure output directory exists
	std::filesystem::create_directories(out_dir);

	// Build full path and write. For multi-channel images the tensor is assumed to be in CHW order;
	// stb_image_write expects interleaved RGB(A). For C>1 we convert CHW -> HWC interleaved here.
	std::string out_path = out_dir + "/" + filename;
	if (C == 1) {
		// single-channel PNG
		if (!stbi_write_png(out_path.c_str(), static_cast<int>(W), static_cast<int>(H), 1, out_buf.data(), static_cast<int>(W) * 1)) {
			std::cerr << "Failed to write PNG: " << out_path << "\n";
		}
	} else {
		// convert CHW -> HWC (interleaved)
		std::vector<uint8_t> interleaved(H * W * C);
		for (size_t c = 0; c < C; ++c) {
			size_t plane_offset = c * H * W;
			for (size_t h = 0; h < H; ++h) {
				for (size_t w = 0; w < W; ++w) {
					size_t src_idx = plane_offset + h * W + w;
					size_t dst_idx = (h * W + w) * C + c;
					interleaved[dst_idx] = out_buf[src_idx];
				}
			}
		}
		if (!stbi_write_png(out_path.c_str(), static_cast<int>(W), static_cast<int>(H), static_cast<int>(C), interleaved.data(), static_cast<int>(W) * static_cast<int>(C))) {
			std::cerr << "Failed to write PNG: " << out_path << "\n";
		}
	}

	std::cout << "Wrote image: " << out_path << "\n";
}

// Working vae for the MNIST dataset
struct VAE {

	Allocator* allocator;

	TensorPool<float> tensorPool;

	Sequential<float> enc_conv;
	Sequential<float> dec_conv;
	Linear<float> fc_mu;
	Linear<float> fc_logvar;
	MSEloss<float> mseLoss;
	KLDloss<float> kldLoss;

	SDGoptim<float> optim;

	uint32_t latent_dim = 64;

	std::unique_ptr<MNISTDataloader<float>> dataLoader;

	VAE(Allocator* allocator) : allocator(allocator) {

		tensorPool = TensorPool<float>(allocator);

		dataLoader = std::make_unique<MNISTDataloader<float>>(&tensorPool, 16, 100);

		optim = SDGoptim<float>(allocator);
		optim.batch_size = 16;
		optim.lr = 5e-04;

		// ------------------------------
		// Encoder
		// ------------------------------
		auto conv1 = std::make_unique<Conv2d<float>>(
			&tensorPool, 1, 32, 16,
			28, 28,
			4, 4,
			"conv1",
			2, 2   // stride
		);

		auto relu1 = std::make_unique<ReLU<float>>(&tensorPool, "relu1");

		auto conv2 = std::make_unique<Conv2d<float>>(
			&tensorPool, 32, 64, 16,
			conv1->output_height, conv1->output_width,
			4, 4,
			"conv2",
			2, 2
		);

		auto relu2 = std::make_unique<ReLU<float>>(&tensorPool, "relu2");

		auto enc_fc = std::make_unique<Linear<float>>(
			&tensorPool,
			64 * conv2->output_height * conv2->output_width,
			512,
			16,
			"enc_fc"
		);

		// mu / logvar heads
		fc_mu          = Linear<float>(&tensorPool, 512, latent_dim, 16, "fc_mu");
		fc_logvar      = Linear<float>(&tensorPool, 512, latent_dim, 16, "fc_logvar");

		// ------------------------------
		// Decoder
		// ------------------------------
		auto dec_fc = std::make_unique<Linear<float>>(
			&tensorPool,
			latent_dim,
			64 * conv2->output_height * conv2->output_width,
			16,
			"dec_fc"
		);

		auto relu3 = std::make_unique<ReLU<float>>(&tensorPool, "relu3");

		// ---- First Upsample + Conv ----
		uint32_t up1_h = conv2->output_height * 2;
		uint32_t up1_w = conv2->output_width  * 2;

		auto upsample1 = std::make_unique<Upsample<float>>(
			&tensorPool,
			conv2->output_height, conv2->output_width,
			up1_h,
			up1_w,
			"upsample1"
		);

		auto dec_conv1 = std::make_unique<Conv2d<float>>(
			&tensorPool,
			64, 32,
			16,
			up1_h, up1_w,
			3, 3,
			"dec_conv1",
			1, 1,   // stride
			1, 1    // padding
		);

		auto relu4 = std::make_unique<ReLU<float>>(&tensorPool, "relu4");

		// ---- Second Upsample + Conv ----
		uint32_t up2_h = up1_h * 2;
		uint32_t up2_w = up1_w * 2;

		auto upsample2 = std::make_unique<Upsample<float>>(
			&tensorPool,
			dec_conv1->output_height, dec_conv1->output_width,
			up2_h,
			up2_w,
			"upsample2"
		);

		auto dec_conv2 = std::make_unique<Conv2d<float>>(
			&tensorPool,
			32, 1,
			16,
			up2_h, up2_w,
			3, 3,
			"dec_conv2",
			1, 1,
			1, 1
		);

		// Xavier for final conv
		tensorPool.tensor_fill_random(
			dec_conv2->weight_tensor->name,
			2,                      // Glorot (uniform or normal mode flag)
			32 * 3 * 3,             // fan_in
			1  * 3 * 3,             // fan_out
			0.0f,
			0.0f
		);

		auto tanh = std::make_unique<TanH<float>>(&tensorPool, "tanh");

		// ------------------------------
		// Build Sequential Networks
		// ------------------------------
		enc_conv = Sequential<float>(&tensorPool, "enc");
		enc_conv.addLayer(std::move(conv1));
		enc_conv.addLayer(std::move(relu1));
		enc_conv.addLayer(std::move(conv2));
		enc_conv.addLayer(std::move(relu2));
		enc_conv.addLayer(std::move(enc_fc));

		dec_conv = Sequential<float>(&tensorPool, "dec");
		dec_conv.addLayer(std::move(dec_fc));
		dec_conv.addLayer(std::move(relu3));
		dec_conv.addLayer(std::move(upsample1));
		dec_conv.addLayer(std::move(dec_conv1));
		dec_conv.addLayer(std::move(upsample2));
		dec_conv.addLayer(std::move(dec_conv2));
		dec_conv.addLayer(std::move(tanh));

		mseLoss = MSEloss<float>(&tensorPool, 16, "mse");
		kldLoss = KLDloss<float>(&tensorPool, 16, "kld");

		optim.load_tensors(enc_conv);
		optim.load_tensors(fc_logvar);
		optim.load_tensors(fc_mu);
		optim.load_tensors(dec_conv);
	}

	std::pair<Tensor<float>*, Tensor<float>*> encode(Tensor<float>* input) {
		
		auto h = enc_conv(input);
		auto mu = fc_mu(h);
		auto logvar = fc_logvar(h);

		return {mu, logvar};
	}

	Tensor<float>* reparameterize(Tensor<float>* mu, Tensor<float>* logvar){
		auto &std_tensor = (0.5f * *logvar).exp();
		auto &eps_tensor = tensorPool.createTensor({16, 1, latent_dim}, "eps");
		
		// Use Gaussian/Normal distribution N(0, 1) for VAE reparameterization
		// init_type=1: Normal distribution with mean=0.0, stddev=1.0
		tensorPool.tensor_fill_random("eps", 1, 0, 0, 0.0f, 1.0f);
		
		return &(*mu + std_tensor * eps_tensor);
	}

	Tensor<float>* decode(Tensor<float>* z){
		return dec_conv(z);
	}

	Tensor<float>* forward(Tensor<float>* input){
		auto [mu, logvar] = encode(input);
		auto z = reparameterize(mu, logvar);
		return decode(z);
	}

	Tensor<float>* loss_function(Tensor<float>* recon, Tensor<float>* x, Tensor<float>* mu, Tensor<float>* logvar) {
		mseLoss.target = x;
		auto &ml = *mseLoss(recon);
		kldLoss.logvar_tensor = logvar;
		kldLoss.mu_tensor = mu;
		auto &kld = *kldLoss(x);

		return &(ml + kld);
	}

	void train(int epoch){
		using clock = std::chrono::high_resolution_clock;
		using ms    = std::chrono::duration<double, std::milli>;

		auto epoch_start = clock::now();

		dataLoader->loadMNIST("dataset/train.idx3-ubyte", "dataset/labels.idx1-ubyte");

		Tensor<float>* loss  = nullptr;
		Tensor<float>* recon = nullptr;

		double fwd_ms = 0.0;
		double bwd_ms = 0.0;
		double opt_ms = 0.0;

		progressbar bar(100, true, std::cout);

		for(int i = 0; i < dataLoader->num_batches; i++){
			auto* input = dataLoader->imagesBatchTensors[i];

			// ---- forward pass ----
			auto t0 = clock::now();
			auto [mu, logvar] = encode(input);
			auto z = reparameterize(mu, logvar);
			recon = decode(z);
			loss  = loss_function(recon, input, mu, logvar);
			auto t1 = clock::now();
			fwd_ms += ms(t1 - t0).count();

			// ---- backward pass ----
			auto t2 = clock::now();
			loss->backward();
			auto t3 = clock::now();
			bwd_ms += ms(t3 - t2).count();

			// ---- optimizer step ----
			auto t4 = clock::now();
			optim.step();
			tensorPool.zero_out_all_grads();
			auto t5 = clock::now();
			opt_ms += ms(t5 - t4).count();

			bar.update();
		}

		int n = dataLoader->num_batches;

		auto epoch_end = clock::now();
		double epoch_ms = ms(epoch_end - epoch_start).count();

		double avg_fwd = fwd_ms / n;
		double avg_bwd = bwd_ms / n;
		double avg_opt = opt_ms / n;

		auto l = tensorPool.find_mean_of_tensor(loss->name);

		std::cout << "epoch loss: " << l << "\n";
		std::cout << "epoch time: " << epoch_ms << " ms\n";
		std::cout << "avg forward:  " << avg_fwd << " ms/batch\n";
		std::cout << "avg backward: " << avg_bwd << " ms/batch\n";
		std::cout << "avg optim:    " << avg_opt << " ms/batch\n";

		auto s = recon->shape;
		recon->view({16, 1, 28, 28});
		write_tensor_image_png(recon, "recon_" + std::to_string(epoch) + ".png");
		recon->view(s);
	}
};

/*
int main(void){
	
	Init init;
	device_initialization(init);
	Allocator* allocator = new Allocator(&init);

	{
		VAE vae = VAE(allocator);
		//vae.dataLoader->loadMNIST("dataset/train.idx3-ubyte", "dataset/labels.idx1-ubyte");
		for(int i = 0; i < 100; i++){
			vae.train(i);
		}
	}
		
	delete allocator;
	return 0;
}
*/

/*
int main(void){
	Init init;
	device_initialization(init);

	// Allocator must be heap-allocated and must outlive all library resources that use it.
	Allocator* allocator = new Allocator(&init);

	{
		// All resources that depend on allocator live in this scope.
		TensorPool<float> tensorPool(allocator);

		MNISTDataloader<float> data(&tensorPool, 1, 1);
		data.loadMNIST("dataset/train.idx3-ubyte", "dataset/labels.idx1-ubyte");

		MSEloss<float> mse(&tensorPool, 1, "mse");

		// imagesBatchTensors holds Tensor<float>* entries
		Tensor<float>* in = data.imagesBatchTensors[0];
		Tensor<float>* tar = data.imagesBatchTensors[0];

		mse.target = tar;
		mse.forward(in);

		mse.output->print();

		// leaving this scope will destroy tensorPool, data, mse, and any other objects that used allocator
	}

	// Now it's safe to delete the allocator after all dependent resources have been destroyed.
	delete allocator;

	return 0;
}
*/

// A handwritten digit recognision neural network
struct Trainer {
	
	// Vulkan stuff:
	Allocator* allocator; // need to use heap-allocated allocator for it to work. I don't actually know why this is the case.

	// Data prep:
	TensorPool<float> tensorPool;
	std::unique_ptr<MNISTDataloader<float>> dataLoader;
	std::unique_ptr<MNISTDataloader<float>> testDataLoader;

	// contains the model
	Sequential<float> sequence;
	std::unique_ptr<SoftmaxCrossEntropy<float>> softmax;
	SDGoptim<float> optim;

	// Ctor:
	Trainer(Allocator* allocator) : allocator(allocator){
		tensorPool = TensorPool<float>(allocator);

		dataLoader = std::make_unique<MNISTDataloader<float>>(&tensorPool, 16, 100);
		testDataLoader = std::make_unique<MNISTDataloader<float>>(&tensorPool, 1, 100, "mnist_image_batch_test_");

		sequence = Sequential<float>(&tensorPool, "digit-recognision");

		optim = SDGoptim<float>(allocator);
		optim.batch_size = 16;
		optim.lr = 1e-02;

		// build the model:
		// build conv1, grab its output dims, move into sequence
		// make sure "mode" is set to 0 for training and 1 for eval in the norm layers
		{
			auto c1 = std::make_unique<Conv2d<float>>(&tensorPool, 1, 5, 16, 28, 28, 3, 3, "conv1", 1, 1, 0, 0);
			auto out_w1 = c1->output_width;
			auto out_h1 = c1->output_height;
			sequence.addLayer(std::move(c1));
			// batchnorm1 depends on conv1 output dims
			sequence.addLayer(std::make_unique<BatchNorm2d<float>>(&tensorPool, 5, out_w1, out_h1, 16, "bn1"));
			// build conv2 using conv1's output dims as its input dims, grab conv2 dims, move into sequence
			auto c2 = std::make_unique<Conv2d<float>>(&tensorPool, 5, 5, 16, out_h1, out_w1, 3, 3, "conv2", 1, 1, 0, 0);
			auto out_w2 = c2->output_width;
			auto out_h2 = c2->output_height;
			sequence.addLayer(std::move(c2));
			// batchnorm2 depends on conv2 output dims
			sequence.addLayer(std::make_unique<BatchNorm2d<float>>(&tensorPool, 5, out_w2, out_h2, 16, "bn2"));

			// remaining layers can be constructed inline using conv2 output dims
			//sequence.addLayer(std::make_unique<FlattenTo1d<float>>(&tensorPool, "f"));
			sequence.addLayer(std::make_unique<Linear<float>>(&tensorPool, 5 * out_w2 * out_h2, 1024, 16, "linear1"));
			sequence.addLayer(std::make_unique<Layernorm<float>>(&tensorPool, vec{16, 1, 1024}, vec{1, 1024}, "ln1"));
			sequence.addLayer(std::make_unique<ReLU<float>>(&tensorPool, "relu1"));
			sequence.addLayer(std::make_unique<Linear<float>>(&tensorPool, 1024, 10, 16, "linear2"));
		}
		softmax = std::make_unique<SoftmaxCrossEntropy<float>>(&tensorPool, 10, 1, 16, "softmax");
		optim.load_tensors(sequence);
	}

	void train_epoch(int i){
		dataLoader->loadMNIST("dataset/train.idx3-ubyte", "dataset/labels.idx1-ubyte");

		Tensor<float>* input;
		progressbar bar(100, true, std::cout);
		for (uint32_t i = 0; i < dataLoader->num_batches; i++){
			input = dataLoader->imagesBatchTensors[i];
			sequence.forward(input);
			softmax->target = dataLoader->labelTensors[i];
			auto loss = softmax->forward(sequence.output);
			
			loss->backward();
			
			//dynamic_cast<Conv2d<float>*>(sequence.layers[0].get())->weight_tensor->printGradient();

			optim.step();
			tensorPool.zero_out_all_grads();
			bar.update();
		}
		auto loss = tensorPool.find_mean_of_tensor(softmax->output->name);
		auto cls = tensorPool.get_highest_classes_from_dist(softmax->softmax_output->name);
		write_tensor_image_png(input, "image_" + std::to_string(cls[3]) + "_epoch:" + std::to_string(i) + ".png", 3);
		std::cout << "train_loss: " << loss << "\n";
	}

	void test_epoch(){
		testDataLoader->loadMNIST("dataset/test.idx3-ubyte", "dataset/test_labels.idx1-ubyte");
		for (uint32_t k = 0; k < testDataLoader->num_batches; k++){
			auto* input = testDataLoader->imagesBatchTensors[k];
			sequence.forward(input);
			softmax->target = testDataLoader->labelTensors[k];
			softmax->forward(sequence.output);
		}
		auto loss = tensorPool.find_mean_of_tensor(softmax->output->name);
		std::cout << "val_loss: " << loss << "\n";
	}

	void save_model(){
		auto save_stream = std::ofstream("models/MNIST_model.vnsr", std::ios::binary);
		sequence.serializeTrainableTensors(save_stream);
		save_stream.close();
	}

	void load_model(){
		auto load_stream = std::ifstream("models/MNIST_model.vnsr", std::ios::binary);
		sequence.loadFromSerializedTensor(load_stream);
		load_stream.close();
	}

};

/*
int main(void){
	
	Init init;
	device_initialization(init);
	Allocator* allocator = new Allocator(&init);

	{
		Trainer t = Trainer(allocator);

		for (int i = 0; i < 4; i++){
			t.train_epoch(i);
			//t.test_epoch();
		}
		t.save_model();
	}

	delete allocator;

	return 0;
}
*/

/*
int main(void) {
	Init init;
	device_initialization(init);
	Allocator* allocator = new Allocator(&init);
	TensorPool<float> pool(allocator);

	auto &input = pool.createTensor({16, 1, 256}, "input");
	pool.tensor_fill_random("input", -2.5f, 2.5f);

	auto linear1 = Linear<float>(&pool, 256, 256, 16, "linear1");
	auto r1 = ReLU<float>(&pool, 256, 16, "relu1");
	auto bn1 = Layernorm<float>(&pool, {16, 1, 256}, {1, 256}, "bn1");
	auto linear2 = Linear<float>(&pool, 256, 256, 16, "linear2");
	auto r2 = ReLU<float>(&pool, 256, 16, "relu2");
	auto bn2 = BatchNorm1d<float>(&pool, 256, 1, 16, "bn2");
	auto linear3 = Linear<float>(&pool, 256, 16, 16, "linear3");
	auto softmax = SoftmaxCrossEntropy<float>(&pool, 16, 1, 16, "softmax");

	auto &targets = pool.createTensor({16, 1, 16}, "targets");
	targets.setElement(1.0f, {0, 0, 1});
	targets.setElement(1.0f, {1, 0, 2});
	targets.setElement(1.0f, {2, 0, 3});
	targets.setElement(1.0f, {3, 0, 4});
	targets.setElement(1.0f, {4, 0, 5});
	targets.setElement(1.0f, {5, 0, 5});
	targets.setElement(1.0f, {6, 0, 6});
	targets.setElement(1.0f, {7, 0, 7});
	targets.setElement(1.0f, {8, 0, 8});
	targets.setElement(1.0f, {9, 0, 9});
	targets.setElement(1.0f, {10, 0, 0});
	targets.setElement(1.0f, {12, 0, 10});
	targets.setElement(1.0f, {13, 0, 11});
	targets.setElement(1.0f, {14, 0, 12});
	targets.setElement(1.0f, {15, 0, 13});

	softmax.target = &targets;

	// longer residual connection: input + output after two layers (linear1->bn1->linear2->bn2)
	auto &o = input + *r2(bn2(linear2(r1(bn1(linear1(&input))))));

	auto loss = softmax(linear3(&o));

	loss->backward();

	//linear1.weights->printGradient();

	delete allocator;
	return 0;
}
*/

/*
// broadcasting test
int main(void) {
	Init init;
	device_initialization(init);
	Allocator* allocator = new Allocator(&init);
	TensorPool<float> pool(allocator);

	auto &t1 = pool.createTensor({5, 32, 32}, "t1");
	pool.tensor_fill_random("t1", 1.0f, 1.0f);

	auto &t2 = pool.createTensor({1, 1, 1}, "t2");

	t2.setElement(0.5f, {0, 0, 0});

	auto &out = t1 * t2;

	out.print();

	return 0;
}
*/

// Example usage: load MNIST batch and write the first image to output_images/mnist_sample.png
/*
int main(void){
	Init init;
	device_initialization(init);
	Allocator* allocator = new Allocator(&init);
	TensorPool<float> tensorPool(allocator);

	auto loader = MNISTDataloader<float>(&tensorPool, 16, 100);
	loader.loadMNIST("dataset/train.idx3-ubyte", "dataset/labels.idx1-ubyte");

	// take first batch tensor and write its first image
	Tensor<float>* batch = loader.imagesBatchTensors[0];
	write_tensor_image_png(batch, "mnist_sample.png", 0, "output_images");

	// optional: print the label for that image
	float* lbl = reinterpret_cast<float*>(loader.labelTensors[0]->dataBuffer->memMap);
	int label = std::distance(lbl, std::max_element(lbl, lbl + 10));
	std::cout << "Label = " << label << std::endl;

	delete allocator;
	return 0;
}
*/

// MSE loss sanity check
/*
int main(void){

	Init init;
	device_initialization(init);
	Allocator* allocator = new Allocator(&init);

	TensorPool<float> tensorPool = TensorPool<float>(allocator);

	MSEloss<float> mse = MSEloss<float>(&tensorPool, 16, "mse");

	Tensor<float>* input = &tensorPool.createTensor({16, 3, 48, 48}, "input");
	Tensor<float>* target = &tensorPool.createTensor({16, 3, 48, 48}, "target");

	tensorPool.tensor_fill_random(input->name, -1.0f, 1.0f);
	tensorPool.tensor_fill_random(target->name, -0.1f, 0.1f);

	mse.target = target;
	mse.forward(input);

	mse.output->print();

	delete allocator;
	return 0;
}
*/