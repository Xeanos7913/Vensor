/*
This main file contains some testing code (in the commented out parts) and an example MNIST handwritten digit recognision neural network training loop.
*/

#include <iostream>
#include <fstream>
#define DEBUG
#include "../include/Tensor.hpp"
#include "../include/Neural.hpp"
#include "../include/VkCalcium.hpp"
#include "../include/Dataloader.hpp"
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

// Layernorm test
/*
int main(void) {
	Init init;
	device_initialization(init);
	Allocator* allocator = new Allocator(&init);
	TensorPool<float> tensorPool(allocator);

	auto &input = tensorPool.createTensor({16, 32, 32}, "input");
	tensorPool.tensor_fill_random("input", -1.0f, 1.0f);
	auto batchnorm = Layernorm<float>(&tensorPool, {16, 32, 32}, {32, 32}, "batchnorm");
	auto batchnorm2 = Layernorm<float>(&tensorPool, {16, 32, 32}, {32, 32}, "batchnorm2");

	batchnorm.forward(&input);
	batchnorm2.forward(batchnorm.output);

	batchnorm2.backward(batchnorm.output);
	batchnorm.backward(&input);

	batchnorm2.save_mean->print();
	batchnorm2.save_rstd->print();
	
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
	TensorPool<float> tensorPool(allocator);

	auto &inputImage = tensorPool.createTensor({5, 1, 28, 28}, "inputImage");
	tensorPool.tensor_fill_random("inputImage", -1.0f, 1.0f); // make input image completely white

	auto conv = std::make_unique<Conv2d3x3<float>>(&tensorPool, 1, 2, 5, 28, 28, "conv1", 1, 1, 0, 0);
	auto bn = std::make_unique<BatchNorm2d<float>>(&tensorPool, 2, 26, 26, 5, "batchnorm-2d");
	auto flatten = std::make_unique<FlattenTo<float>>(&tensorPool, vec{5, 1, 2 * 26 * 26}, "flatten"); // output is (5, 1, 2 * 26 * 26)
	auto linear = std::make_unique<Linear<float>>(&tensorPool, vec{1, 16, 2 * 26 * 26}, vec{5, 1, 16}, "linear"); // output is (5, 1, 16)
	auto batchnorm = std::make_unique<BatchNorm1d<float>>(&tensorPool, 16, 1, 5, "batchnorm");
	auto softmax = std::make_unique<SoftmaxCrossEntropy<float>>(&tensorPool, 16, 1, 5, "softmax");

	// set all the labels to be class = 1
	auto &target = tensorPool.createTensor({5, 1, 16}, "targets");
	target.dataBuffer->set(1, 1.0f);
	target.dataBuffer->set(off2or3(target.strides, 1, 0, 1), 1.0f);
	target.dataBuffer->set(off2or3(target.strides, 2, 0, 1), 1.0f);
	target.dataBuffer->set(off2or3(target.strides, 3, 0, 1), 1.0f);
	target.dataBuffer->set(off2or3(target.strides, 4, 0, 1), 1.0f);
	softmax->target = &target;

	auto model = Sequential<float>(&tensorPool, "model");
	model.addLayer(std::move(conv));
	model.addLayer(std::move(bn));
	model.addLayer(std::move(flatten));
	model.addLayer(std::move(linear));
	model.addLayer(std::move(batchnorm));

	// run one forward-backward pass
	model.forward(&inputImage);
	softmax->forward(model.output);
	softmax->backward(model.output);
	model.backward(&inputImage);

	for (auto t : model.layers[1]->getTensors()){
		std::cout << "Name: " << t->name << ": \n";
		t->printGradient();
	}

	delete allocator;
	return 0;
}
*/

// A handwritten digit recognision neural network
struct Trainer {
	
	// Vulkan stuff:
	Init init;
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
	Trainer() {
		device_initialization(init);
		allocator = new Allocator(&init);
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
			auto c1 = std::make_unique<Conv2d3x3<float>>(&tensorPool, 1, 5, 1, 28, 28, "conv1", 1, 1, 0, 0);
			auto out_w1 = c1->output_width;
			auto out_h1 = c1->output_height;
			sequence.addLayer(std::move(c1));
			// batchnorm1 depends on conv1 output dims
			sequence.addLayer(std::make_unique<BatchNorm2d<float>>(&tensorPool, 5, out_w1, out_h1, 1, "bn1", 1));
			// build conv2 using conv1's output dims as its input dims, grab conv2 dims, move into sequence
			auto c2 = std::make_unique<Conv2d3x3<float>>(&tensorPool, 5, 5, 1, out_h1, out_w1, "conv2", 1, 1, 0, 0);
			auto out_w2 = c2->output_width;
			auto out_h2 = c2->output_height;
			sequence.addLayer(std::move(c2));
			// batchnorm2 depends on conv2 output dims
			sequence.addLayer(std::make_unique<BatchNorm2d<float>>(&tensorPool, 5, out_w2, out_h2, 1, "bn2", 1));

			// remaining layers can be constructed inline using conv2 output dims
			sequence.addLayer(std::make_unique<FlattenTo1d<float>>(&tensorPool, "f"));
			sequence.addLayer(std::make_unique<Linear<float>>(&tensorPool, 5 * out_w2 * out_h2, 1024, 1, "linear1"));
			sequence.addLayer(std::make_unique<BatchNorm1d<float>>(&tensorPool, 1024, 1, 1, "bn3", 1));
			sequence.addLayer(std::make_unique<ReLU<float>>(&tensorPool, 1024, 1, "relu1"));
			sequence.addLayer(std::make_unique<Linear<float>>(&tensorPool, 1024, 10, 1, "linear2"));
		}
		softmax = std::make_unique<SoftmaxCrossEntropy<float>>(&tensorPool, 10, 1, 1, "softmax");
		optim.load_tensors(sequence);
	}

	void train_epoch(){
		dataLoader->loadMNIST("dataset/train.idx3-ubyte", "dataset/labels.idx1-ubyte");
		for (uint32_t i = 0; i < dataLoader->num_batches; i++){
			auto* input = dataLoader->imagesBatchTensors[i];
			sequence.forward(input);
			softmax->target = dataLoader->labelTensors[i];
			softmax->forward(sequence.output);
			
			softmax->backward(sequence.output);
			sequence.backward(input);
			optim.step();
		}
		auto loss = tensorPool.find_mean_of_tensor(softmax->output->name);
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

	~Trainer(){
		delete allocator;
	}
};

int main(void){
	
	Trainer t = Trainer();

	t.load_model();

	for (int i = 0; i < 1; i++){
		//t.train_epoch();
		t.test_epoch();
	}

	return 0;
}

// dataloader sanity check:
/*
int main(void){
	Init init;
	device_initialization(init);
	Allocator* allocator = new Allocator(&init);
	TensorPool<float> tensorPool(allocator);

	uint32_t batch_idx = 0;
	uint32_t img_idx   = 7;

	auto loader = MNISTDataloader<float>(&tensorPool, 16, 100);
	loader.loadMNIST("dataset/train.idx3-ubyte", "dataset/labels.idx1-ubyte");

	Tensor<float>* tensor = loader.imagesBatchTensors[batch_idx];
	float* data = reinterpret_cast<float*>(tensor->dataBuffer->memMap);

	uint32_t B = loader.batch_size;
	uint32_t C = loader.channels; // 1
	uint32_t H = loader.height;   // 28
	uint32_t W = loader.width;    // 28

	// NCHW layout => offset = (b * C * H * W) + (c * H * W)
	const float* src = data + img_idx * C * H * W;

	// convert to uint8
	std::vector<uint8_t> img8(H * W * C);
	for (size_t i = 0; i < H * W * C; ++i)
		img8[i] = static_cast<uint8_t>(std::clamp(src[i], 0.0f, 1.0f) * 255.0f);

	// write to PNG
	stbi_write_png("mnist_sample.png", W, H, C, img8.data(), W * C);

	// optional: print the label
	float* lbl = reinterpret_cast<float*>(loader.labelTensors[batch_idx]->dataBuffer->memMap);
	int label = std::distance(lbl + img_idx * 10, std::max_element(lbl + img_idx * 10, lbl + (img_idx + 1) * 10));
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