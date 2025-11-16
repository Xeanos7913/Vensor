#pragma once
#include "VkBootstrap.h"
#include "volk.h"
#include <iostream>
#include <string>
#include <vector>
#include <functional>
#include "VkMemAlloc.hpp"
#include "signal.hpp"

// used by SequentialgpuTask
//struct PushConstants {
//    uint64_t bufferAddress;
//};

// uses the default descriptors of vulkan.
template<typename T>
class gpuTask {
private:
    // Vulkan handles:
    VkQueue queue;
    VkPipelineLayout pipeline_layout;
    VkPipeline compute_pipeline;
    VkCommandPool command_pool;
    VkCommandBuffer command_buffer;

    // the output is a standalone buffer
    StandaloneBuffer<T> output;

    // Vulkan stuff:
    VkDescriptorSet descriptor_set; // only one descriptor set per task
    VkDescriptorPool descriptor_pool; // the pool will only contain one desc_set
    VkDescriptorSetLayout descriptor_set_layout; // the layout will have bindings for all the inputs and then the output

    uint32_t taskNumber = 0;
    uint32_t numInputs = 0;
    uint32_t numOutputs = 0;

    Init& init; // global vulkan handles

    std::vector<char> spv_code = std::vector<char>{};// we'll have shader bytecode in this
    
    std::function<void()> callback;

    //get the compute queue family inside the GPU
    int get_queues() {
        auto gq = init.device.get_queue(vkb::QueueType::compute);
        if (!gq.has_value()) {
            std::cout << "failed to get queue: " << gq.error().message() << "\n";
            return -1;
        }
        queue = gq.value();
        return 0;
    }

    // create the descriptors for the inputs and the output. Inputs are readonly, output is writeonly
    void create_descriptors() {
        uint32_t total_buffers = static_cast<uint32_t>(buffers.buffers.size() + 1);

        // --- Descriptor Pool ---
        VkDescriptorPoolSize pool_size = {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            total_buffers
        };

        VkDescriptorPoolCreateInfo pool_info = {};
        pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.maxSets = 1;
        pool_info.poolSizeCount = 1;
        pool_info.pPoolSizes = &pool_size;
        init.disp.createDescriptorPool(&pool_info, nullptr, &descriptor_pool);

        // --- Descriptor Set Layout ---
        std::vector<VkDescriptorSetLayoutBinding> bindings(total_buffers);
        for (uint32_t i = 0; i < buffers.buffers.size(); ++i) {
            // the input descriptors are managed by the MemPool, so they're already created
            bindings[i] = buffers.buffers[i].binding;
        }
        output.bindingIndex = buffers.buffers.size();
        output.createDescriptors();
        bindings.back() = output.binding;

        VkDescriptorSetLayoutCreateInfo dsl_info = {};
        dsl_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dsl_info.bindingCount = total_buffers;
        dsl_info.pBindings = bindings.data();
        init.disp.createDescriptorSetLayout(&dsl_info, nullptr, &descriptor_set_layout);

        // --- Descriptor Set Allocation ---
        VkDescriptorSetAllocateInfo ds_allocate_info = {};
        ds_allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ds_allocate_info.descriptorPool = descriptor_pool;
        ds_allocate_info.descriptorSetCount = 1;
        ds_allocate_info.pSetLayouts = &descriptor_set_layout;
        init.disp.allocateDescriptorSets(&ds_allocate_info, &descriptor_set);

        // --- Update Descriptor Set ---
        std::vector<VkWriteDescriptorSet> write_desc_sets(total_buffers);

        // Input Buffers
        for (uint32_t i = 0; i < buffers.buffers.size(); ++i) {
            buffers.buffers[i].updateDescriptorSet(descriptor_set);
            write_desc_sets[i] = buffers.buffers[i].wrt_desc_set;
        }
        // output buffer
        output.updateDescriptorSet(descriptor_set);
        write_desc_sets.back() = output.wrt_desc_set;

        init.disp.updateDescriptorSets(write_desc_sets.size(), write_desc_sets.data(), 0, nullptr);
    }

    // create the compute pipeline for the gpu execution of this task
    void create_pipeline() {
        VkShaderModuleCreateInfo create_info = {};
        create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        create_info.codeSize = spv_code.size();
        create_info.pCode = reinterpret_cast<const uint32_t*>(spv_code.data());

        VkShaderModule shader_module;
        init.disp.createShaderModule(&create_info, nullptr, &shader_module);

        VkPipelineShaderStageCreateInfo shader_stage_info = {};
        shader_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shader_stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        shader_stage_info.module = shader_module;
        shader_stage_info.pName = "main";

        VkPipelineLayoutCreateInfo pipeline_layout_info = {};
        pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout_info.setLayoutCount = 1;
        pipeline_layout_info.pSetLayouts = &descriptor_set_layout;
        pipeline_layout_info.pushConstantRangeCount = 0;
        init.disp.createPipelineLayout(&pipeline_layout_info, nullptr, &pipeline_layout);

        VkComputePipelineCreateInfo pipeline_info = {};
        pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipeline_info.stage = shader_stage_info;
        pipeline_info.layout = pipeline_layout;
        pipeline_info.basePipelineHandle = VK_NULL_HANDLE;
        init.disp.createComputePipelines(VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &compute_pipeline);

        init.disp.destroyShaderModule(shader_module, nullptr);
    }

    // The command pool where we'll submit our commands for buffer recording and compute shader dispatching:
    void create_command_pool() {
        VkCommandPoolCreateInfo pool_info = {};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.queueFamilyIndex = init.device.get_queue_index(vkb::QueueType::compute).value();
        init.disp.createCommandPool(&pool_info, nullptr, &command_pool);

        VkCommandBufferAllocateInfo allocate_info = {};
        allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocate_info.commandPool = command_pool;
        allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocate_info.commandBufferCount = 1;
        init.disp.allocateCommandBuffers(&allocate_info, &command_buffer);
    }

public:
    // must have function, because this class is a listener object. It must be named in this exact way
    // NEEDS TO BE A PUBLIC METHOD
	void onSignal() {
		// this is called when the memPool or output buffer broadcasts a signal
        std::cout << "Signal Called\n";
        if (callback) {
            callback();
        }
	}

    template<typename Func, typename... Args>
    void setCallback(Func&& func, Args&&... args) {
        callback = [&func, &...capturedArgs = args]() mutable {
            std::invoke(func, capturedArgs...);
            };
    }

    gpuTask(std::vector<char>& spvByteCodeCompShader, Init& init, uint32_t numInputs, uint32_t numOutputs, Allocator* allocator) : spv_code(spvByteCodeCompShader), init(init), buffers(MemPool<T>(numInputs, allocator)), output(numOutputs, allocator) {
        get_queues();
        create_command_pool();

		buffers.descUpdateQueued.addListener(this);
		output.descUpdateQueued.addListener(this);
    };
    gpuTask(gpuTask& base, uint32_t numInputs) {
        init = base.init;
        queue = base.queue;
        pipeline_layout = base.pipeline_layout;
        command_pool = base.command_pool;
        command_buffer = base.command_buffer;
        compute_pipeline = base.compute_pipeline;
        descriptor_pool = base.descriptor_pool;
        descriptor_set = base.descriptor_set;
        descriptor_set_layout = base.descriptor_set_layout;
        buffers = MemPool<T>(&init, numInputs);
    }
    MemPool<T> buffers;
    // load floating point inputs into GPU memory using memory staging:
    void load_inputs(std::vector<std::vector<T>>& input_data) {
        // Ensure input vectors are properly sized
        size_t num_inputs = input_data.size();
        for (int i = 0; i < num_inputs; i++) {
            if (!buffers.push_back(input_data[i])) {
                throw std::runtime_error("Couldn't push_back into input buffer! Out of memory!");
            }
        }
        numInputs = num_inputs;

        create_descriptors();
        create_pipeline();
    }
    
    void updateBuffers() {
        numInputs = buffers.size();
        create_descriptors();
        create_pipeline();
    }

    std::vector<T> compute() {
        
        // --- Begin command buffer recording ---
        VkCommandBufferBeginInfo begin_info = {};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        init.disp.beginCommandBuffer(command_buffer, &begin_info);

        // Bind compute pipeline and descriptor set
        init.disp.cmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute_pipeline);
        init.disp.cmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);

        // Dispatch compute shader
        uint32_t workgroup_count = 3;
        init.disp.cmdDispatch(command_buffer, workgroup_count, 1, 1);

        // End command buffer recording
        init.disp.endCommandBuffer(command_buffer);

        // Create Fence for GPU Synchronization
        VkFenceCreateInfo fence_info = {};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkFence compute_fence;
        init.disp.createFence(&fence_info, nullptr, &compute_fence);

        // Submit command buffer with fence
        VkSubmitInfo submit_info = {};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command_buffer;
        init.disp.queueSubmit(queue, 1, &submit_info, compute_fence);

        // Wait for GPU to finish computation
        init.disp.waitForFences(1, &compute_fence, VK_TRUE, UINT64_MAX);

        // Cleanup fence
        init.disp.destroyFence(compute_fence, nullptr);
        cleanup();

        return output.downloadBuffer();
    }

    ~gpuTask() {}

private:
    void cleanup() {
        init.disp.destroyCommandPool(command_pool, nullptr);
        init.disp.destroyPipeline(compute_pipeline, nullptr);
        init.disp.destroyPipelineLayout(pipeline_layout, nullptr);
        init.disp.destroyDescriptorPool(descriptor_pool, nullptr);
        init.disp.destroyDescriptorSetLayout(descriptor_set_layout, nullptr);
    }
};
template<typename T>
struct ComputeSequence;

// For multiple gpuTask's to work one after the other
template<typename T>
class SequentialgpuTask {
    friend struct ComputeSequence<T>;
private:
    // Vulkan handles:
    VkQueue queue;
    VkPipelineLayout pipeline_layout;
    VkPipeline compute_pipeline;
    VkCommandPool command_pool;

    // the input buffers are inside the MemPool of the ComputeSequence
    std::vector<uint32_t> inputIndices; // the indices of the buffers stored inside the MemPool

    // the output buffers 
    std::vector<uint32_t> outputIndices; // the indices of the buffers stored inside the MemPool

    // Vulkan stuff (the ComputeSequence contains all of the pool stuff):
    VkDescriptorSet descriptor_set = {}; // only one descriptor set per task
    VkDescriptorPool descriptor_pool; // pool is managed by the ComputeSequence
    VkDescriptorSetLayout descriptor_set_layout; // the layout will have bindings for all the inputs and then the outputs

    uint32_t taskNumber = 0;
    uint32_t numInputs = 0;
    uint32_t numOutputs = 0;

    Init& init; // global vulkan handles

    std::vector<char> spv_code = std::vector<char>{};// we'll have shader bytecode in this

    // create the descriptors for the inputs and the outputs. Inputs are readonly, output is writeonly
    void create_descriptors(MemPool<T>& memPool) {
        uint32_t total_buffers = static_cast<uint32_t>(inputIndices.size() + outputIndices.size());

        // --- Descriptor Set Layout ---
        std::vector<VkDescriptorSetLayoutBinding> bindings(total_buffers);
        for (uint32_t i = 0; i < inputIndices.size(); i++) {
            bindings[i] = memPool.buffers[inputIndices[i]].binding;
        }
        for (uint32_t i = 0; i < outputIndices.size(); i++) {
            bindings[i + inputIndices.size()] = memPool.buffers[outputIndices[i]].binding;
        }

        VkDescriptorSetLayoutCreateInfo dsl_info = {};
        dsl_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dsl_info.bindingCount = total_buffers;
        dsl_info.pBindings = bindings.data();
        init.disp.createDescriptorSetLayout(&dsl_info, nullptr, &descriptor_set_layout);
    }

    // called after descriptor sets are allocated by the ComputeSequence
    void update_descriptors(MemPool<T>& memPool) {
        uint32_t total_buffers = static_cast<uint32_t>(inputIndices.size() + outputIndices.size());
        // --- Update Descriptor Set ---
        std::vector<VkWriteDescriptorSet> write_desc_sets(total_buffers);

        // Input Buffers
        for (uint32_t i = 0; i < inputIndices.size(); i++) {
            memPool.buffers[inputIndices[i]].updateDescriptorSet(descriptor_set);
            write_desc_sets[i] = memPool.buffers[inputIndices[i]].wrt_desc_set;
        }
        // output buffers
        for (uint32_t i = 0; i < outputIndices.size(); i++) {
            memPool.buffers[outputIndices[i]].updateDescriptorSet(descriptor_set);
            write_desc_sets[i + inputIndices.size()] = memPool.buffers[outputIndices[i]].wrt_desc_set;
        }

        init.disp.updateDescriptorSets(write_desc_sets.size(), write_desc_sets.data(), 0, nullptr);
    }

    // create the compute pipeline for the gpu execution of this task
    void create_pipeline() {
        VkShaderModuleCreateInfo create_info = {};
        create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        create_info.codeSize = spv_code.size();
        create_info.pCode = reinterpret_cast<const uint32_t*>(spv_code.data());

        VkShaderModule shader_module;
        init.disp.createShaderModule(&create_info, nullptr, &shader_module);

        VkPipelineShaderStageCreateInfo shader_stage_info = {};
        shader_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shader_stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        shader_stage_info.module = shader_module;
        shader_stage_info.pName = "main";

        VkPipelineLayoutCreateInfo pipeline_layout_info = {};
        pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout_info.setLayoutCount = 1;
        pipeline_layout_info.pSetLayouts = &descriptor_set_layout;
        pipeline_layout_info.pushConstantRangeCount = 0;
        init.disp.createPipelineLayout(&pipeline_layout_info, nullptr, &pipeline_layout);

        VkComputePipelineCreateInfo pipeline_info = {};
        pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipeline_info.stage = shader_stage_info;
        pipeline_info.layout = pipeline_layout;
        pipeline_info.basePipelineHandle = VK_NULL_HANDLE;
        init.disp.createComputePipelines(VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &compute_pipeline);

        init.disp.destroyShaderModule(shader_module, nullptr);
    }

public:
    // must have function, because this class is a listener object. It must be named in this exact way
    // NEED TO BE PUBLIC
    void onSignal() {
        // this is called when the memPool broadcasts a signal
    }

    SequentialgpuTask(std::vector<char>& spvByteCodeCompShader, Init& init) : spv_code(spvByteCodeCompShader), init(init) {};

    // load buffer indices. The buffers themselves are stored in one massive MemPool in the ComputeSequence for efficiency
    void load_indices(std::vector<uint32_t>& inputIndicess, std::vector<uint32_t>& outputIndicess) {
        numInputs = inputIndicess.size();
        inputIndices = inputIndicess;
        numOutputs = outputIndicess.size();
        outputIndices = outputIndicess;
    }

    // dispatch commands
    void compute(VkCommandBuffer& command_buffer, uint32_t workgroup_x, uint32_t workgroup_y, uint32_t workgroup_z) {
        // Bind compute pipeline and descriptor set
        init.disp.cmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute_pipeline);
        init.disp.cmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);

        // Dispatch compute shader
        init.disp.cmdDispatch(command_buffer, workgroup_x, workgroup_y, workgroup_z);
    }

    ~SequentialgpuTask() {}

private:
    void cleanup() {
        init.disp.destroyCommandPool(command_pool, nullptr);
        init.disp.destroyPipeline(compute_pipeline, nullptr);
        init.disp.destroyPipelineLayout(pipeline_layout, nullptr);
        init.disp.destroyDescriptorPool(descriptor_pool, nullptr);
        init.disp.destroyDescriptorSetLayout(descriptor_set_layout, nullptr);
    }
};

int device_initialization(Init& init) {  

    if(volkInitialize() != VK_SUCCESS){
        throw std::runtime_error("Volk couldn't load the vulkan loader from system. It may be missing.");
    }

    vkb::InstanceBuilder instance_builder;  
    auto instance_ret = instance_builder.use_default_debug_messenger()  
        .request_validation_layers()  
        .require_api_version(VK_API_VERSION_1_3) // Important!    
        .set_headless() // Skip vk-bootstrap trying to create WSI  
        .build();  
    if (!instance_ret) {  
        std::cout << "failed to create instance: ";
        std::cout << instance_ret.error().message() << "\n";  
        return -1;  
    }  
    init.instance = instance_ret.value();  

    volkLoadInstance(init.instance.instance);

    std::cout << "Vulkan Instance created\n";
    init.inst_disp = init.instance.make_table();  
    std::cout << "Instance Dispatch Table created\n";

    VkPhysicalDeviceBufferDeviceAddressFeatures buffer_device_address_features = {};  
    buffer_device_address_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;  
    buffer_device_address_features.bufferDeviceAddress = VK_TRUE;  

    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT float32_atomic_features = {};
    float32_atomic_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
    float32_atomic_features.shaderBufferFloat32Atomics = VK_TRUE;  
    float32_atomic_features.shaderBufferFloat32AtomicAdd = VK_TRUE;  
    float32_atomic_features.shaderSharedFloat32Atomics = VK_TRUE;  
    float32_atomic_features.shaderSharedFloat32AtomicAdd = VK_TRUE;  
    float32_atomic_features.shaderImageFloat32Atomics = VK_TRUE;  
    float32_atomic_features.shaderImageFloat32AtomicAdd = VK_TRUE; 

    VkPhysicalDeviceSubgroupSizeControlFeaturesEXT subgroup_size_control_features = {};
    subgroup_size_control_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT;
    subgroup_size_control_features.subgroupSizeControl = VK_TRUE;
    subgroup_size_control_features.computeFullSubgroups = VK_TRUE;

    VkPhysicalDeviceShaderSubgroupExtendedTypesFeatures shader_subgroup_extended_types_features = {};
    shader_subgroup_extended_types_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_EXTENDED_TYPES_FEATURES;
    shader_subgroup_extended_types_features.shaderSubgroupExtendedTypes = VK_TRUE;

    buffer_device_address_features.pNext = &float32_atomic_features;  
    float32_atomic_features.pNext = &subgroup_size_control_features;
    subgroup_size_control_features.pNext = &shader_subgroup_extended_types_features;

    vkb::PhysicalDeviceSelector phys_device_selector(init.instance);  
    auto phys_device_ret = phys_device_selector  
        .add_required_extension(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME)  
        .add_required_extension(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME)  
        .add_required_extension(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME)
        .add_required_extension(VK_KHR_SHADER_SUBGROUP_EXTENDED_TYPES_EXTENSION_NAME)
        .add_required_extension_features(buffer_device_address_features)  
        .add_required_extension_features(float32_atomic_features)
        .add_required_extension_features(subgroup_size_control_features)
        .add_required_extension_features(shader_subgroup_extended_types_features)
        .select();  

    if (!phys_device_ret) {  
        std::cout << "failed to select physical device: ";
        std::cout << phys_device_ret.error().message() << "\n";  
        return -1;  
    }  
    vkb::PhysicalDevice physical_device = phys_device_ret.value();  
    std::cout << "Physical Device selected: " << physical_device.properties.deviceName << "\n";

    vkb::DeviceBuilder device_builder{ physical_device };  
    auto device_ret = device_builder.build();  
    if (!device_ret) {  
        std::cout << "failed to create logical device: ";
        std::cout << device_ret.error().message() << "\n";  
        return -1;  
    }  
    init.device = device_ret.value();  
    std::cout << "Logical Device created\n";
    volkLoadDevice(init.device.device);

    init.disp = init.device.make_table();  
    std::cout << "Device Dispatch Table created\n";

    return 0;  
}

// Sequentially executes SequentialgpuTasks efficiently. 
// Uses one huge memory pool for all the inputs and outputs of the tasks so one task's output can be the next task's input
template<typename T>
struct ComputeSequence {
    std::vector<SequentialgpuTask<T>> tasks;
    VkQueue queue;
    VkCommandPool commandPool;
    VkDescriptorPool descriptorPool;
    uint32_t totalNumBuffers;

    VkCommandBuffer command_buffer;
    MemPool<T> allBuffers;

    Allocator* allocator;

    ComputeSequence(uint32_t numBuffers, Allocator* allocator) : allocator(allocator), allBuffers(numBuffers, allocator) {
        get_queues();
        create_command_pool();
    };

    //get the compute queue family inside the GPU
    int get_queues() {
        auto gq = allocator->init->device.get_queue(vkb::QueueType::compute);
        if (!gq.has_value()) {
            std::cout << "failed to get queue: " << gq.error().message() << "\n";
            return -1;
        }
        queue = gq.value();
        return 0;
    }

    void create_command_pool() {
        VkCommandPoolCreateInfo pool_info = {};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.queueFamilyIndex = allocator->init->device.get_queue_index(vkb::QueueType::compute).value();
        allocator->init->disp.createCommandPool(&pool_info, nullptr, &commandPool);
    }

    void create_descriptor_pool() {
        // --- Descriptor Pool ---
        VkDescriptorPoolSize pool_size = {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            totalNumBuffers
        };

        VkDescriptorPoolCreateInfo pool_info = {};
        pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.maxSets = tasks.size();
        pool_info.poolSizeCount = 1;
        pool_info.pPoolSizes = &pool_size;
        allocator->init->disp.createDescriptorPool(&pool_info, nullptr, &descriptorPool);
    }

    void initializeCompute() {
        // create all the descriptors

        create_descriptor_pool();

        auto desc_set_layouts = std::vector<VkDescriptorSetLayout>(tasks.size());
        auto desc_sets = std::vector<VkDescriptorSet>(tasks.size());
        int i = 0;
        for (auto& task : tasks) {
            task.queue = queue;
            task.command_pool = commandPool;
            task.descriptor_pool = descriptorPool;
            task.create_descriptors(allBuffers);
            desc_set_layouts[i] = (task.descriptor_set_layout);
            desc_sets[i] = (task.descriptor_set);
            i++;
        }

        // allocate the descriptors
        VkDescriptorSetAllocateInfo ds_allocate_info = {};
        ds_allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ds_allocate_info.descriptorPool = descriptorPool;
        ds_allocate_info.descriptorSetCount = tasks.size();
        ds_allocate_info.pSetLayouts = desc_set_layouts.data();
        allocator->init->disp.allocateDescriptorSets(&ds_allocate_info, desc_sets.data());
        
        // update the descriptors
        i = 0;
        for (auto& task : tasks) {
            task.descriptor_set = desc_sets[i];
            task.update_descriptors(allBuffers);
            task.create_pipeline();
            i++;
        }
    }

    void addBuffers(std::vector<std::vector<T>>& inputs) {
        for (int i = 0; i < inputs.size(); i++) {
            if (!allBuffers.push_back(inputs[i])) {
                throw std::runtime_error("couldn't add buffers to compute sequence");
            }
        }
    }

    void addTask(SequentialgpuTask<T>& task) {
        tasks.push_back(task);
    }

    // do the tasks sequentially for given iterations
    void doTasks(uint32_t iterations, uint32_t workgroup_x, uint32_t workgroup_y, uint32_t workgroup_z) {
        std::vector<VkCommandBuffer> command_buffers(tasks.size());
        std::vector<VkSemaphore> semaphores(tasks.size());

        // Allocate command buffers
        VkCommandBufferAllocateInfo alloc_info = {};
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.commandPool = commandPool;
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandBufferCount = static_cast<uint32_t>(tasks.size());
        allocator->init->disp.allocateCommandBuffers(&alloc_info, command_buffers.data());

        // Create semaphores
        VkSemaphoreCreateInfo semaphore_info = {};
        semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        for (size_t i = 0; i < tasks.size(); i++) {
            allocator->init->disp.createSemaphore(&semaphore_info, nullptr, &semaphores[i]);
        }

        // Record commands
        for (size_t i = 0; i < tasks.size(); i++) {
            VkCommandBufferBeginInfo begin_info = {};
            begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

            allocator->init->disp.beginCommandBuffer(command_buffers[i], &begin_info);
            tasks[i].compute(command_buffers[i], workgroup_x, workgroup_y, workgroup_z);
            allocator->init->disp.endCommandBuffer(command_buffers[i]);
        }

        // Submit tasks with semaphore chaining
        for (uint32_t iter = 0; iter < iterations; iter++) {
            for (size_t i = 0; i < tasks.size(); i++) {
                VkSubmitInfo submit_info = {};
                submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                submit_info.commandBufferCount = 1;
                submit_info.pCommandBuffers = &command_buffers[i];

                VkPipelineStageFlags wait_stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

                // First task has no semaphore dependency
                if (i == 0) {
                    submit_info.waitSemaphoreCount = 0;
                    submit_info.pWaitSemaphores = nullptr;
                }
                else {
                    submit_info.waitSemaphoreCount = 1;
                    submit_info.pWaitSemaphores = &semaphores[i - 1];
                    submit_info.pWaitDstStageMask = &wait_stages;
                }

                submit_info.signalSemaphoreCount = 1;
                submit_info.pSignalSemaphores = &semaphores[i];

                allocator->init->disp.queueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE);
            }
        }

        // Wait for the last task to finish
        allocator->init->disp.queueWaitIdle(queue);

        // Cleanup
        for (size_t i = 0; i < tasks.size(); i++) {
            allocator->init->disp.destroySemaphore(semaphores[i], nullptr);
        }
        allocator->init->disp.freeCommandBuffers(commandPool, static_cast<uint32_t>(tasks.size()), command_buffers.data());
    }
};

//push const contains address of uniform buffer. Uniform buffer contains all the rest of the addresses.
template<typename T, typename PushConst, typename UniformBuffer>
struct gpuTaskNoDesc {
    // Vulkan handles:
    VkQueue queue;
    VkPipelineLayout pipeline_layout;
    VkPipeline compute_pipeline;
    VkCommandPool command_pool;
    VkCommandBuffer command_buffer;

    PushConst push_const;

    std::vector<char> spv_code = std::vector<char>{};
    std::function<void()> callback;
    Allocator* allocator;

	std::unique_ptr<StandaloneBuffer<UniformBuffer>> uniformBuffer; // the uniform buffer for the compute shader. Contains the addresses of other input buffers along with anything else.

    //get the compute queue family inside the GPU
    int get_queues() {
        auto gq = allocator->init->device.get_queue(vkb::QueueType::compute);
        if (!gq.has_value()) {
            std::cout << "failed to get queue: " << gq.error().message() << "\n";
            return -1;
        }
        queue = gq.value();
        return 0;
    }

    // create the compute pipeline for the gpu execution of this task  
    void create_pipeline(uint32_t tile_size[3]) {  
        VkShaderModuleCreateInfo create_info = {};  
        create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;  
        create_info.codeSize = spv_code.size();  
        create_info.pCode = reinterpret_cast<const uint32_t*>(spv_code.data());  
        VkShaderModule shader_module;  
        allocator->init->disp.createShaderModule(&create_info, nullptr, &shader_module);  

        VkPipelineShaderStageCreateInfo shader_stage_info = {};  
        shader_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;  
        shader_stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;  
        shader_stage_info.module = shader_module;  
        shader_stage_info.pName = "main";  

        // Move these outside the if block so their lifetime is sufficient
        struct WG { uint32_t x, y, z; } wg;
        VkSpecializationMapEntry maps[3];
        VkSpecializationInfo spec{};
        if (tile_size != nullptr) {  
            wg = { tile_size[0], tile_size[1], tile_size[2] };  
            maps[0] = {0, offsetof(WG, x), sizeof(uint32_t)};  
            maps[1] = {1, offsetof(WG, y), sizeof(uint32_t)};  
            maps[2] = {2, offsetof(WG, z), sizeof(uint32_t)};  
            spec.mapEntryCount = 3;  
            spec.pMapEntries = maps;  
            spec.dataSize = sizeof(WG);  
            spec.pData = &wg;  
            shader_stage_info.pSpecializationInfo = &spec;  
        } // else: skip specialization info, shader must provide tile size

        VkPushConstantRange push_constant_range = {};
        push_constant_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_constant_range.offset = 0;
        push_constant_range.size = sizeof(PushConst);

        VkPipelineLayoutCreateInfo pipeline_layout_info = {};
        pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout_info.setLayoutCount = 0; // no descriptor sets
        pipeline_layout_info.pushConstantRangeCount = 1;
        pipeline_layout_info.pPushConstantRanges = &push_constant_range;
        allocator->init->disp.createPipelineLayout(&pipeline_layout_info, nullptr, &pipeline_layout);

        VkComputePipelineCreateInfo pipeline_info = {};
        pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipeline_info.stage = shader_stage_info;
        pipeline_info.layout = pipeline_layout;
        pipeline_info.basePipelineHandle = VK_NULL_HANDLE;
        allocator->init->disp.createComputePipelines(VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &compute_pipeline);
        allocator->init->disp.destroyShaderModule(shader_module, nullptr);
    }

    // Modify the create_command_pool method to include the VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT flag  
    void create_command_pool() {  
        VkCommandPoolCreateInfo pool_info = {};  
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;  
        pool_info.queueFamilyIndex = allocator->init->device.get_queue_index(vkb::QueueType::compute).value();  
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; // Add this flag to allow command buffer reset  
        allocator->init->disp.createCommandPool(&pool_info, nullptr, &command_pool);  

        VkCommandBufferAllocateInfo allocate_info = {};  
        allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;  
        allocate_info.commandPool = command_pool;  
        allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;  
        allocate_info.commandBufferCount = 1;  
        allocator->init->disp.allocateCommandBuffers(&allocate_info, &command_buffer);  
    }

public:

    gpuTaskNoDesc(const std::vector<char>& spvByteCodeCompShader, Allocator* allocator, uint32_t tile_size[3]) : allocator(allocator), spv_code(spvByteCodeCompShader) {
		get_queues();
		create_command_pool();
		uniformBuffer = std::make_unique<StandaloneBuffer<UniformBuffer>>(1, allocator, VK_SHADER_STAGE_ALL);
		create_pipeline(tile_size);
    }

    gpuTaskNoDesc(){}

    // must have function, because this class is a listener object. It must be named in this exact way
    // NEEDS TO BE A PUBLIC METHOD
    void onSignal() {
        // this is called when the memPool or output buffer broadcasts a signal
        std::cout << "Signal Called\n";
        if (callback) {
            callback();
        }
    }

    template<typename Func, typename... Args>
    void setCallback(Func&& func, Args&&... args) {
        callback = [&func, &...capturedArgs = args]() mutable {
            std::invoke(func, capturedArgs...);
            };
    }

    void loadUniform(UniformBuffer& buf, PushConst& push) {
        push_const = push;
		uniformBuffer->alloc(buf);
    }

	// stops CPU execution until the GPU has finished.
    void execute(uint32_t workgroup_size[3]) {
        // Reset command buffer to avoid reuse issues
        allocator->init->disp.resetCommandBuffer(command_buffer, 0);

        // --- Begin command buffer recording ---
        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        allocator->init->disp.beginCommandBuffer(command_buffer, &begin_info);

        // Ensure visibility of prior shader writes
        VkMemoryBarrier mem_barrier{};
        mem_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mem_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mem_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        allocator->init->disp.cmdPipelineBarrier(
            command_buffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            1, &mem_barrier,
            0, nullptr,
            0, nullptr);

        // Bind compute pipeline
        allocator->init->disp.cmdBindPipeline(
            command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute_pipeline);

        // Push constants
        allocator->init->disp.cmdPushConstants(
            command_buffer, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
            0, sizeof(PushConst), &push_const);

        // Dispatch
        allocator->init->disp.cmdDispatch(
            command_buffer, workgroup_size[0], workgroup_size[1], workgroup_size[2]);

        // End recording
        allocator->init->disp.endCommandBuffer(command_buffer);

        // Fence for sync
        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkFence fence;
        allocator->init->disp.createFence(&fence_info, nullptr, &fence);

        // Submit & wait
        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command_buffer;

        allocator->init->disp.queueSubmit(queue, 1, &submit_info, fence);
        allocator->init->disp.waitForFences(1, &fence, VK_TRUE, UINT64_MAX);

        // Cleanup
        allocator->init->disp.destroyFence(fence, nullptr);
    }
};

template<typename PushConst>
struct SquentialGpuTaskNoDesc {
    // Vulkan handles:
    VkPipelineLayout pipeline_layout;
    VkPipeline compute_pipeline;
    Allocator* allocator;

	PushConst push_const; // the push constants that will be used by the compute shader

	std::vector<char> spv_code = std::vector<char>{}; // we'll have shader bytecode in this

    SquentialGpuTaskNoDesc(std::string& spvByteCodeCompShader, Allocator* allocator, PushConst push_const) : push_const(push_const), allocator(allocator), spv_code(spvByteCodeCompShader.begin(), spvByteCodeCompShader.end()) {
		create_pipeline();
	};

    void onSignal() {
        // this is called when the memPool broadcasts a signal
        std::cout << "Signal Called\n";
    }

    void create_pipeline() {
        VkShaderModuleCreateInfo create_info = {};
        create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        create_info.codeSize = spv_code.size();
        create_info.pCode = reinterpret_cast<const uint32_t*>(spv_code.data());
        VkShaderModule shader_module;
        allocator->init->disp.createShaderModule(&create_info, nullptr, &shader_module);
        VkPipelineShaderStageCreateInfo shader_stage_info = {};
        shader_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shader_stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        shader_stage_info.module = shader_module;
        shader_stage_info.pName = "main";
        VkPipelineLayoutCreateInfo pipeline_layout_info = {};
        pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout_info.setLayoutCount = 0; // no descriptor sets
        pipeline_layout_info.pushConstantRangeCount = 1;
        pipeline_layout_info.pPushConstantRanges = &PushConst::getPushConstantRange();
        allocator->init->disp.createPipelineLayout(&pipeline_layout_info, nullptr, &pipeline_layout);
        VkComputePipelineCreateInfo pipeline_info = {};
        pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipeline_info.stage = shader_stage_info;
        pipeline_info.layout = pipeline_layout;
        pipeline_info.basePipelineHandle = VK_NULL_HANDLE;
        allocator->init->disp.createComputePipelines(VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &compute_pipeline);
        allocator->init->disp.destroyShaderModule(shader_module, nullptr);
    }

	void execute(VkCommandBuffer& command_buffer, uint32_t workgroup_x, uint32_t workgroup_y = 1, uint32_t workgroup_z = 1) {
		// Bind compute pipeline and push constants
		allocator->init->disp.cmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute_pipeline);
		allocator->init->disp.cmdPushConstants(command_buffer, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConst), &push_const);
		// Dispatch compute shader
		allocator->init->disp.cmdDispatch(command_buffer, workgroup_x, workgroup_y, workgroup_z);
	}
};

template<typename UniformBuffer, typename PushConst>
struct ComputeSequenceBufferReference {
    Allocator* allocator;
    std::unique_ptr<StandaloneBuffer<UniformBuffer>> uniform; // the buffer that will carry the device addresses of the actual buffers that're used

	std::vector<std::unique_ptr<SquentialGpuTaskNoDesc<PushConst>>> tasks; // the tasks that will be executed sequentially

    VkCommandPool pool;
    VkQueue queue;
    // multiple cmd bufs because simulation can occur multiple times per frame
	std::vector<VkCommandBuffer> commandBuffers; // the command buffers that will be used to execute the tasks

	ComputeSequenceBufferReference(Allocator* allocator) : allocator(allocator), uniform(std::make_unique<StandaloneBuffer<UniformBuffer>>(1, allocator)) {}

    void addTask(const std::string& shaderCode, PushConst& pushConst) {
		auto task = std::make_unique<SquentialGpuTaskNoDesc<PushConst>>(shaderCode, allocator, pushConst);
		tasks.push_back(std::move(task));
    }

    void createCommandBuffers(const uint32_t MAX_FRAMES) {
        // Get compute queue
        auto gq = allocator->init->device.get_queue(vkb::QueueType::compute);
        if (!gq.has_value()) {
            std::cout << "failed to get queue: " << gq.error().message() << "\n";
            return;
        }
        queue = gq.value();
        // Create command pool
        VkCommandPoolCreateInfo pool_info = {};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.queueFamilyIndex = allocator->init->device.get_queue_index(vkb::QueueType::compute).value();
        allocator->init->disp.createCommandPool(&pool_info, nullptr, &pool);
        // Allocate command buffers
        VkCommandBufferAllocateInfo alloc_info = {};
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.commandPool = pool;
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandBufferCount = static_cast<uint32_t>(MAX_FRAMES);
        commandBuffers.resize(MAX_FRAMES);
        allocator->init->disp.allocateCommandBuffers(&alloc_info, commandBuffers.data());
    }

    void recordCmds(const uint32_t MAX_FRAMES, uint32_t workgroup_x, uint32_t workgroup_y = 1, uint32_t workgroup_z = 1) {
		createCommandBuffers(MAX_FRAMES);
    
		// Record commands for each task
		for (size_t i = 0; i < commandBuffers.size(); i++) {
			VkCommandBufferBeginInfo begin_info = {};
			begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
			allocator->init->disp.beginCommandBuffer(commandBuffers[i], &begin_info);
            for (size_t j = 0; j < tasks.size(); j++) {
                tasks[j]->execute(commandBuffers[i], workgroup_x, workgroup_y, workgroup_z);
            }
			allocator->init->disp.endCommandBuffer(commandBuffers[i]);
		}
    }

	void executeTasks(const uint32_t iterations) {
        for (uint32_t iter = 0; iter < iterations; iter++) {
            for (uint32_t i = 0; i < commandBuffers.size(); i++) {
                VkSubmitInfo submit_info = {};
                submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                submit_info.commandBufferCount = 1;
                submit_info.pCommandBuffers = &commandBuffers[i];
                allocator->init->disp.queueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE);
            }
            allocator->init->disp.queueWaitIdle(queue);
        }
	}

};
