/*
This is a tiny graphics engine built using vulkan.
*/

#pragma once  
#define GLM_FORCE_RADIANS  
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#define GLFW_INCLUDE_VULKAN
#define TINYOBJLOADER_IMPLEMENTATION
constexpr int MAX_FRAMES_IN_FLIGHT = 3;

#include <glm/glm.hpp>  
#include <glm/gtc/matrix_transform.hpp>  
#include <glm/gtc/type_ptr.hpp>  
#include <glm/gtx/hash.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>
#include "volk.h"
#include <GLFW/glfw3.h>  
#include "VkBootstrap.h"  
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <unordered_map>
#include "VkMemAlloc.hpp"
#include "tiny_obj_loader.h"

int calcium_device_initialization(Init* init, GLFWwindow* window, VkSurfaceKHR& surface) {  
    if(volkInitialize() != VK_SUCCESS){
        throw std::runtime_error("Volk couldn't load the vulkan loader from system. It may be missing.");
    }
    vkb::InstanceBuilder instance_builder;  
    auto instance_ret = instance_builder  
        .use_default_debug_messenger()  
        .request_validation_layers()  
        .require_api_version(VK_API_VERSION_1_3) // Important!  
        .build();  
    if (!instance_ret) {  
        std::cout << instance_ret.error().message() << "\n";  
        return -1;  
    }  
    init->instance = instance_ret.value();  
    volkLoadInstance(init->instance.instance);

    init->inst_disp = init->instance.make_table();  

    glfwCreateWindowSurface(init->instance.instance, window, nullptr, &surface);  

    // Enable required features for bindless textures, buffer device address, shader objects, and ray tracing  
    VkPhysicalDeviceDescriptorIndexingFeaturesEXT descriptor_indexing_features = {};  
    descriptor_indexing_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES_EXT;  
    descriptor_indexing_features.descriptorBindingPartiallyBound = VK_TRUE;  
    descriptor_indexing_features.descriptorBindingVariableDescriptorCount = VK_TRUE;  
    descriptor_indexing_features.runtimeDescriptorArray = VK_TRUE;  
    descriptor_indexing_features.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;  

    VkPhysicalDeviceBufferDeviceAddressFeatures buffer_device_address_features = {};  
    buffer_device_address_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;  
    buffer_device_address_features.bufferDeviceAddress = VK_TRUE;  

    VkPhysicalDeviceShaderObjectFeaturesEXT shader_object_features = {};  
    shader_object_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT;  
    shader_object_features.shaderObject = VK_TRUE;  

    VkPhysicalDeviceRayTracingPipelineFeaturesKHR ray_tracing_pipeline_features = {};  
    ray_tracing_pipeline_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;  
    ray_tracing_pipeline_features.rayTracingPipeline = VK_TRUE;  

    VkPhysicalDeviceAccelerationStructureFeaturesKHR acceleration_structure_features = {};  
    acceleration_structure_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;  
    acceleration_structure_features.accelerationStructure = VK_TRUE;  

    VkPhysicalDeviceRayQueryFeaturesKHR ray_query_features = {};  
    ray_query_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;  
    ray_query_features.rayQuery = VK_TRUE;

    std::cout << "Selecting physical device...\n";
    vkb::PhysicalDeviceSelector phys_device_selector(init->instance);  
    auto phys_device_ret = phys_device_selector  
        .set_surface(surface)  
        .add_required_extension(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME)  
        .add_required_extension(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME)  
        .add_required_extension(VK_EXT_SHADER_OBJECT_EXTENSION_NAME)  
        .add_required_extension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME)  
        .add_required_extension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME)  
        .add_required_extension(VK_KHR_RAY_QUERY_EXTENSION_NAME)  
        .add_required_extension(VK_EXT_PIPELINE_CREATION_FEEDBACK_EXTENSION_NAME)  
        .add_required_extension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME)
        .add_required_extension_features(descriptor_indexing_features)  
        .add_required_extension_features(buffer_device_address_features)  
        .add_required_extension_features(shader_object_features)  
        .add_required_extension_features(ray_tracing_pipeline_features)  
        .select();  

    if (!phys_device_ret) {  
        std::cout << phys_device_ret.error().message() << "\n";  
        return -1;  
    }  
    if (!phys_device_ret) {
        std::cout << "Physical device selection failed: " << phys_device_ret.error().message() << "\n";
        return -1;
    } else {
        std::cout << "Physical device selected successfully.\n";
    }

    vkb::PhysicalDevice physical_device = phys_device_ret.value();  

    vkb::DeviceBuilder device_builder{ physical_device };  

    auto device_ret = device_builder.build();  
    if (!device_ret) {  
        std::cout << "Device creation failed: " << device_ret.error().message() << "\n";  
        return -1;  
    } else {
        std::cout << "Logical device created successfully.\n";
    }
    init->device = device_ret.value();  
    volkLoadDevice(init->device.device);
    init->disp = init->device.make_table();  

    return 0;  
}

// Read shader bytecode from a file
#ifndef readShaderCode
#define readShaderCode
std::vector<char> readShaderBytecode(const std::string& filename) {
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
#endif // !readShaderCode

struct VertexInputDesc {
    std::vector<VkVertexInputBindingDescription> bindings;
    std::vector<VkVertexInputAttributeDescription> attributes;

    VkPipelineVertexInputStateCreateFlags flags = 0;
};

struct Vertex {
    int positionIndex;
    int texCoordsIndex;
    int normalIndex;
    glm::vec3 tangent;
    glm::vec3 bitangent;
	int materialIndex;

    bool operator==(const Vertex& other) {
        return other.positionIndex == positionIndex && other.normalIndex == normalIndex && other.tangent == tangent && other.bitangent == bitangent && other.materialIndex == materialIndex;
    }

    static VertexInputDesc getVertexDesc() {
        VertexInputDesc desc;

        desc.bindings.push_back({
            0,                          // binding
            sizeof(Vertex),            // stride
            VK_VERTEX_INPUT_RATE_VERTEX // inputRate
            });

        desc.attributes = {
            {0, 0, VK_FORMAT_R32_SINT, offsetof(Vertex, positionIndex)},
            {1, 0, VK_FORMAT_R32_SINT, offsetof(Vertex, texCoordsIndex)},
            {2, 0, VK_FORMAT_R32_SINT, offsetof(Vertex, normalIndex)},
            {3, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, tangent)},
            {4, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, bitangent)},
			{5, 0, VK_FORMAT_R32_SINT, offsetof(Vertex, materialIndex)}
        };

        return desc;
    }
};

// Custom hash function for Vertex
namespace std {
    template <>
    struct hash<Vertex> {
        size_t operator()(const Vertex& vertex) const {
            size_t seed = 0;
            hash<float> hasher;

            // Has positionIndex
			seed ^= hasher(vertex.positionIndex) + 0x9e3779b9 + (seed << 6) + (seed >> 2);

			// Hash texCoordsIndex
			seed ^= hasher(vertex.texCoordsIndex) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			
            // Hash normalIndex
			seed ^= hasher(vertex.normalIndex) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			
            // Hash tangent
			seed ^= hasher(vertex.tangent.x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			seed ^= hasher(vertex.tangent.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			seed ^= hasher(vertex.tangent.z) + 0x9e3779b9 + (seed << 6) + (seed >> 2);

			// Hash bitangent
			seed ^= hasher(vertex.bitangent.x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			seed ^= hasher(vertex.bitangent.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			seed ^= hasher(vertex.bitangent.z) + 0x9e3779b9 + (seed << 6) + (seed >> 2);

			// Hash materialIndex
			seed ^= hasher(vertex.materialIndex) + 0x9e3779b9 + (seed << 6) + (seed >> 2);

			// Hash positionIndex
			seed ^= hasher(vertex.positionIndex) + 0x9e3779b9 + (seed << 6) + (seed >> 2);

            return seed;
        }
    };
}

struct Edge {
    Vertex* v1;
    Vertex* v2;

    Edge(Vertex* a, Vertex* b) {
        if (a < b) { v1 = a; v2 = b; }
        else { v1 = b; v2 = a; }
    }

    bool operator==(const Edge& other) const {
        return v1 == other.v1 && v2 == other.v2;
    }
};

struct Triangle {
    Vertex* v1;
    Vertex* v2;
    Vertex* v3;

    Triangle(Vertex* a, Vertex* b, Vertex* c) : v1(a), v2(b), v3(c) {}
};

// Custom hash function for Edge
namespace std {
    template <>
    struct hash<Edge> {
        size_t operator()(const Edge& edge) const {
            size_t seed = 0;
            hash<Vertex*> hasher;

            // Combine the hashes of both vertices (order-independent)
            seed ^= hasher(edge.v1) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= hasher(edge.v2) + 0x9e3779b9 + (seed << 6) + (seed >> 2);

            return seed;
        }
    };
}

// Custom hash function for Triangle
namespace std {
    template <>
    struct hash<Triangle> {
        size_t operator()(const Triangle& triangle) const {
            size_t seed = 0;
            hash<Vertex*> hasher;

            // Sort the pointers first (ensures order-independent hashing)
            std::array<Vertex*, 3> sortedVertices = { triangle.v1, triangle.v2, triangle.v3 };
            std::sort(sortedVertices.begin(), sortedVertices.end());

            // Combine hashes of all three vertices
            for (auto* v : sortedVertices) {
                seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }

            return seed;
        }
    };
}

// Define Edge comparison function
struct EdgeEqual {
    bool operator()(const Edge& e1, const Edge& e2) const {
        return (e1.v1 == e2.v1 && e1.v2 == e2.v2);
    }
};

// Define Triangle comparison function
struct TriangleEqual {
    bool operator()(const Triangle& t1, const Triangle& t2) const {
        std::array<Vertex*, 3> sortedT1 = { t1.v1, t1.v2, t1.v3 };
        std::array<Vertex*, 3> sortedT2 = { t2.v1, t2.v2, t2.v3 };

        std::sort(sortedT1.begin(), sortedT1.end());
        std::sort(sortedT2.begin(), sortedT2.end());

        return sortedT1 == sortedT2;
    }
};

struct PushConst {
    glm::mat4 model;
    VkDeviceAddress uniformBuf;
    int materialIndex;
    int positionOffset;
    int uvOffset;
    int normalOffset;
	int indexOffset;
	int frameIndex = 0;
    int numLights = 0;
};

struct DepthPushConst{
	glm::mat4 model;
	VkDeviceAddress posBuf;
    VkDeviceAddress indexBuf;
	int positionOffset;
	int indexOffset;
};

// this allows easy subpass chaining
struct RenderSubpass {
    std::vector<FBAttachment*> colorAttachments;
    std::vector<FBAttachment*> inputAttachments;
    FBAttachment* depthAttachment;

    // Backing storage for VkAttachmentReferences (must persist)
    std::vector<VkAttachmentReference> inputAttachmentRefs{};
    std::vector<VkAttachmentReference> colorAttachmentRefs{};

    VkSubpassDescription subpassDescription{};
    VkSubpassDependency subpassDependency{};

    bool hasDepthStencil = false;

    // subpasses are a doubly linked list
    RenderSubpass* pNext = nullptr;
    RenderSubpass* pPrevious = nullptr;
    int idx = 0; // the index of this subpass in the linked list

    RenderSubpass() {};

    RenderSubpass(VkAccessFlags accessMask, VkPipelineStageFlags stageFlags, RenderSubpass* pPrevious) : pPrevious(pPrevious) {
        if (pPrevious == nullptr) {
            idx = 0;
            subpassDependency.srcSubpass = VK_SUBPASS_EXTERNAL;
            subpassDependency.dstSubpass = 0;
            subpassDependency.srcAccessMask = 0;
            subpassDependency.dstAccessMask = accessMask;
            subpassDependency.srcStageMask = stageFlags;
            subpassDependency.dstStageMask = stageFlags;
            subpassDependency.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT; // this is the default, but you can change it if needed
        }
        else if (pPrevious != nullptr) {
            pPrevious->pNext = this;
            idx = pPrevious->idx + 1;
            pPrevious->subpassDependency.dstSubpass = idx;
            subpassDependency.srcSubpass = idx - 1;
            subpassDependency.dstSubpass = idx;
            subpassDependency.srcAccessMask = pPrevious->subpassDependency.dstAccessMask;
            subpassDependency.dstAccessMask = accessMask;
            subpassDependency.srcStageMask = pPrevious->subpassDependency.dstStageMask;
            subpassDependency.dstStageMask = stageFlags;
            subpassDependency.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT; // this is the default, but you can change it if needed
        }
    };

    void addInputAttachment(FBAttachment* attachment) {
        inputAttachments.push_back(attachment);
        inputAttachmentRefs.resize(inputAttachments.size());
        for (size_t i = 0; i < inputAttachments.size(); ++i) {
            inputAttachmentRefs[i] = inputAttachments[i]->attachmentReference;
        }
        subpassDescription.inputAttachmentCount = inputAttachmentRefs.size();
        subpassDescription.pInputAttachments = inputAttachmentRefs.data();
    }

    void addColorAttachment(FBAttachment* attachment) {
        colorAttachments.push_back(attachment);
        colorAttachmentRefs.resize(colorAttachments.size());
        for (size_t i = 0; i < colorAttachments.size(); ++i) {
            colorAttachmentRefs[i] = colorAttachments[i]->attachmentReference;
        }
        subpassDescription.colorAttachmentCount = colorAttachmentRefs.size();
        subpassDescription.pColorAttachments = colorAttachmentRefs.data();
    }

    void addDepthStencilAttachment(FBAttachment* attachment) {
        depthAttachment = attachment;
        subpassDescription.pDepthStencilAttachment = &depthAttachment->attachmentReference;
    }

    std::vector<VkAttachmentReference> getColorAttachmentReferences() {
        std::vector<VkAttachmentReference> refs;
        for (auto& attachment : colorAttachments) {
            refs.push_back(attachment->attachmentReference);
        }
        return refs;
    }

    std::vector<VkAttachmentReference> getInputAttachmentReferences() {
        std::vector<VkAttachmentReference> refs;
        for (auto& attachment : inputAttachments) {
            refs.push_back(attachment->attachmentReference);
        }
        return refs;
    }
};

struct UniformBuf {
    VkDeviceAddress positions;
    VkDeviceAddress normals;
    VkDeviceAddress uvs;
    VkDeviceAddress indices;
};

struct Swapchain {
    vkb::Swapchain swapchain;

    Init* init;

    int width, height;

    Swapchain() : init(nullptr) {};
    Swapchain(Init* init, int width, int height) :
        init(init), width(width), height(height) {
        createSwapchain();
    };

    void createSwapchain() {
        vkb::SwapchainBuilder builder{ init->device };

        VkSurfaceFormatKHR format{};
        format.format = VK_FORMAT_A2B10G10R10_UNORM_PACK32;
        format.colorSpace = VK_COLORSPACE_SRGB_NONLINEAR_KHR;

        VkPresentModeKHR presentMode = VK_PRESENT_MODE_MAILBOX_KHR;

        auto swap_ret = builder.set_desired_present_mode(presentMode)
            .set_desired_extent(width, height)
            .set_desired_format(format)
            .set_old_swapchain(swapchain)
            .build();

        if (!swap_ret) {
            std::cout << swap_ret.error().message() << " " << swap_ret.vk_result() << "\n";
        }

        swapchain = swap_ret.value();
    }
};

// pipeline can be chained to form a pipeline graph, where each pipeline can have its own render pass and framebuffer. The main graphics pipeline is the root of the graph.
// pipelines are stored as a single linked list, with the root pipeline pointing to the previous pipeline.
struct Pipeline {
    VkRenderPass renderPass = VK_NULL_HANDLE;
    RenderSubpass subpass;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;

    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;

    std::vector<VkPushConstantRange> pushConsts;
    std::unique_ptr<UniformBuffer<UniformBuf>> uniformBuf;
    std::vector<std::unique_ptr<Image2D>> images;
    std::vector<std::unique_ptr<ImageArray>> imageArrays;
    std::vector<std::unique_ptr<ImagePool>> imagePool; // imagePool is a scam. It doesn't exist. Need to remove it later.
    std::vector<VkDescriptorSetLayout> descriptorSetLayouts;

    std::unique_ptr<Framebuffer> framebuffer;

    Swapchain* swapchain;
    Allocator* allocator;

    std::unique_ptr<Pipeline> pPreviousPipeline;

    std::vector<VkShaderModule> shaders;
    std::vector<VkPipelineShaderStageCreateInfo> shader_stages;

    std::function<void()> createPipelineFunction;

    bool isComputeOnly = false;

    Pipeline(Swapchain* swapchain, Allocator* allocator, int width, int height) : swapchain(swapchain), allocator(allocator) {
        framebuffer = std::make_unique<Framebuffer>(width, height, 1, MAX_FRAMES_IN_FLIGHT, renderPass, allocator);
    }

    Pipeline(Swapchain* swapchain, Allocator* allocator) : swapchain(swapchain), allocator(allocator) {
        framebuffer = std::make_unique<Framebuffer>(swapchain->width, swapchain->height, 1, MAX_FRAMES_IN_FLIGHT, renderPass, allocator);
    }

	// no need to make framebuffers, this is for compute pipelines only
    Pipeline(Allocator* allocator) : allocator(allocator) {
		isComputeOnly = true;
    }

    Pipeline() {};

    // make sure the pipeline is destroyed before the allocator
    ~Pipeline() {
        if (renderPass != VK_NULL_HANDLE) {
            allocator->init->disp.destroyRenderPass(renderPass, nullptr);
        }
        if (pipeline != VK_NULL_HANDLE) {
            allocator->init->disp.destroyPipeline(pipeline, nullptr);
        }
        if (layout != VK_NULL_HANDLE) {
            allocator->init->disp.destroyPipelineLayout(layout, nullptr);
        }
        if (descriptorSetLayout != VK_NULL_HANDLE) {
            allocator->init->disp.destroyDescriptorSetLayout(descriptorSetLayout, nullptr);
        }
        if (descriptorPool != VK_NULL_HANDLE) {
            allocator->init->disp.destroyDescriptorPool(descriptorPool, nullptr);
        }
    }

    // initializes the pipeline by creating render pass, framebuffer, descriptor sets, pipeline layout, and pipeline
    virtual void initialize() {

        if (framebuffer->attachments[0].attachments.size() > 0) {
            createRenderPassFB();
            framebuffer->renderPass = renderPass;
            framebuffer->init();
        }
        else {
            createRenderPassNoFB();
        }
        if (images.size() > 0 || imagePool.size() > 0 || imageArrays.size() > 0) {
            createDescriptorPool();
            createDescriptorLayouts();
        }

        createPipelineLayout();
        createPipeline();
    }

    // no hand-holding here. You need to write this on your own.
    virtual void createPipeline() {
        createPipelineFunction();
    }

    // adds an image array object (see VkMemAlloc.hpp for implementation and docs) to the pipeline and updates descriptor sets
    void addImageArray(uint32_t maxImages) {
        std::unique_ptr<ImageArray> imageArray = std::make_unique<ImageArray>(maxImages, allocator);
        imageArray->updateDescriptorSets();
        imageArrays.push_back(std::move(imageArray));
    }

    // adds a push constant range to the pipeline layout. There are multiple push constant ranges allowed because muliple shaders may have different push constant requirements.
    void addPushConstant(VkDeviceSize range, VkDeviceSize offset, VkShaderStageFlags stage) {
        VkPushConstantRange constant = {};
        constant.offset = offset;
        constant.size = range;
        constant.stageFlags = stage;

        pushConsts.push_back(constant);
    }

    // adds a FBAttachment object to the framebuffer and subpass as a color attachment. See VkMemAlloc.hpp for FBAttachment implementation and docs.
    void addColorAttachment(std::shared_ptr<FBAttachment>& attachment) {
        subpass.addColorAttachment(attachment.get());
        framebuffer->addAttachment(attachment);
    }

    // adds a FBAttachment object to the framebuffer and subpass as a depth attachment. See VkMemAlloc.hpp for FBAttachment implementation and docs.
    void setDepthAttachment(std::shared_ptr<FBAttachment>& attachment) {
        subpass.addDepthStencilAttachment(attachment.get());
        framebuffer->addAttachment(attachment);
    }

    void createDescriptorPool() {
        VkDescriptorPoolSize poolSizes = {
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            static_cast<uint32_t>(images.size())
        };

        VkDescriptorPoolCreateInfo poolInfo = {};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSizes;
        allocator->init->disp.createDescriptorPool(&poolInfo, nullptr, &descriptorPool);
    }

    void createDescriptorLayouts() {
        std::vector<VkDescriptorSetLayoutBinding> bindings(images.size());
        for (size_t i = 0; i < images.size(); ++i) {
            images[i]->createDescriptors(i, VK_SHADER_STAGE_ALL);
            bindings[i] = images[i]->binding;
        }
        VkDescriptorSetLayoutCreateInfo dsl_info = {};
        dsl_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dsl_info.bindingCount = images.size();
        dsl_info.pBindings = bindings.data();
        allocator->init->disp.createDescriptorSetLayout(&dsl_info, nullptr, &descriptorSetLayout);

        VkDescriptorSetAllocateInfo ds_allocate_info = {};
        ds_allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ds_allocate_info.descriptorPool = descriptorPool;
        ds_allocate_info.descriptorSetCount = 1;
        ds_allocate_info.pSetLayouts = &descriptorSetLayout;
        allocator->init->disp.allocateDescriptorSets(&ds_allocate_info, &descriptorSet);

        std::vector<VkWriteDescriptorSet> writeSets(images.size());
        for (size_t i = 0; i < images.size(); i++) {
            images[i]->updateDescriptorSet(descriptorSet, 0);
            writeSets[i] = images[i]->wrt_desc_set;
        }

        allocator->init->disp.updateDescriptorSets(images.size(), writeSets.data(), 0, nullptr);

		for (auto& imageArray : imageArrays) {
			imageArray->updateDescriptorSets();
		}

        for (auto& pool : imagePool) {
            pool->updateDescriptorSets();
        }
    }

    // adds an Image2D object (see VkMemAlloc.hpp for implementation and docs) to the pipeline and updates descriptor sets
    void addImage(std::unique_ptr<Image2D>& image) {
		auto oldLayout = image->imageLayout;
        if (oldLayout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL && oldLayout != VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL) {
            allocator->transitionImageLayout(image->image, image->format, oldLayout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, image->mipLevels);
        }
        image->createDescriptors(images.size(), VK_SHADER_STAGE_ALL);
        image->updateDescriptorSet(descriptorSet, 0);
        images.push_back(std::move(image));
    }

    // no.
    void addImagePool(std::unique_ptr<ImagePool>& pool) {
        //imagePool.push_back(std::move(pool));
        std::cout << "Don't use this\n";
    }

    // if there is custom framebuffer, that means it'll output to that framebuffer, which is handled by the pipeline object itself
    virtual void createRenderPassFB() {
        std::vector<VkAttachmentDescription> attachments = framebuffer->getAttachmentDescriptions();

        VkRenderPassCreateInfo render_pass_info = {};
        render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        render_pass_info.attachmentCount = attachments.size();
        render_pass_info.pAttachments = attachments.data();
        render_pass_info.subpassCount = 1;
        render_pass_info.pSubpasses = &subpass.subpassDescription;
        render_pass_info.dependencyCount = 1;
        render_pass_info.pDependencies = &subpass.subpassDependency;
        render_pass_info.flags = 0;
		render_pass_info.pNext = nullptr;

        if (allocator->init->disp.createRenderPass(&render_pass_info, nullptr, &renderPass) != VK_SUCCESS) {
            std::cout << "failed to create render pass\n";
        }
    }

    // no frameBuffer attached means this is the final pipeline, beyond which the frames will be presented. It'll output to the swapchain's framebuffers, which is handled by the engine.
    virtual void createRenderPassNoFB() {
        VkAttachmentDescription color_attachment = {};
        color_attachment.format = swapchain->swapchain.image_format;
        color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
        color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference color_attachment_ref = {};
        color_attachment_ref.attachment = 0;
        color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass = {};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &color_attachment_ref;

        VkSubpassDependency dependency = {};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        VkRenderPassCreateInfo render_pass_info = {};
        render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        render_pass_info.attachmentCount = 1;
        render_pass_info.pAttachments = &color_attachment;
        render_pass_info.subpassCount = 1;
        render_pass_info.pSubpasses = &subpass;
        render_pass_info.dependencyCount = 1;
        render_pass_info.pDependencies = &dependency;

        if (allocator->init->disp.createRenderPass(&render_pass_info, nullptr, &renderPass) != VK_SUCCESS) {
            std::cout << "failed to create render pass\n";
        }
    }

    // creates pipeline layout based on available descriptor set layouts from images, image arrays, and image pools
    // no idea how i managed to write this monstrosity, but it works ¯\_(ツ)_/¯
    virtual void createPipelineLayout() {

        VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;

        if (images.size() > 0 && imageArrays.size() > 0 && imagePool.size() > 0) {
            descriptorSetLayouts.resize(imageArrays.size() + 1  + imagePool.size());
            descriptorSetLayouts[0] = descriptorSetLayout;
            for (size_t i = 0; i < imageArrays.size(); ++i) {
                descriptorSetLayouts[i + 1] = imageArrays[i]->descSetLayout;
            }
            for (size_t i = 0; i < imagePool.size(); i++) {
                descriptorSetLayouts[1 + imageArrays.size() + i] = imagePool[i]->layout;
            }
            pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(descriptorSetLayouts.size());
            pipelineLayoutInfo.pSetLayouts = descriptorSetLayouts.data();
        }
        else if (images.size() > 0 && imageArrays.size() > 0 && imagePool.size() == 0) {
			descriptorSetLayouts.resize(imageArrays.size() + 1);
			descriptorSetLayouts[0] = descriptorSetLayout;
			for (size_t i = 0; i < imageArrays.size(); ++i) {
				descriptorSetLayouts[i + 1] = imageArrays[i]->descSetLayout;
			}
			pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(descriptorSetLayouts.size());
			pipelineLayoutInfo.pSetLayouts = descriptorSetLayouts.data();
		}
        else if (images.size() == 0 && imageArrays.size() > 0 && imagePool.size() > 0) {
            descriptorSetLayouts.resize(imageArrays.size() + imagePool.size());
            for (size_t i = 0; i < imageArrays.size(); ++i) {
                descriptorSetLayouts[i] = imageArrays[i]->descSetLayout;
            }
            for (size_t i = 0; i < imagePool.size(); i++) {
                descriptorSetLayouts[imageArrays.size() + i] = imagePool[i]->layout;
            }
            pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(descriptorSetLayouts.size());
            pipelineLayoutInfo.pSetLayouts = descriptorSetLayouts.data();
        }
        else if (images.size() > 0 && imageArrays.size() == 0 && imagePool.size() > 0) {
            descriptorSetLayouts.resize(imagePool.size() + 1);
            descriptorSetLayouts[0] = descriptorSetLayout;
            for (size_t i = 0; i < imagePool.size(); i++) {
                descriptorSetLayouts[1 + i] = imagePool[i]->layout;
            }
            pipelineLayoutInfo.setLayoutCount = descriptorSetLayouts.size();
            pipelineLayoutInfo.pSetLayouts = descriptorSetLayouts.data();
        }
        else if (images.size() == 0 && imageArrays.size() == 0 && imagePool.size() > 0) {
            descriptorSetLayouts.resize(imagePool.size());
            for (size_t i = 0; i < imagePool.size(); i++) {
                descriptorSetLayouts[i] = imagePool[i]->layout;
            }
        }
        else {
            pipelineLayoutInfo.setLayoutCount = 0;
            pipelineLayoutInfo.pSetLayouts = nullptr;
        }

        pipelineLayoutInfo.pushConstantRangeCount = pushConsts.size();
        pipelineLayoutInfo.pPushConstantRanges = pushConsts.data();

        if (allocator->init->disp.createPipelineLayout(&pipelineLayoutInfo, nullptr, &layout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!");
        }
    }

    std::vector<char> readShaderBytecode(const std::string& filename) {
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

    void addShaderModule(const std::string& path, VkShaderStageFlagBits stage) {
        auto code = readShaderBytecode(path);
        VkShaderModuleCreateInfo create_info = {};
        create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        create_info.codeSize = code.size();
        create_info.pCode = reinterpret_cast<const uint32_t*>(code.data());

        VkShaderModule shaderModule;
        if (allocator->init->disp.createShaderModule(&create_info, nullptr, &shaderModule) != VK_SUCCESS) {
            throw std::runtime_error("couldn't add shader module");
        }

        VkPipelineShaderStageCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        info.stage = stage;
        info.module = shaderModule;
        info.pName = "main";

        shaders.push_back(shaderModule);
        shader_stages.push_back(info);
    }

    void bindDescSets(VkCommandBuffer& cmd) const {
        if (images.size() > 0 && imageArrays.size() > 0) {
            std::vector<VkDescriptorSet> sets;
			sets.reserve(1 + imageArrays.size() + imagePool.size());
            sets.push_back(descriptorSet);
            for (auto& arr : imageArrays) sets.push_back(arr->descSet);
            for (auto& img : imagePool) sets.push_back(img->set);
            allocator->init->disp.cmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, static_cast<uint32_t>(sets.size()), sets.data(), 0, nullptr);
        }
        else if (images.size() == 0 && imageArrays.size() > 0) {
            std::vector<VkDescriptorSet> sets;
			sets.reserve(imageArrays.size() + imagePool.size());
			for (auto& arr : imageArrays) sets.push_back(arr->descSet);
			for (auto& img : imagePool) sets.push_back(img->set);
			allocator->init->disp.cmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, static_cast<uint32_t>(sets.size()), sets.data(), 0, nullptr);
		}
        else if (images.size() == 0 && imageArrays.size() == 0 && imagePool.size() > 0) {
			std::vector<VkDescriptorSet> sets;
			sets.reserve(imagePool.size());
			for (auto& img : imagePool) sets.push_back(img->set);
            allocator->init->disp.cmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, static_cast<uint32_t>(sets.size()), sets.data(), 0, nullptr);
        }
        else if (images.size() > 0 && imageArrays.size() == 0 && imagePool.size() > 0) {
			std::vector<VkDescriptorSet> sets;
			sets.reserve(1 + imagePool.size());
			sets.push_back(descriptorSet);
			for (auto& img : imagePool) sets.push_back(img->set);
			allocator->init->disp.cmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, static_cast<uint32_t>(sets.size()), sets.data(), 0, nullptr);
		}
		else if (images.size() > 0 && imageArrays.size() == 0 && imagePool.size() == 0) {
			allocator->init->disp.cmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &descriptorSet, 0, nullptr);
		}
		else {
			allocator->init->disp.cmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 0, nullptr, 0, nullptr);
        }
    }

    // renders the pipeline by beginning the render pass, executing the secondary command buffers, and ending the render pass
    void render(VkCommandBuffer& cmd, VkCommandBuffer& renderCmds, VkClearValue& clearValue, int frameIndex) const {

        VkRenderPassBeginInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        info.renderPass = renderPass;
        info.framebuffer = framebuffer->framebuffers[frameIndex];
        info.renderArea.offset = { 0, 0 };
        info.renderArea.extent = swapchain->swapchain.extent;
        info.clearValueCount = 1;
        info.pClearValues = &clearValue;

        allocator->init->disp.cmdBeginRenderPass(cmd, &info, VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS);
        allocator->init->disp.cmdExecuteCommands(cmd, 1, &renderCmds);
        allocator->init->disp.cmdEndRenderPass(cmd);
    }

    // renders the pipeline by beginning the render pass, executing the secondary command buffers, and ending the render pass
    void render(VkCommandBuffer& cmd, VkCommandBuffer& renderCmds, VkRenderPassBeginInfo& info) const {
        allocator->init->disp.cmdBeginRenderPass(cmd, &info, VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS);
        allocator->init->disp.cmdExecuteCommands(cmd, 1, &renderCmds);
        allocator->init->disp.cmdEndRenderPass(cmd);
    }
};

// a simple depth-only pipeline for shadow mapping / depth pre-pass, etc.
struct DepthPipeline : public Pipeline {

    DepthPipeline(Swapchain* swapchain, Allocator* allocator) : Pipeline(swapchain, allocator) {};

    DepthPipeline(const std::string& vertexPath, const std::string& fragmentPath, Swapchain* swapchain, Allocator* allocator) : Pipeline(swapchain, allocator) {
		addShaderModule(vertexPath, VK_SHADER_STAGE_VERTEX_BIT);
		addShaderModule(fragmentPath, VK_SHADER_STAGE_FRAGMENT_BIT);
		addPushConstant(sizeof(DepthPushConst), 0, VK_SHADER_STAGE_VERTEX_BIT);

        subpass = RenderSubpass(VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, nullptr);

        auto depthAttachment = std::make_shared<FBAttachment>(swapchain->width, swapchain->height, 0, VK_FORMAT_D32_SFLOAT,
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, allocator);

		setDepthAttachment(depthAttachment);

        initialize();
	}

    void createPipeline() override {

        VkGraphicsPipelineCreateInfo pipelineInfo = {};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = shaders.size();

        // No vertex input state needed because we use a custom vertex processor
        VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 0; // No bindings
        vertexInputInfo.pVertexBindingDescriptions = nullptr; // No binding descriptions
        vertexInputInfo.vertexAttributeDescriptionCount = 0; // No attributes
        vertexInputInfo.pVertexAttributeDescriptions = nullptr; // No attribute descriptions

        pipelineInfo.pVertexInputState = &vertexInputInfo;

        // Input Assembly
        VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;
        pipelineInfo.pInputAssemblyState = &inputAssembly;

        VkViewport viewport = {};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float)swapchain->width;
        viewport.height = (float)swapchain->height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        VkRect2D scissor = {};
        scissor.offset = { 0, 0 };
        scissor.extent = swapchain->swapchain.extent;

        VkPipelineViewportStateCreateInfo viewport_state = {};
        viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewport_state.viewportCount = 1;
        viewport_state.pViewports = &viewport;
        viewport_state.scissorCount = 1;
        viewport_state.pScissors = &scissor;

        VkPipelineRasterizationStateCreateInfo rasterizer = {};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;

        VkPipelineMultisampleStateCreateInfo multisampling = {};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
        colorBlendAttachment.colorWriteMask = 0; // Disable all color writes
        colorBlendAttachment.blendEnable = VK_FALSE; // No blending needed

        VkPipelineColorBlendStateCreateInfo color_blending = {};
        color_blending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        color_blending.logicOpEnable = VK_FALSE; // Logic operations not needed
        color_blending.logicOp = VK_LOGIC_OP_COPY; // Default logic operation
        color_blending.attachmentCount = 1; // One attachment
        color_blending.pAttachments = &colorBlendAttachment;
        color_blending.blendConstants[0] = 0.0f;
        color_blending.blendConstants[1] = 0.0f;
        color_blending.blendConstants[2] = 0.0f;
        color_blending.blendConstants[3] = 0.0f;

        std::vector<VkDynamicState> dynamic_states = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };

        VkPipelineDynamicStateCreateInfo dynamic_info = {};
        dynamic_info.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamic_info.dynamicStateCount = static_cast<uint32_t>(dynamic_states.size());
        dynamic_info.pDynamicStates = dynamic_states.data();

        VkPipelineDepthStencilStateCreateInfo depthStencil = {};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_TRUE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_GREATER; // REVERSE-Z: greater is closer
        depthStencil.depthBoundsTestEnable = VK_FALSE;
        depthStencil.minDepthBounds = 0.0f;
        depthStencil.maxDepthBounds = 1.0f;
        depthStencil.stencilTestEnable = VK_FALSE;
        depthStencil.front = {};
        depthStencil.back = {};

        VkGraphicsPipelineCreateInfo pipeline_info = {};
        pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipeline_info.stageCount = shader_stages.size();
        pipeline_info.pStages = shader_stages.data();
        pipeline_info.pVertexInputState = &vertexInputInfo;
        pipeline_info.pInputAssemblyState = &inputAssembly;
        pipeline_info.pViewportState = &viewport_state;
        pipeline_info.pRasterizationState = &rasterizer;
        pipeline_info.pMultisampleState = &multisampling;
        pipeline_info.pColorBlendState = &color_blending;
        pipeline_info.pDynamicState = &dynamic_info;
		pipeline_info.pDepthStencilState = &depthStencil;
        pipeline_info.layout = layout;
        pipeline_info.renderPass = renderPass;
        pipeline_info.subpass = 0;
        pipeline_info.basePipelineHandle = VK_NULL_HANDLE;

		if (allocator->init->disp.createGraphicsPipelines(VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline) != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline");
		}

        for (auto & shader : shaders) {
            allocator->init->disp.destroyShaderModule(shader, nullptr);
        }
    }
};

// main graphics pipeline that uses the depth pipeline as a previous pipeline to read depth images from
// root of the pipeline graph
struct GraphicsPipeline : public Pipeline {

    GraphicsPipeline(const std::string& vertexPath, const std::string& fragmentPath, Swapchain* swapchain, Allocator* allocator) :Pipeline(swapchain, allocator) {

        addShaderModule(vertexPath, VK_SHADER_STAGE_VERTEX_BIT);
        addShaderModule(fragmentPath, VK_SHADER_STAGE_FRAGMENT_BIT);
        addPushConstant(sizeof(PushConst), 0, VK_SHADER_STAGE_VERTEX_BIT);

        addImageArray(100);
        
        std::unique_ptr<Pipeline> depthPipeline = std::make_unique<DepthPipeline>("compiled_shaders/depth.vert.spv", "compiled_shaders/depth.frag.spv", swapchain, allocator);
        pPreviousPipeline = std::move(depthPipeline);

		addImage(pPreviousPipeline->framebuffer->attachments[0].attachments[0]->image); // add depth image to this pipeline (in this case, it's set = 0, binding = 0)
		addImage(pPreviousPipeline->framebuffer->attachments[1].attachments[0]->image); // add depth image number 2 to this pipeline (in this case, it's set = 0, binding = 1)
		addImage(pPreviousPipeline->framebuffer->attachments[2].attachments[0]->image); // add depth image number 3 to this pipeline (in this case, it's set = 0, binding = 2)
        
        initialize();
    }

    GraphicsPipeline(Init* init, Swapchain* swapchain, Allocator* allocator) : Pipeline(swapchain, allocator) {}

    GraphicsPipeline() {};

	void createPipeline() override {
		VkGraphicsPipelineCreateInfo pipelineInfo = {};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineInfo.stageCount = shaders.size();

		std::vector<VkPipelineShaderStageCreateInfo> shaderStages(shaders.size());
		for (size_t i = 0; i < shaders.size(); ++i) {
			shaderStages[i] = shader_stages[i];
		}
		pipelineInfo.pStages = shaderStages.data();

        // No vertex input state needed
        VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 0; // No bindings
        vertexInputInfo.pVertexBindingDescriptions = nullptr; // No binding descriptions
        vertexInputInfo.vertexAttributeDescriptionCount = 0; // No attributes
        vertexInputInfo.pVertexAttributeDescriptions = nullptr; // No attribute descriptions

        pipelineInfo.pVertexInputState = &vertexInputInfo;
        
		// Input Assembly
		VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
		inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		inputAssembly.primitiveRestartEnable = VK_FALSE;
		pipelineInfo.pInputAssemblyState = &inputAssembly;
		
        VkViewport viewport = {};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float)swapchain->width;
        viewport.height = (float)swapchain->height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        VkRect2D scissor = {};
        scissor.offset = { 0, 0 };
        scissor.extent = swapchain->swapchain.extent;

        VkPipelineViewportStateCreateInfo viewport_state = {};
        viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewport_state.viewportCount = 1;
        viewport_state.pViewports = &viewport;
        viewport_state.scissorCount = 1;
        viewport_state.pScissors = &scissor;

        VkPipelineRasterizationStateCreateInfo rasterizer = {};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;

        VkPipelineMultisampleStateCreateInfo multisampling = {};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
        colorBlendAttachment.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo color_blending = {};
        color_blending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        color_blending.logicOpEnable = VK_FALSE;
        color_blending.logicOp = VK_LOGIC_OP_COPY;
        color_blending.attachmentCount = 1;
        color_blending.pAttachments = &colorBlendAttachment;
        color_blending.blendConstants[0] = 0.0f;
        color_blending.blendConstants[1] = 0.0f;
        color_blending.blendConstants[2] = 0.0f;
        color_blending.blendConstants[3] = 0.0f;

        std::vector<VkDynamicState> dynamic_states = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };

        VkPipelineDynamicStateCreateInfo dynamic_info = {};
        dynamic_info.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamic_info.dynamicStateCount = static_cast<uint32_t>(dynamic_states.size());
        dynamic_info.pDynamicStates = dynamic_states.data();

        VkGraphicsPipelineCreateInfo pipeline_info = {};
        pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipeline_info.stageCount = shaderStages.size();
        pipeline_info.pStages = shaderStages.data();
        pipeline_info.pVertexInputState = &vertexInputInfo;
        pipeline_info.pInputAssemblyState = &inputAssembly;
        pipeline_info.pViewportState = &viewport_state;
        pipeline_info.pRasterizationState = &rasterizer;
        pipeline_info.pMultisampleState = &multisampling;
        pipeline_info.pColorBlendState = &color_blending;
        pipeline_info.pDynamicState = &dynamic_info;
        pipeline_info.layout = layout;
        pipeline_info.renderPass = renderPass;
        pipeline_info.subpass = 0;
        pipeline_info.basePipelineHandle = VK_NULL_HANDLE;

        if (allocator->init->disp.createGraphicsPipelines(VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline) != VK_SUCCESS) {
            std::cout << "failed to create pipline\n";
        }

        for (auto& shader : shaders) {
            allocator->init->disp.destroyShaderModule(shader, nullptr);
        }
	}
};

struct Material {
    std::string texPath;

    Material(){};

    void loadTexture(const std::string& path) {
        texPath = path;
    }
};

struct PackedIndex {
    uint32_t pos;
    uint32_t norm;
    uint32_t uv;
    uint32_t _padding;
};

// resources for a 3D mesh
// transported onto the GPU as a single vertex buffer with packed indices later
struct Mesh {
    std::vector<Triangle> triangles;
    std::vector<glm::vec4> positions;
    std::vector<glm::vec2> uvs;
    std::vector<glm::vec4> normals;
	std::vector<PackedIndex> packedIndices;

    unsigned int indexCount;

    Material material;

    bool meshUploaded = false;

    Mesh() {};

    void loadModel(const std::string& objFilePath) {
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn, err;

        bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, objFilePath.c_str());

        if (!warn.empty()) std::cerr << "[WARN] " << warn << std::endl;
        if (!err.empty()) std::cerr << "[ERROR] " << err << std::endl;
        if (!ret) return;

        // Convert positions
        for (size_t i = 0; i < attrib.vertices.size(); i += 3) {
            glm::vec4 pos(attrib.vertices[i], attrib.vertices[i + 1], attrib.vertices[i + 2], 0.0f);
            positions.push_back(pos);
        }

        // Convert uvs
        for (size_t i = 0; i < attrib.texcoords.size(); i += 2) {
            glm::vec2 uv(attrib.texcoords[i], attrib.texcoords[i + 1]);
            uvs.push_back(uv);
        }

        // Convert normals
        for (size_t i = 0; i < attrib.normals.size(); i += 3) {
            glm::vec4 norm(attrib.normals[i], attrib.normals[i + 1], attrib.normals[i + 2], 0.0f);
            normals.push_back(norm);
        }

		// Process shapes
        for (const auto& shape : shapes) {
            for (const auto& index : shape.mesh.indices) {
                packedIndices.push_back({
                    static_cast<uint32_t>(index.vertex_index),
                    static_cast<uint32_t>(index.normal_index),
                    static_cast<uint32_t>(index.texcoord_index)
                    });
            }
        }
    }

    Mesh(const Mesh& other) {
		material = other.material;
        meshUploaded = other.meshUploaded;
    }

    ~Mesh() {}
};

struct PointLight {
	glm::vec3 position;
	glm::vec3 color;
	float intensity;
	PointLight(glm::vec3 position, glm::vec3 color, float intensity) : position(position), color(color), intensity(intensity) {}
};

enum Camera_Movement {
    FORWARD,
    BACKWARD,
    LEFT,
    RIGHT
};

class Camera {
    glm::vec3 Up;
    glm::vec3 Right;
    glm::vec3 WorldUp;
    float Yaw;
    float Pitch;

    float MovementSpeed;
    float MouseSensitivity;

public:

    glm::vec3 Front;
    glm::vec3 Position;
    float Zoom;
    Camera(glm::vec3 position, glm::vec3 up, float yaw, float pitch) : Front(glm::vec3(0.0f, 0.0f, -1.0f)), MovementSpeed(25.5f), MouseSensitivity(0.1f), Zoom(45.0f) {
        Position = position;
        WorldUp = up;
        Yaw = yaw;
        Pitch = pitch;
        updateCameraVectors();
    }

    Camera() {};

    glm::mat4 GetViewMatrix() const {
        return glm::lookAt(Position, Position + Front, Up);
    }

    glm::mat4 GetProjectionMatrix(float windowHeight, float windowWidth, float FOV) {
        return glm::perspective(glm::radians(FOV), windowWidth / windowHeight, 1000.0f, 1.0f);
    }

    glm::mat4 GetProjectionMatrixReverse(float windowHeight, float windowWidth, float FOV) {
        glm::mat4 projectionMatrix = glm::perspective(glm::radians(FOV), windowWidth / windowHeight, 1000.0f, 1.0f);
        return projectionMatrix;
    }

    glm::mat4 GetOrthogonalMatrix(float windowHeight, float windowWidth, float FOV) {
        float aspectRatio = windowWidth / windowHeight;
        float orthoHeight = FOV;
        float orthoWidth = FOV * aspectRatio;
        return glm::ortho<float>(-120, 120, -120, 120, -500, 500);
    }

    void ProcessKeyboard(int direction, float deltaTime) {
        float velocity = MovementSpeed * deltaTime;
        if (direction == FORWARD)
            Position += Front * velocity;
        if (direction == BACKWARD)
            Position -= Front * velocity;
        if (direction == LEFT)
            Position -= Right * velocity;
        if (direction == RIGHT)
            Position += Right * velocity;
    }

    void ProcessMouseMovement(float xoffset, float yoffset, GLboolean constrainPitch) {
        xoffset *= MouseSensitivity;
        yoffset *= MouseSensitivity;

        Yaw += xoffset;
        Pitch += yoffset;

        xoffset = 0.0f;
        yoffset = 0.0f;

        if (constrainPitch) {
            if (Pitch > 89.0f)
                Pitch = 89.0f;
            if (Pitch < -89.0f)
                Pitch = -89.0f;
        }

        updateCameraVectors();
    }

    void ProcessMouseScroll(float yoffset) {
        Zoom -= yoffset;
        if (Zoom < 1.0f)
            Zoom = 1.0f;
        if (Zoom > 45.0f)
            Zoom = 45.0f;
    }

    void updateCameraVectors() {
        glm::vec3 front;
        front.x = cos(glm::radians(Yaw)) * cos(glm::radians(Pitch));
        front.y = sin(glm::radians(Pitch));
        front.z = sin(glm::radians(Yaw)) * cos(glm::radians(Pitch));
        Front = glm::normalize(front);

        Right = glm::normalize(glm::cross(Front, WorldUp));
        Up = glm::normalize(glm::cross(Right, Front));
    }
};

struct Entity;

struct Transform {

    glm::vec3 position;
    glm::quat rotation;
    glm::vec3 scale;

    glm::mat4 model;

    Transform(glm::vec3 position, glm::quat rotation, glm::vec3 scale) : position(position), rotation(rotation), scale(scale) {}

    void updateGlobalTransform() {
        glm::mat4 localTransform = glm::mat4(1.0f);
        localTransform = glm::translate(localTransform, position);
        localTransform *= glm::toMat4(rotation);
        localTransform = glm::scale(localTransform, scale);
        model = localTransform;
    }

    glm::vec3 getPosition() const {
        return position;
    }
    glm::quat getRotation() const {
        return rotation;
    }
    glm::vec3 getScale() const {
        return scale;
    }

    void setPosition(glm::vec3 position) {
        this->position = position;
    }
    void setRotation(glm::quat rotation) {
        this->rotation = rotation;
    }
    void setScale(glm::vec3 scale) {
        this->scale = scale;
    }

    void translate(glm::vec3 translation) {
        position += translation;
    }
    void rotate(glm::quat rotation) {
        this->rotation = rotation * this->rotation;
    }
    void scaleBy(glm::vec3 scale) {
        this->scale *= scale;
    }
};

struct Entity {
    Transform transform;
    Mesh mesh;
    Init* init;

    Entity(glm::vec3 position, glm::quat rotation, glm::vec3 scale, bool useBuiltInTransform)
        : transform(position, rotation, scale) {
        //mesh.traingle();
    }

    void update() { transform.updateGlobalTransform(); }

private:
    
};

struct Scene {
    std::vector<Entity> entities;
	Camera camera;

    Allocator* allocator;

	// non duplicated unique vertex data
    // Vec4 instead of Vec3 because Vec4 is perfectly sized at 16 bytes, which means that std430 inside of shaders will work (because vec3's stride is 16 bytes).
	// Even though we're wasting 4 bytes per attribute, this is way faster than using Vec3, which is 12 bytes and is slow.
	// Buffer_reference_align is always set to 16 bytes, because that seems to be minimum vulkan alignment for me, according to vkGetPhysicalDeviceProperties
	// You can get the min alignment of ur device from Allocator->getAlignment()

    // see MemPool in VkMemAlloc.hpp for more info on how these work
    std::unique_ptr<MemPool<glm::vec4>> positionPool;
	std::unique_ptr<MemPool<glm::vec2>> uvPool;
	std::unique_ptr<MemPool<glm::vec4>> normalPool;

	// index data for all meshes. Avoids duplicated vertex attributes.
    // The number of elements inside of one mesh's index buffer is equal to the number of times the vertex shader will be called for that mesh
	std::unique_ptr<MemPool<PackedIndex>> packedIndexPool;

    std::unique_ptr<StandaloneBuffer<UniformBuf>> uniforms;

    GraphicsPipeline* pipeline;

    bool updateUnfiforms = true;

    Scene() {};

    Scene(Allocator* allocator, GraphicsPipeline* pipeline) : allocator(allocator), uniforms(std::make_unique<StandaloneBuffer<UniformBuf>>(1, allocator, VK_SHADER_STAGE_ALL)), pipeline(pipeline),
        positionPool(std::make_unique<MemPool<glm::vec4>>(1000, allocator, VK_SHADER_STAGE_VERTEX_BIT)),
        uvPool(std::make_unique<MemPool<glm::vec2>>(1000, allocator, VK_SHADER_STAGE_VERTEX_BIT)), 
        normalPool(std::make_unique<MemPool<glm::vec4>>(1000, allocator, VK_SHADER_STAGE_VERTEX_BIT)),
        packedIndexPool(std::make_unique<MemPool<PackedIndex>>(1000, allocator, VK_SHADER_STAGE_VERTEX_BIT)){
        
        positionPool->descUpdateQueued.addListener(this);
		uvPool->descUpdateQueued.addListener(this);
		normalPool->descUpdateQueued.addListener(this);
		packedIndexPool->descUpdateQueued.addListener(this);
		entities.reserve(20);
		camera = Camera(glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), 0.0f, 0.0f);
	};

    // must-have function for MemPool to communicate descriptor / buffer reference updates.
    void onSignal() {
		updateUnfiforms = true;
    }

	void addEntity(Entity& entity) {
		entities.push_back(entity);
        positionPool->push_back(entity.mesh.positions);
		uvPool->push_back(entity.mesh.uvs);
		normalPool->push_back(entity.mesh.normals);
		packedIndexPool->push_back(entity.mesh.packedIndices);
		pipeline->imageArrays[0]->push_back(entity.mesh.material.texPath);
		pipeline->imageArrays[0]->updateDescriptorSets();
	}

    void removeEntity(uint32_t idx, bool instaClean = true) {
		if (idx >= entities.size()) return;
		positionPool->erase(idx, instaClean);
		uvPool->erase(idx, instaClean);
		normalPool->erase(idx, instaClean);
		packedIndexPool->erase(idx, instaClean);
		pipeline->imageArrays[0]->erase(idx);
		pipeline->imageArrays[0]->updateDescriptorSets();
		entities.erase(entities.begin() + idx);
    }

    void defragment() {
        positionPool->cleanGaps();
		uvPool->cleanGaps();
		normalPool->cleanGaps();
		packedIndexPool->cleanGaps();
    }

    void renderScene(VkCommandBuffer& commandBuffer, int width, int height, int frameIndex) {
        
        if (updateUnfiforms) {
            UniformBuf uniformsData = {};
		    uniformsData.indices = packedIndexPool->getBufferAddress();
		    uniformsData.positions = positionPool->getBufferAddress();
		    uniformsData.uvs = uvPool->getBufferAddress();
		    uniformsData.normals = normalPool->getBufferAddress();
			std::vector<UniformBuf> uniformsDataVec(1, uniformsData);
            uniforms->alloc(uniformsDataVec);
            uniforms->getBufferAddress();
            updateUnfiforms = false;
        }
		
        PushConst pushConst = {};
		pushConst.uniformBuf = uniforms->getBufferAddress();
        pushConst.frameIndex = frameIndex;
        
        for (int i = 0; i < entities.size(); i++) {
			auto& entity = entities[i];
            entity.update();
            pushConst.model = camera.GetProjectionMatrix(height, width, 50.0f) * camera.GetViewMatrix() * entity.transform.model;
            pushConst.materialIndex = i;

            // element offsets represent the beginning of the current mesh's data in the shared memPools
            pushConst.positionOffset = positionPool->buffers[i].elementOffset;
			pushConst.uvOffset = uvPool->buffers[i].elementOffset;
			pushConst.normalOffset = normalPool->buffers[i].elementOffset;
			pushConst.indexOffset = packedIndexPool->buffers[i].elementOffset;

            allocator->init->disp.cmdPushConstants(commandBuffer, pipeline->layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConst), &pushConst);
            allocator->init->disp.cmdDraw(commandBuffer, packedIndexPool->buffers[i].numElements, 1, 0, 0);
        }
    };

    void renderSceneDepth(VkCommandBuffer& commandBuffer, int width, uint32_t height) {
        DepthPushConst pushConst = {};
		pushConst.indexBuf = packedIndexPool->getBufferAddress();
		pushConst.posBuf = positionPool->getBufferAddress();

        for (int i = 0; i < entities.size(); i++) {
            auto& entity = entities[i];
            entity.update();
            pushConst.model = camera.GetProjectionMatrixReverse(height, width, 50.0f) * camera.GetViewMatrix() * entity.transform.model;
            pushConst.indexOffset = packedIndexPool->buffers[i].elementOffset;
            pushConst.positionOffset = positionPool->buffers[i].elementOffset;

			allocator->init->disp.cmdPushConstants(commandBuffer, pipeline->pPreviousPipeline->layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(DepthPushConst), &pushConst);
			allocator->init->disp.cmdDraw(commandBuffer, packedIndexPool->buffers[i].numElements, 1, 0, 0);
        }
    }

    void renderDirectionalShadows(VkCommandBuffer& commandBuffer, int size) {
        DepthPushConst pushConst = {};
        pushConst.indexBuf = packedIndexPool->getBufferAddress();
        pushConst.posBuf = positionPool->getBufferAddress();
        glm::mat4 lightProj = glm::ortho<float>(-size, size, -size, size, 1000.0f, 0.1f);
        glm::vec3 lightPos = camera.Position - glm::normalize(glm::vec3(45.0f, 45.0f, 45.0f)) * 100.0f;
        glm::mat4 lightView = glm::lookAt(lightPos, lightPos + glm::normalize(glm::vec3(45.0f, 45.0f, 45.0f)), glm::vec3(0.0f, 1.0f, 0.0f));

        for (int i = 0; i < entities.size(); i++) {
            auto& entity = entities[i];
            entity.update();
            pushConst.model = lightProj * lightView * entity.transform.model;
            pushConst.indexOffset = packedIndexPool->buffers[i].elementOffset;
            pushConst.positionOffset = positionPool->buffers[i].elementOffset;
            allocator->init->disp.cmdPushConstants(commandBuffer, pipeline->pPreviousPipeline->layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(DepthPushConst), &pushConst);
            allocator->init->disp.cmdDraw(commandBuffer, packedIndexPool->buffers[i].numElements, 1, 0, 0);
        }
    }
};

// main engine structure
struct Engine {
     
    Init init;  

    GLFWwindow* window;

    VkSurfaceKHR surface;

    Allocator* allocator;

    VkQueue graphicsQueue;
    VkQueue presentQueue;

    Swapchain swapchain;
    GraphicsPipeline* pipeline;

    std::vector<VkImage> swapchainImages;
    std::vector<VkImageView> swapchainImageViews;
    std::vector<VkFramebuffer> framebuffers;

    VkCommandPool commandPool;
    std::vector<VkCommandBuffer> commandBuffers;
	std::vector<VkCommandBuffer> secondaryCmdBufs;
	std::vector<VkCommandBuffer> secondaryCmdBufsDepth;

    std::vector<VkSemaphore> availiables;
    std::vector<VkSemaphore> finishes;
    std::vector<VkFence> inFlights;
    std::vector<VkFence> imagesInFlights;

    std::function<void()> update;
    std::function<void()> start;

    size_t currentFrame = 0;

    int width, height;

    // only one active scene in all engines.
    static Scene scene;

    Engine(int width, int height, const std::string& vertexPath, const std::string& fragmentPath) : height(height), width(width) {  
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);  
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);  
        window = glfwCreateWindow(width, height, "Calcium", nullptr, nullptr);  
        if (!window) {  
            throw std::runtime_error("failed to create window");  
        }

        glfwSetCursorPosCallback(window, mouse_callback);
        glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

        calcium_device_initialization(&init, window, surface);

        allocator = new Allocator(&init);
        
        createSwapchain();
        get_queues();
        createCommandPool();
        
		allocator->graphicsPool = commandPool;
        allocator->graphicsQueue = graphicsQueue;

        createGraphicsPipeline(vertexPath, fragmentPath);
        
        scene = Scene(allocator, pipeline);

        createFramebuffers();
        createCommandBuffers();
        createSyncObjects();
    };

    ~Engine() {
        delete pipeline;
        
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            init.disp.destroySemaphore(finishes[i], nullptr);
            init.disp.destroySemaphore(availiables[i], nullptr);
            init.disp.destroyFence(inFlights[i], nullptr);
        }

        init.disp.destroyCommandPool(commandPool, nullptr);

        for (auto framebuffer : framebuffers) {
            init.disp.destroyFramebuffer(framebuffer, nullptr);
        }

        swapchain.swapchain.destroy_image_views(swapchainImageViews);
        delete allocator;

        vkb::destroy_swapchain(swapchain.swapchain);
		vkb::destroy_surface(init.instance, surface);
        vkb::destroy_device(init.device);
        vkb::destroy_instance(init.instance);
        glfwDestroyWindow(window);
        glfwTerminate();
    };

    void get_queues() {
        auto gq = init.device.get_queue(vkb::QueueType::graphics);
        if (!gq.has_value()) {
            std::cout << "failed to get graphics queue: " << gq.error().message() << "\n";
        }
        graphicsQueue = gq.value();

        auto pq = init.device.get_queue(vkb::QueueType::present);
        if (!pq.has_value()) {
            std::cout << "failed to get present queue: " << pq.error().message() << "\n";
        }
        presentQueue = pq.value();
    }

    void createSwapchain() {
        swapchain = Swapchain(&init, width, height);
    }

    void createGraphicsPipeline(const std::string& vertexPath, const std::string& fragmentPath) {
        pipeline = new GraphicsPipeline(vertexPath, fragmentPath, &swapchain, allocator);
    }

    void createFramebuffers() {
        auto imagesResult = swapchain.swapchain.get_images();
        if (!imagesResult.has_value()) {
            std::cout << "Failed to get swapchain images\n";
        }
        swapchainImages = imagesResult.value();

        auto imageViewsResult = swapchain.swapchain.get_image_views();
        if (!imageViewsResult.has_value()) {
            std::cout << "Failed to get swapchain image views\n";
        }
        swapchainImageViews = imageViewsResult.value();

        framebuffers.resize(swapchainImageViews.size());

        for (size_t i = 0; i < swapchainImageViews.size(); i++) {
            VkImageView attachments[] = { swapchainImageViews[i] };

            VkFramebufferCreateInfo framebuffer_info = {};
            framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebuffer_info.renderPass = pipeline->renderPass;
            framebuffer_info.attachmentCount = 1;
            framebuffer_info.pAttachments = attachments;
            framebuffer_info.width =  swapchain.swapchain.extent.width;
            framebuffer_info.height = swapchain.swapchain.extent.height;
            framebuffer_info.layers = 1;

            if (init.disp.createFramebuffer(&framebuffer_info, nullptr, &framebuffers[i]) != VK_SUCCESS) {
                throw std::runtime_error("couldn't create framebuffers");
                return;
            }
        }
    }

    void createCommandPool() {
        VkCommandPoolCreateInfo pool_info = {};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        auto queueFamilyIndexOpt = init.device.get_queue_index(vkb::QueueType::graphics);
        if (!queueFamilyIndexOpt.has_value()) {
            std::cout << "Failed to get graphics queue family index\n";
            // You may want to handle this error more robustly
            pool_info.queueFamilyIndex = 0;
        } else {
            pool_info.queueFamilyIndex = queueFamilyIndexOpt.value();
        }
		pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

        if (init.disp.createCommandPool(&pool_info, nullptr, &commandPool) != VK_SUCCESS) {
            std::cout << "failed to create command pool\n";
            return; // failed to create command pool
        }
    }

    void createCommandBuffers() {
        commandBuffers.resize(framebuffers.size());

        VkCommandBufferAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();

        if (init.disp.allocateCommandBuffers(&allocInfo, commandBuffers.data()) != VK_SUCCESS) {
            throw std::runtime_error("couldn't allocate cmd bufs");
        }
        secondaryCmdBufs.resize(framebuffers.size());
		VkCommandBufferAllocateInfo allocInfo2 = {};
		allocInfo2.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo2.level = VK_COMMAND_BUFFER_LEVEL_SECONDARY;
		allocInfo2.commandPool = commandPool;
		allocInfo2.commandBufferCount = (uint32_t)secondaryCmdBufs.size();

        if (init.disp.allocateCommandBuffers(&allocInfo2, secondaryCmdBufs.data()) != VK_SUCCESS) {
            throw std::runtime_error("couldn't allocate secondary cmd bufs");
        }

		secondaryCmdBufsDepth.resize(framebuffers.size());
		VkCommandBufferAllocateInfo allocInfo3 = {};
		allocInfo3.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo3.level = VK_COMMAND_BUFFER_LEVEL_SECONDARY;
		allocInfo3.commandPool = commandPool;
		allocInfo3.commandBufferCount = (uint32_t)secondaryCmdBufsDepth.size();
		if (init.disp.allocateCommandBuffers(&allocInfo3, secondaryCmdBufsDepth.data()) != VK_SUCCESS) {
			throw std::runtime_error("couldn't allocate secondary cmd bufs");
		}
    }

    void recordPrimaryCmds(uint32_t imageIndex) {
        VkCommandBufferBeginInfo begin_info = {};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        if (init.disp.beginCommandBuffer(commandBuffers[imageIndex], &begin_info) != VK_SUCCESS) {
            throw std::runtime_error("can't begin command buffer recording");
        }
            
		// viewport and scissor
        VkViewport viewport = {};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float)swapchain.swapchain.extent.width;
        viewport.height = (float)swapchain.swapchain.extent.height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        VkRect2D scissor = {};
        scissor.offset = { 0, 0 };
        scissor.extent = swapchain.swapchain.extent;

        // Update secondary command buffers every frame for depth render pass  
        VkCommandBufferInheritanceInfo inheritanceInfoDepth = {};
        inheritanceInfoDepth.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO;
        inheritanceInfoDepth.renderPass = pipeline->pPreviousPipeline->renderPass;
        inheritanceInfoDepth.framebuffer = pipeline->pPreviousPipeline->framebuffer->framebuffers[imageIndex];
        inheritanceInfoDepth.pNext = nullptr;

        VkCommandBufferBeginInfo beginInfoDepth = {};
        beginInfoDepth.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfoDepth.flags = VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT;
        beginInfoDepth.pInheritanceInfo = &inheritanceInfoDepth;

        // Issue draw calls  
        init.disp.beginCommandBuffer(secondaryCmdBufsDepth[imageIndex], &beginInfoDepth);
        init.disp.cmdSetViewport(secondaryCmdBufsDepth[imageIndex], 0, 1, &viewport);
        init.disp.cmdSetScissor(secondaryCmdBufsDepth[imageIndex], 0, 1, &scissor);
        init.disp.cmdBindPipeline(secondaryCmdBufsDepth[imageIndex], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline->pPreviousPipeline->pipeline);
        scene.renderSceneDepth(secondaryCmdBufsDepth[imageIndex], width, height);
        init.disp.endCommandBuffer(secondaryCmdBufsDepth[imageIndex]);

        // Update secondary command buffers every frame for main render pass  
        VkCommandBufferInheritanceInfo inheritanceInfo = {};
        inheritanceInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO;
        inheritanceInfo.renderPass = pipeline->renderPass;
        inheritanceInfo.framebuffer = framebuffers[imageIndex];

        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT;
        beginInfo.pInheritanceInfo = &inheritanceInfo;

        // Issue draw calls  
        init.disp.beginCommandBuffer(secondaryCmdBufs[imageIndex], &beginInfo);
        init.disp.cmdSetViewport(secondaryCmdBufs[imageIndex], 0, 1, &viewport);
        init.disp.cmdSetScissor(secondaryCmdBufs[imageIndex], 0, 1, &scissor);
        init.disp.cmdBindPipeline(secondaryCmdBufs[imageIndex], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline->pipeline);
        pipeline->bindDescSets(secondaryCmdBufs[imageIndex]);
        scene.renderScene(secondaryCmdBufs[imageIndex], width, height, imageIndex);
        init.disp.endCommandBuffer(secondaryCmdBufs[imageIndex]);

        // bind the depth pipeline first
        VkClearValue clearDepth{ { { 0.0f, 0 } } };
        pipeline->pPreviousPipeline->render(commandBuffers[imageIndex], secondaryCmdBufsDepth[imageIndex], clearDepth, imageIndex);
            
		// now bind the main pipeline
        VkRenderPassBeginInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        info.framebuffer = framebuffers[imageIndex];
        info.clearValueCount = 1;
        VkClearValue clearColor{ { { 0.0f, 0.0f, 0.0f, 1.0f } } };
        info.pClearValues = &clearColor;
        info.renderArea.offset = { 0, 0 };
        info.renderArea.extent = swapchain.swapchain.extent;
        info.renderPass = pipeline->renderPass;

        pipeline->render(commandBuffers[imageIndex], secondaryCmdBufs[imageIndex], info);

        if (init.disp.endCommandBuffer(commandBuffers[imageIndex]) != VK_SUCCESS) {
            throw std::runtime_error("couldn't end cmd buf");
        }
    }

    void createSyncObjects() {
        availiables.resize(MAX_FRAMES_IN_FLIGHT);
        finishes.resize(MAX_FRAMES_IN_FLIGHT);
        inFlights.resize(MAX_FRAMES_IN_FLIGHT);
        imagesInFlights.resize(swapchain.swapchain.image_count, VK_NULL_HANDLE);

        VkSemaphoreCreateInfo semaphore_info = {};
        semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fence_info = {};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            if (init.disp.createSemaphore(&semaphore_info, nullptr, &availiables[i]) != VK_SUCCESS ||
                init.disp.createSemaphore(&semaphore_info, nullptr, &finishes[i]) != VK_SUCCESS ||
                init.disp.createFence(&fence_info, nullptr, &inFlights[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create sync objects");
            }
        }
    }

    void drawFrame() {
        init.disp.waitForFences(1, &inFlights[currentFrame], VK_TRUE, UINT64_MAX);

        uint32_t image_index = 0;
        VkResult result = init.disp.acquireNextImageKHR(
            swapchain.swapchain, UINT64_MAX, availiables[currentFrame], VK_NULL_HANDLE, &image_index);

        //if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        //    return recreate_swapchain(init, data);
        //}
        if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
            throw std::runtime_error("failed to aquire next swapchain image");
        }

        if (imagesInFlights[image_index] != VK_NULL_HANDLE) {
            init.disp.waitForFences(1, &imagesInFlights[image_index], VK_TRUE, UINT64_MAX);
        }
        imagesInFlights[image_index] = inFlights[currentFrame];

        VkSubmitInfo submitInfo = {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        VkSemaphore wait_semaphores[] = { availiables[currentFrame] };
        VkPipelineStageFlags wait_stages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = wait_semaphores;
        submitInfo.pWaitDstStageMask = wait_stages;

        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[image_index];

        VkSemaphore signal_semaphores[] = { finishes[currentFrame] };
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signal_semaphores;

        init.disp.resetFences(1, &inFlights[currentFrame]);
        
        recordPrimaryCmds(image_index);

        if (init.disp.queueSubmit(graphicsQueue, 1, &submitInfo, inFlights[currentFrame]) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit frames to queue");
        }

        VkPresentInfoKHR present_info = {};
        present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

        present_info.waitSemaphoreCount = 1;
        present_info.pWaitSemaphores = signal_semaphores;

        VkSwapchainKHR swapChains[] = { swapchain.swapchain };
        present_info.swapchainCount = 1;
        present_info.pSwapchains = swapChains;

        present_info.pImageIndices = &image_index;

        result = init.disp.queuePresentKHR(presentQueue, &present_info);
        //if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
        //    return recreate_swapchain(init, data);
        //}
        if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to present to surface");
        }

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    // Store the previous state of the mouse buttons
    std::unordered_map<int, bool> mouseButtonStates;

    bool isMouseButtonPressed(GLFWwindow* window, int button) {
        int currentState = glfwGetMouseButton(window, button);

        // Check if the button was not pressed previously and is now pressed
        if (currentState == GLFW_PRESS && !mouseButtonStates[button]) {
            mouseButtonStates[button] = true;
            return true;
        }
        // If the button is released, remove it from the map
        else if (currentState == GLFW_RELEASE) {
            mouseButtonStates.erase(button);
        }

        return false;
    }

    std::unordered_map<int, bool> releaseMouseButtonStates;

    bool isMouseButtonReleased(GLFWwindow* window, int button) {
        int currentState = glfwGetMouseButton(window, button);

        // Check if the button was pressed previously and is now released
        if (currentState == GLFW_RELEASE && releaseMouseButtonStates[button]) {
            releaseMouseButtonStates[button] = false; // Update the state to released
            return true;
        }
        // If the button is pressed, update the map
        else if (currentState == GLFW_PRESS) {
            releaseMouseButtonStates[button] = true;
        }

        return false;
    }

    bool isMouseButtonPressedDown(GLFWwindow* window, int button) {
        if (glfwGetMouseButton(window, button)) return true;
        else return false;
    }

    glm::vec2 getCursorPosition(GLFWwindow* window) {
        double xPos;
        double yPos;
        glfwGetCursorPos(window, &xPos, &yPos);
        return glm::vec2(xPos, yPos);
    }

    std::unordered_map<int, bool> keyStates;

    bool isKeyPressed(GLFWwindow* window, int key) {
        int currentState = glfwGetKey(window, key);

        // Check if key was not pressed previously and is now pressed
        if (currentState == GLFW_PRESS && !keyStates[key]) {
            keyStates[key] = true;
            return true;
        }
        // Update the key state
        else if (currentState == GLFW_RELEASE) {
            keyStates.erase(key);
        }

        return false;
    }

    bool isKeyPressedDown(GLFWwindow* window, int key) {
        if (glfwGetKey(window, key) == GLFW_PRESS) {
            return true;
        }
        else return false;
    }

    static float xoffset;
    static float yoffset;
	static bool cursorEnabled;

    static void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
        static float lastX = 400, lastY = 300;
        static bool firstMouse = true;
        static float movementInterval = 0.005f; // Set interval duration in seconds (50 ms)
        static float lastMovementTime = 0.0f;  // Store the last time movement was triggered

        if (firstMouse) {
            lastX = xpos;
            lastY = ypos;
            firstMouse = false;
        }

        xoffset = xpos - lastX;
        yoffset = lastY - ypos;
        lastX = xpos;
        lastY = ypos;
        if (!cursorEnabled)
            scene.camera.ProcessMouseMovement(xoffset, -yoffset, true);
    }

    void toggleCursor(GLFWwindow* window, bool enableCursor) {
        if (enableCursor) {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);  // Show and unlock the cursor
            cursorEnabled = true;
        }
        else {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);  // Hide and lock the cursor
            cursorEnabled = false;
        }
    }

    float deltaTime = 0;

    void processInput(Camera& camera) {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);

        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            camera.ProcessKeyboard(FORWARD, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            camera.ProcessKeyboard(BACKWARD, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            camera.ProcessKeyboard(LEFT, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            camera.ProcessKeyboard(RIGHT, deltaTime);
    }

    // main loop
    void run() {  
        start();
        auto lastTime = std::chrono::high_resolution_clock::now();  
        
        while (!glfwWindowShouldClose(window)) {  
            auto currentTime = std::chrono::high_resolution_clock::now();
            deltaTime = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - lastTime).count();
            lastTime = currentTime;

            if (isMouseButtonPressedDown(window, GLFW_MOUSE_BUTTON_2)) {
                toggleCursor(window, false);
            }
            else toggleCursor(window, true);

			processInput(scene.camera);
            
            drawFrame();

            auto current = scene.entities[1].transform.getPosition();
		    auto next = current + deltaTime * glm::vec3(1.0f, 0.0f, 0.0f);
		    scene.entities[1].transform.setPosition(next);

			//std::cout << glm::to_string(scene.camera.Position) << "\n";

            glfwPollEvents();
             
        }  
        init.disp.deviceWaitIdle();  
    }
};

bool Engine::cursorEnabled = true;
float Engine::xoffset = 0.0f;
float Engine::yoffset = 0.0f;

Scene Engine::scene{};