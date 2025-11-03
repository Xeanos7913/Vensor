#pragma once

#include <vulkan/vulkan_core.h>
#include "VkBootstrap.h"  
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <unordered_map>
#include <functional>
#include "VkMemAlloc.hpp"

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


// Doesn't own a renderpass. Instead, a renderpass owns this.
// To create a custom pipeline, inherit from this and override the createPipeline() function
// Or alternatively, pass an std::function that creates the pipeline you wish
// For buffers, just use buffer reference. For bindless descriptors, need to override a few more things.
struct PipelineLite {
    VkPipeline pipeline;
    VkPipelineLayout layout;

    VkDescriptorSet descriptorSet;
    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorPool descriptorPool;

    std::vector<VkDescriptorSetLayout> descriptorSetLayouts;

    std::vector<VkPushConstantRange> pushConsts;
    std::vector<std::shared_ptr<Image2D>> images;
    std::vector<ImageArray> imageArrays;

    std::vector<VkShaderModule> shaders;
    std::vector<VkPipelineShaderStageCreateInfo> shader_stages;

    Swapchain* swapchain;
    Allocator* allocator;

    uint32_t subpassIndex = 0;

    int width, height;

    std::function<void()> pipelineCreationFunction;

    PipelineLite(Swapchain* swapchain, Allocator* allocator, int width, int height) : swapchain(swapchain), allocator(allocator), width(width), height(height) { }

    PipelineLite(Swapchain* swapchain, Allocator* allocator) : swapchain(swapchain), allocator(allocator) {
    }

    PipelineLite() {};

    virtual void initialize() {

        if (images.size() > 0) {
            createDescriptorPool();
            createDescriptorLayouts();
        }

        createPipelineLayout();
        createPipeline();
    }

    // no hand-holding here. You need to write this on your own.
    virtual void createPipeline() {
        pipelineCreationFunction();
    }

	virtual void addImageArray(uint32_t maxImages, VkImageUsageFlags usage, VkImageLayout layout, VkShaderStageFlags stage) {
		ImageArray imageArray = ImageArray(maxImages, allocator);
        imageArray.updateDescriptorSets();
		imageArrays.push_back(imageArray);
	}

    virtual void addPushConstant(VkDeviceSize range, VkDeviceSize offset, VkShaderStageFlags stage) {
        VkPushConstantRange constant = {};
        constant.offset = offset;
        constant.size = range;
        constant.stageFlags = stage;

        pushConsts.push_back(constant);
    }

    virtual void createDescriptorPool() {
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

    virtual void createDescriptorLayouts() {
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
			imageArray.updateDescriptorSets();
		}
    }

    virtual void addImage(std::shared_ptr<Image2D> image) {
        image->createDescriptors(images.size(), VK_SHADER_STAGE_ALL);
        image->updateDescriptorSet(descriptorSet, 0);
        images.push_back(image);
    }

    virtual void createPipelineLayout() {

        VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;

        if (images.size() > 0 && imageArrays.size() > 0) {
			descriptorSetLayouts.resize(imageArrays.size() + 1);
			descriptorSetLayouts[0] = descriptorSetLayout;
			for (size_t i = 0; i < imageArrays.size(); ++i) {
				descriptorSetLayouts[i + 1] = imageArrays[i].descSetLayout;
			}
			pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(descriptorSetLayouts.size());
            pipelineLayoutInfo.pSetLayouts = descriptorSetLayouts.data();
        }
        else if (images.size() == 0 && imageArrays.size() > 0) {
			descriptorSetLayouts.resize(imageArrays.size());
			for (size_t i = 0; i < imageArrays.size(); ++i) {
				descriptorSetLayouts[i] = imageArrays[i].descSetLayout;
			}
			pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(descriptorSetLayouts.size());
			pipelineLayoutInfo.pSetLayouts = descriptorSetLayouts.data();
		}
        else if (images.size() > 0 && imageArrays.size() == 0) {
            pipelineLayoutInfo.setLayoutCount = 1;
            pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
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

    virtual void bindDescSets(VkCommandBuffer& cmd) const {
        allocator->init->disp.cmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &descriptorSet, 0, nullptr);
        for (int i = 0; i < imageArrays.size(); i++) {
            allocator->init->disp.cmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, i + 1, 1, &imageArrays[i].descSet, 0, nullptr);
        }
    }
};

struct Subpass {
    std::vector<FBAttachment*> colorAttachments;
    std::vector<FBAttachment*> inputAttachments;
    FBAttachment* depthAttachment;

    // Backing storage for VkAttachmentReferences (must persist)
    std::vector<VkAttachmentReference> inputAttachmentRefs;
    std::vector<VkAttachmentReference> colorAttachmentRefs;

    VkSubpassDescription subpassDescription;
    VkSubpassDependency subpassDependency;
    VkClearValue clearValue;

    PipelineLite* pipeline;

    bool hasDepthStencil = false;

    Allocator* allocator;

    // subpasses are a doubly linked list
    Subpass* pNext = nullptr;
    Subpass* pPrevious = nullptr;
    int idx = 0; // the index of this subpass in the linked list

    Subpass() {};

    Subpass(Allocator* allocator, VkAccessFlags accessMask, VkPipelineStageFlags stageFlags, int idx, Subpass* pPrevious) : allocator(allocator), idx(idx), pPrevious(pPrevious){
        if (idx == 0 && pPrevious == nullptr) {
            subpassDependency.srcSubpass = VK_SUBPASS_EXTERNAL;
            subpassDependency.dstSubpass = 0;
            subpassDependency.srcAccessMask = 0;
            subpassDependency.dstAccessMask = accessMask;
            subpassDependency.srcStageMask = stageFlags;
            subpassDependency.dstStageMask = stageFlags;
			subpassDependency.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT; // this is the default, but you can change it if needed
        }
        else if (pPrevious != nullptr) {
            pPrevious->subpassDependency.dstSubpass = idx;
            subpassDependency.srcSubpass = idx - 1;
            subpassDependency.dstSubpass = idx;
            subpassDependency.srcAccessMask = pPrevious->subpassDependency.dstAccessMask; // might need to customize
            subpassDependency.dstAccessMask = accessMask;
            subpassDependency.srcStageMask = pPrevious->subpassDependency.dstStageMask;   // same
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

enum class AttachmentType {
    INPUT,
    COLOR,
    DEPTH
};

struct Renderer {
    std::vector<std::unique_ptr<Subpass>> subpasses;
    std::vector<std::unique_ptr<PipelineLite>> pipelines;

    Allocator* allocator;
    Swapchain* swapchain;

    VkRenderPass renderPass;

	Framebuffer framebuffer;

    int width, height, maxFrames;

    Renderer(Allocator* allocator, Swapchain* swapchain, int numSubpasses, int maxFrames) :  allocator(allocator), swapchain(swapchain), maxFrames(maxFrames), framebuffer(swapchain->width, swapchain->height, 1, maxFrames, renderPass, allocator) {
        if (numSubpasses < 1) throw std::runtime_error("need at least one subpass");

        // the default subpass is always there
        auto subpass = std::make_unique<Subpass>(allocator, VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, nullptr);
        auto attachment = std::make_shared<FBAttachment>(swapchain->width, swapchain->height, 0, swapchain->swapchain.image_format, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, allocator);
        subpass->addColorAttachment(attachment.get());
        subpass->clearValue = VkClearValue{ { { 0.0f, 0.0f, 0.0f, 1.0f } } };
        subpasses.push_back(std::move(subpass));
		addAttachmeentToSubpass(0, AttachmentType::COLOR, attachment);
    }

    Renderer(Allocator* allocator, int width, int height, int numSubpasses, int maxFrames) : allocator(allocator), swapchain(nullptr), width(width), height(height), maxFrames(maxFrames) {
        if (numSubpasses < 1) throw std::runtime_error("need at least one subpass");

        // the default subpass is always there
        auto subpass = std::make_unique<Subpass>(allocator, VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, nullptr);
        auto attachment = std::make_shared<FBAttachment>(width, height, 0, VK_FORMAT_UNDEFINED, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, allocator);
        subpass->addColorAttachment(attachment.get());
        subpass->clearValue = VkClearValue{ { { 0.0f, 0.0f, 0.0f, 1.0f } } };
        subpasses.push_back(std::move(subpass));
		addAttachmeentToSubpass(0, AttachmentType::COLOR, attachment);
    }

    void addSubpass(VkAccessFlags accessFlag, VkPipelineStageFlagBits stageFlags, int pipelineIdx) {
        auto subpass = std::make_unique<Subpass>(allocator, accessFlag, stageFlags, subpasses.size(), subpasses.back().get());
        subpass->pPrevious = subpasses.back().get();
        subpass->pipeline = pipelines[pipelineIdx].get();
        subpasses.back()->pNext = subpass.get();
        subpasses.push_back(std::move(subpass));
    }

	void addSubpass(std::unique_ptr<Subpass>& subpass) {
		if (subpasses.size() == 0) {
			throw std::runtime_error("Renderer must have at least one subpass before adding a new one.");
		}
		subpass->pPrevious = subpasses.back().get();
		subpasses.back()->pNext = subpass.get();
		subpasses.push_back(std::move(subpass));
	}

    // before adding, the pipeline must already know the index of the subpass it's going to affect
    void addPipeline(std::unique_ptr<PipelineLite>& pipe) {
        pipelines.push_back(std::move(pipe));
    }

    void addAttachmeentToSubpass(int subpassIdx, AttachmentType type, std::shared_ptr<FBAttachment>& attachment) {  

        switch (type) {  
        case AttachmentType::INPUT:  
            subpasses[subpassIdx]->addInputAttachment(attachment.get());  
			framebuffer.addAttachment(attachment);
            break;  
        case AttachmentType::COLOR:  
            subpasses[subpassIdx]->addColorAttachment(attachment.get());  
			framebuffer.addAttachment(attachment);
            break;  
        case AttachmentType::DEPTH:  
            subpasses[subpassIdx]->addDepthStencilAttachment(attachment.get());  
			framebuffer.addAttachment(attachment);
            break;  
        default:  
            throw std::runtime_error("can't discern attachment type");  
            break;  
        }  
    }

    void build() {
        std::vector<VkAttachmentDescription> attachs = framebuffer.getAttachmentDescriptions();
        std::vector<VkAttachmentReference> attachRefs = framebuffer.getAttachmentReferences();

        std::vector<VkSubpassDescription> subs(subpasses.size());
        std::vector<VkSubpassDependency> subDependencies(subpasses.size());

        for (int i = 0; i < subpasses.size(); i++) {
            subs[i] = subpasses[i]->subpassDescription;
            subDependencies[i] = subpasses[i]->subpassDependency;
        }

        VkRenderPassCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        createInfo.attachmentCount = attachs.size();
        createInfo.pAttachments = attachs.data();
        createInfo.subpassCount = subs.size();
        createInfo.pSubpasses = subs.data();
        createInfo.dependencyCount = subDependencies.size();
        createInfo.pDependencies = subDependencies.data();

        if (allocator->init->disp.createRenderPass(&createInfo, nullptr, &renderPass) != VK_SUCCESS) {
            std::cout << "failed to create render pass\n";
        }
    }

    void render(VkCommandBuffer cmd, uint32_t width, uint32_t height, std::vector<std::vector<VkCommandBuffer>>& commandBuffers) {

        for (int i = 0; i < maxFrames; i++) {
            VkRenderPassBeginInfo beginInfo = {};
            beginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            beginInfo.framebuffer = framebuffer.framebuffers[i];
            beginInfo.clearValueCount = 0;
            beginInfo.pClearValues = nullptr;
            beginInfo.renderPass = renderPass;
            beginInfo.renderArea.offset = { 0, 0 };
            beginInfo.renderArea.extent = VkExtent2D{ width, height };

            allocator->init->disp.cmdBeginRenderPass(cmd, &beginInfo, VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS);
            for (int j = 0; j < subpasses.size(); j++) {
				if (subpasses[j]->pipeline) {
					allocator->init->disp.cmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, subpasses[j]->pipeline->pipeline);
					subpasses[j]->pipeline->bindDescSets(cmd);
					allocator->init->disp.cmdExecuteCommands(cmd, commandBuffers[j].size(), commandBuffers[j].data());
				}
				allocator->init->disp.cmdNextSubpass(cmd, VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS);
			}
        }
    }
};

// this will use the renderer to generate frames which will be displayed on the screen
struct Projector{

    

};