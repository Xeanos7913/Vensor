#pragma once

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "VkBootstrap.h"
#include "volk.h"
#include <typeinfo>
#include <iostream>
#include <algorithm>
#include <memory>
#include "signal.hpp"
#include "stb_image.h"
#include "stb_image_write.h"

struct Init {
    vkb::Instance instance;
    vkb::InstanceDispatchTable inst_disp;
    vkb::Device device;
    vkb::DispatchTable disp;

    struct func {
		void (*func)(void*);
		void* ptr;
    };

	std::vector<func> funcs;

    template<typename T>
	static void invokeFunc(void* p) {
		static_cast<T*>(p)->destructor();
	}

	template<typename T>
	void addObject(T* object) {
		funcs.push_back({ &invokeFunc<T>, static_cast<void*>(object) });
	}

	void destroy() {
		for (auto& f : funcs) {
			f.func(f.ptr);
		}
		funcs.clear();
	}

	Init() {};
    Init(vkb::Instance instance, vkb::InstanceDispatchTable inst_disp, vkb::Device device, vkb::DispatchTable disp) :
        instance(instance), inst_disp(inst_disp), device(device), disp(disp) {
	}
    ~Init() {
        //destroy();
    }
};

uint32_t get_memory_index(Init& init, const uint32_t type_bits, VkMemoryPropertyFlags properties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) {
    VkPhysicalDeviceMemoryProperties mem_props = init.device.physical_device.memory_properties;
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
        if ((type_bits & (1 << i)) && (mem_props.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    return UINT32_MAX; // No valid memory type found
}

struct Allocator {
    
    Allocator(Init* init) : init(init) {
        commandBuffers.resize(1);
        get_queues();
		create_command_pool();
    };

    Allocator() {};
    size_t getAlignmemt() const {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(init->device.physical_device, &props);
		return props.limits.minStorageBufferOffsetAlignment;
	}
    Allocator(Allocator& other) {
        init = other.init;
        allocQueue = other.allocQueue;
        create_command_pool();
		graphicsPool = other.graphicsPool;
		graphicsQueue = other.graphicsQueue;
    }

    ~Allocator() {

        for (auto& mem : allocated) {
			init->disp.destroyBuffer(mem.first, nullptr);
			init->disp.freeMemory(mem.second, nullptr);
        }

		for (auto& img : images) {
			init->disp.destroyImage(img.first, nullptr);
			init->disp.freeMemory(img.second, nullptr);
		}

        if (!commandBuffers.empty()) {
            init->disp.freeCommandBuffers(commandPool, static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());
			commandBuffers.clear();
        }

		init->disp.destroyCommandPool(commandPool, nullptr);
    }
    
    void killMemory(VkBuffer buffer, VkDeviceMemory memory) {
        if (buffer == VK_NULL_HANDLE || memory == VK_NULL_HANDLE) {
            return;
        }
        auto it = std::find_if(allocated.begin(), allocated.end(),
            [buffer, memory](const std::pair<VkBuffer, VkDeviceMemory>& pair) {
                return pair.first == buffer && pair.second == memory;
            });
        if (it != allocated.end()) {
            allocated.erase(it);
        }
        else {
            std::cerr << "Warning: Attempted to kill memory that was not found in allocated list." << std::endl;
        }
        init->disp.destroyBuffer(buffer, nullptr);
        init->disp.freeMemory(memory, nullptr);
    }

    void killImage(VkImage image, VkDeviceMemory memory) {
		if (image == VK_NULL_HANDLE || memory == VK_NULL_HANDLE) {
			return;
		}
		auto it = std::find_if(images.begin(), images.end(),
			[image, memory](const std::pair<VkImage, VkDeviceMemory>& pair) {
				return pair.first == image && pair.second == memory;
			});
		if (it != images.end()) {
			images.erase(it);
		}
		else {
			std::cerr << "Warning: Attempted to kill image that was not found in allocated list." << std::endl;
		}
        init->disp.destroyImage(image, nullptr);
        init->disp.freeMemory(memory, nullptr);
    }

    std::pair<VkBuffer, VkDeviceMemory> createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, bool addToDeletionQueue = true) {
        VkBuffer buffer{};
		VkDeviceMemory bufferMemory{};

        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        init->disp.createBuffer(&bufferInfo, nullptr, &buffer);

        VkMemoryRequirements memRequirements;
        init->disp.getBufferMemoryRequirements(buffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = get_memory_index(*init, memRequirements.memoryTypeBits, properties);
        
        VkMemoryAllocateFlagsInfo allocFlagsInfo{};
        allocFlagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
        allocFlagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

		allocInfo.pNext = &allocFlagsInfo;

        if (init->disp.allocateMemory(&allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
            throw std::runtime_error("could not allocate memory");
        }
        if (init->disp.bindBufferMemory(buffer, bufferMemory, 0) != VK_SUCCESS) {
            throw std::runtime_error("could not bind memory");
        }
        if (addToDeletionQueue) {
            allocated.emplace_back(buffer, bufferMemory);
        }

        return { buffer, bufferMemory };
    };

	std::pair<VkImage, VkDeviceMemory> createImage(VkDeviceSize width, VkDeviceSize height, uint32_t mipLevels, VkImageUsageFlags usage, VkImageType imageType, VkMemoryPropertyFlags properties, VkFormat format = VK_FORMAT_R8G8B8A8_UNORM,VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT, VkSharingMode sharingMode = VK_SHARING_MODE_EXCLUSIVE) {
		VkImage image{};
		VkDeviceMemory imageMemory{};
		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = imageType;
		imageInfo.extent.width = static_cast<uint32_t>(width);
		imageInfo.extent.height = static_cast<uint32_t>(height);
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = mipLevels;
		imageInfo.arrayLayers = 1;
		imageInfo.format = format;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageInfo.usage = usage;
		imageInfo.samples = samples;
		imageInfo.sharingMode = sharingMode;
		init->disp.createImage(&imageInfo, nullptr, &image);
		VkMemoryRequirements memRequirements;
		init->disp.getImageMemoryRequirements(image, &memRequirements);
		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = get_memory_index(*init, memRequirements.memoryTypeBits, properties);
		if (init->disp.allocateMemory(&allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
			throw std::runtime_error("could not allocate memory");
		}
		if (init->disp.bindImageMemory(image, imageMemory, 0) != VK_SUCCESS) {
			throw std::runtime_error("could not bind memory");
		}
        images.emplace_back(image, imageMemory);
		return { image, imageMemory };
	}

    template<typename U>
    void fillBuffer(VkBuffer buffer, VkDeviceMemory memory, U data, VkDeviceSize offset = 0, VkDeviceSize range = VK_WHOLE_SIZE) {
		auto cmd = beginSingleTimeCommands();
        vkCmdFillBuffer(commandBuffers[cmd], buffer, offset, range, static_cast<uint32_t>(data));
		endSingleTimeCommands(true);
    }

	// avoid as much as possible, this kills performance. But if you need to free memory, this is the way to do it.
    void freeMemory(VkBuffer buffer, VkDeviceSize offset, VkDeviceSize range) {  
        // Step 1: Create a staging buffer to temporarily hold the data
        VkDeviceSize bufferSize;

        // Get memory requirements for the buffer
        VkMemoryRequirements memRequirements;
        init->disp.getBufferMemoryRequirements(buffer, &memRequirements);
        bufferSize = memRequirements.size;

        // Create a staging buffer
        auto [stagingBuffer, stagingMemory] = createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		auto cmd = getSingleTimeCmd();
        // Step 2: Copy data before the gap to the staging buffer
        if (offset > 0) {
            recordCopyBufferCmd(cmd, buffer, stagingBuffer, offset, 0, 0);
        }

        // Step 3: Copy data after the gap to the staging buffer
        VkDeviceSize end = offset + range;
        VkDeviceSize remainingSize = (end <= bufferSize) ? bufferSize - end : 0;
        if (remainingSize > 0) {
            recordCopyBufferCmd(cmd, buffer, stagingBuffer, remainingSize, offset + range, offset);
        }

        // Step 4: Copy the data back from the staging buffer to the original buffer
        if (offset > 0) {
            recordCopyBufferCmd(cmd, stagingBuffer, buffer, offset, 0, 0);
        }

        if (remainingSize > 0) {  
            recordCopyBufferCmd(cmd, stagingBuffer, buffer, remainingSize, offset, offset);
        }

        submitSingleTimeCmd(cmd);

        // Step 5: Clear the staging buffer
		killMemory(stagingBuffer, stagingMemory);
    }

	// this overrides the memory in offset -> insertSize with toInsert(0 -> insertSize).
    void replaceMemory(VkBuffer& buffer, VkDeviceMemory& memory, VkBuffer toInsert, VkDeviceSize insertSize, VkDeviceSize offset) {
        // Step 1: Get memory requirements
        VkMemoryRequirements bufferReq;
        init->disp.getBufferMemoryRequirements(buffer, &bufferReq);

        VkDeviceSize totalSize = bufferReq.size;

        // Step 2: Create staging buffer
        auto [stagingBuffer, stagingMemory] = createBuffer(totalSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

		auto cmd = getSingleTimeCmd();

        // Step 3: Copy data before insertion point
        if (offset > 0) {
            recordCopyBufferCmd(cmd, buffer, stagingBuffer, offset, 0, 0);
        }

        // Step 4: Copy new data to insert
        recordCopyBufferCmd(cmd, toInsert, stagingBuffer, insertSize, 0, offset);

        // Step 5: Copy remaining original buffer data
        VkDeviceSize remainingSize = totalSize - (offset + insertSize);
        if (remainingSize > 0) {
            recordCopyBufferCmd(cmd, buffer, stagingBuffer, remainingSize, offset, offset + insertSize);
        }

		recordCopyBufferCmd(cmd, stagingBuffer, buffer, totalSize, 0, 0);

        submitSingleTimeCmd(cmd);

        // Step 6: Destroy staging buffer
		killMemory(stagingBuffer, stagingMemory);
    }

    // insert blob of memory into given buffer. Make sure given buffer has enough free space to handle the insert block, otherwise it'll kill the remaining trailing data
    void insertMemory(VkBuffer buffer, VkBuffer& toInsert, VkDeviceSize offset, VkDeviceSize inSize) {
        VkMemoryRequirements req;
        init->disp.getBufferMemoryRequirements(buffer, &req);
        auto size = req.size;

        auto [stagingBuffer, stagingMemory] = createBuffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

		auto cmd = getSingleTimeCmd();

        if (offset > 0) {
            recordCopyBufferCmd(cmd, buffer, stagingBuffer, offset, 0, 0);
        }
        recordCopyBufferCmd(cmd, toInsert, stagingBuffer, inSize, 0, offset);
        recordCopyBufferCmd(cmd, buffer, stagingBuffer, size - (offset + inSize), offset, offset + inSize);
        recordCopyBufferCmd(cmd, stagingBuffer, buffer, size, 0, 0);

        submitSingleTimeCmd(cmd);

		killMemory(stagingBuffer, stagingMemory);
    }

	// this is a defragmenter. It will copy the good data from the original buffer to a new buffer, and then free the original buffer, killing the stale memory
    void defragment(VkBuffer& buffer, VkDeviceMemory& memory, std::vector<std::pair<VkDeviceSize, VkDeviceSize>>& aliveMem) {
		// create new staging buffer which will replace the original buffer
		VkDeviceSize bufferSize;
		// Get memory requirements for the buffer
		VkMemoryRequirements memRequirements;
		init->disp.getBufferMemoryRequirements(buffer, &memRequirements);
		bufferSize = memRequirements.size;
		// Create the clean buffer
		auto [stagingBuffer, stagingMemory] = createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        VkDeviceSize runningOffset = 0;

        auto cmd = getSingleTimeCmd();

        // copy all the good buffers into the staging buffer, one after the other
		for (auto& [offset, range] : aliveMem) {
			// copy the data from the original buffer to the new one
			recordCopyBufferCmd(cmd, buffer, stagingBuffer, range, offset, runningOffset);
			runningOffset += range;
		}

		// copy the curated data from the staging buffer to the original buffer
		recordCopyBufferCmd(cmd, stagingBuffer, buffer, bufferSize, 0, 0);

		submitSingleTimeCmd(cmd);

		killMemory(stagingBuffer, stagingMemory);
    }

    void recordCopyBufferCmd(VkCommandBuffer cmdBuffer, VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size, VkDeviceSize srcOffset = 0, VkDeviceSize dstOffset = 0) {
        if (size == 0) {
            return;
        }
        VkBufferCopy copyRegion{};
        copyRegion.size = size;
        copyRegion.dstOffset = dstOffset;
        copyRegion.srcOffset = srcOffset;
        vkCmdCopyBuffer(cmdBuffer, srcBuffer, dstBuffer, 1, &copyRegion);
	}

    void transitionImageLayout(
        VkImage image,
        VkFormat format,
        VkImageLayout oldLayout,
        VkImageLayout newLayout,
        uint32_t mipLevels = 1) {
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;

        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

        barrier.image = image;
		if (format == VK_FORMAT_D32_SFLOAT || format == VK_FORMAT_D32_SFLOAT_S8_UINT ||
			format == VK_FORMAT_D24_UNORM_S8_UINT || format == VK_FORMAT_D16_UNORM_S8_UINT) {
			barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
		}
		else {
			barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		}
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = mipLevels;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;

        // Determine source and destination stages
        VkPipelineStageFlags sourceStage, destinationStage;

        if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
            newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

            sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        }
        else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
            newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        }
        else if (oldLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
			barrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
			sourceStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
			destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		}
        else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
			barrier.srcAccessMask = 0;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
			sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		}
		else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
			barrier.srcAccessMask = 0;
			barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
			sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			destinationStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		}
		else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
			barrier.srcAccessMask = 0;
			barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
			sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			destinationStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        }
        else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL) {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
            sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        }
        else {
            throw std::invalid_argument("unsupported layout transition!");
        }
        auto cmd = beginSingleTimeCommands(true);
        vkCmdPipelineBarrier(
            commandBuffers[cmd],
            sourceStage, destinationStage,
            0,
            0, nullptr,
            0, nullptr,
            1, &barrier
        );
        endSingleTimeCommands(true, true, true);
    }

    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size, VkDeviceSize srcOffset, VkDeviceSize dstOffset, bool async = false) {
		if (size == 0) {
			return;
		}
        auto cmd = beginSingleTimeCommands();
        VkBufferCopy copyRegion{};
        copyRegion.size = size;
        copyRegion.dstOffset = dstOffset;
        copyRegion.srcOffset = srcOffset;
        vkCmdCopyBuffer(commandBuffers[cmd], srcBuffer, dstBuffer, 1, &copyRegion);
        endSingleTimeCommands(async);
    }

	void sequentialCopyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size, VkDeviceSize srcOffset, VkDeviceSize dstOffset) {
		if (size == 0) {
			return;
		}
        auto cmd = beginSingleTimeCommands();
		VkBufferCopy copyRegion{};
		copyRegion.size = size;
		copyRegion.dstOffset = dstOffset;
		copyRegion.srcOffset = srcOffset;
		vkCmdCopyBuffer(commandBuffers[cmd], srcBuffer, dstBuffer, 1, &copyRegion);
        endSingleTimeCommands(true, false);
	}

    // only supports sqare images
	void copyImage(VkImage srcImage, VkImage dstImage, VkDeviceSize size, VkDeviceSize srcOffset, VkDeviceSize dstOffset) {
		auto cmd = beginSingleTimeCommands();
		VkImageCopy copyRegion{};
		copyRegion.extent.width = static_cast<uint32_t>(size);
		copyRegion.extent.height = static_cast<uint32_t>(size);
		copyRegion.extent.depth = 1;
		copyRegion.dstOffset.x = static_cast<int32_t>(dstOffset);
		copyRegion.dstOffset.y = static_cast<int32_t>(dstOffset);
		copyRegion.srcOffset.x = static_cast<int32_t>(srcOffset);
		copyRegion.srcOffset.y = static_cast<int32_t>(srcOffset);
		vkCmdCopyImage(commandBuffers[cmd], srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dstImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);
		endSingleTimeCommands();
	}

    // does not support mipmaps or array textures
    void copyBufferToImage2D(VkBuffer srcBuffer, VkImage dstImage, uint32_t width, uint32_t height) {
        auto cmd = beginSingleTimeCommands();

        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;

        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;

        region.imageOffset = { 0, 0, 0 };
        region.imageExtent = {
            width,
            height,
            1
        };

        vkCmdCopyBufferToImage(
            commandBuffers[cmd],
            srcBuffer,
            dstImage,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &region
        );

		endSingleTimeCommands(true, true, false);
    }

	// this begins the command buffer recording process. Just record the commands in between this function's call and the submitSingleTimeCmd call
	// submitSingleTimeCmd will NOT automatically end the command buffer!!
	VkCommandBuffer getSingleTimeCmd(bool useGraphicsQueue = false) const {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        if (!useGraphicsQueue) {
            allocInfo.commandPool = commandPool;
        }
        else {
            allocInfo.commandPool = graphicsPool;
        }

        allocInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer;
        init->disp.allocateCommandBuffers(&allocInfo, &commandBuffer);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        init->disp.beginCommandBuffer(commandBuffer, &beginInfo);
		return commandBuffer;
	}

	void submitSingleTimeCmd(VkCommandBuffer cmd, bool async = true, bool useGraphicsQueue = false) {
		SubmitSingleTimeCommand(cmd, async, useGraphicsQueue);
	}

    Init* init;
    VkCommandPool commandPool;
    VkCommandPool graphicsPool;
    VkQueue graphicsQueue;
    std::vector<std::pair<VkBuffer, VkDeviceMemory>> allocated;
private:
    VkQueue allocQueue;
    std::vector<VkCommandBuffer> commandBuffers;
    
    std::vector<std::pair<VkImage, VkDeviceMemory>> images;

    int beginSingleTimeCommands(bool useGraphicsQueue = false) {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        if (!useGraphicsQueue) {
            allocInfo.commandPool = commandPool;
		}
		else {
			allocInfo.commandPool = graphicsPool;
		}
        
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer;
        init->disp.allocateCommandBuffers(&allocInfo, &commandBuffer);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        init->disp.beginCommandBuffer(commandBuffer, &beginInfo);
        commandBuffers.push_back(commandBuffer);
		return commandBuffers.size() - 1; // return the index at which we just pushed the command buffer
    }

    int SubmitSingleTimeCommand(VkCommandBuffer commandBuffer, bool async = true, bool useGraphicsQueue = false) const {
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;
        if (async) {
            if (useGraphicsQueue) {
			    init->disp.queueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
                init->disp.queueWaitIdle(graphicsQueue);
                init->disp.freeCommandBuffers(graphicsPool, 1, &commandBuffer);
		    }
		    else if (!useGraphicsQueue) {
			    init->disp.queueSubmit(allocQueue, 1, &submitInfo, VK_NULL_HANDLE);
                init->disp.queueWaitIdle(allocQueue);
				init->disp.freeCommandBuffers(commandPool, 1, &commandBuffer);
		    }
        }
        else {
            if (useGraphicsQueue) {
                init->disp.queueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
            }
            else if (!useGraphicsQueue) {
                init->disp.queueSubmit(allocQueue, 1, &submitInfo, VK_NULL_HANDLE);
            }
            init->disp.deviceWaitIdle();

            if (useGraphicsQueue) {
			    init->disp.freeCommandBuffers(graphicsPool, 1, &commandBuffer);
		    }
		    else {
			    init->disp.freeCommandBuffers(commandPool, 1, &commandBuffer);
		    }
        }
		return 0;
    }

    void endSingleTimeCommands(bool async = false, bool dispatch = true, bool dispathOnGraphics = false) {
        if (commandBuffers.empty()) return;

        auto cmd = commandBuffers.back();
        init->disp.endCommandBuffer(cmd);

        if (dispatch) {
            VkSubmitInfo submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &cmd;
			if (dispathOnGraphics) {
				init->disp.queueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
			}
			else {
				init->disp.queueSubmit(allocQueue, 1, &submitInfo, VK_NULL_HANDLE);
			}

            if (!async) {
                init->disp.deviceWaitIdle();
            }
            else {
                if (!dispathOnGraphics) {
                    init->disp.queueWaitIdle(allocQueue);
				}
                else {
                    init->disp.queueWaitIdle(graphicsQueue);
                }
            }
            if (dispathOnGraphics) {
                init->disp.freeCommandBuffers(graphicsPool, 1, &cmd);
            }
            else {
                init->disp.freeCommandBuffers(commandPool, 1, &cmd);
            }
            commandBuffers.pop_back();
        }
    }

    void submitAllCommands(bool async = false) {
        if (commandBuffers.empty()) return;
		//commandBuffers.erase(commandBuffers.begin()); // the first command buffer is apparently invalid?! will have to fix later, right now, I'm exhausted.
        VkSubmitInfo submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = commandBuffers.size();
            submitInfo.pCommandBuffers = commandBuffers.data();
            init->disp.queueSubmit(allocQueue, 1, &submitInfo, VK_NULL_HANDLE);
        if (async) { init->disp.queueWaitIdle(allocQueue); }    // this stalls the transfer queue, but not the whole device.
		else { init->disp.deviceWaitIdle(); } 		// this stalls the whole device. This is terrible. But it's useful for stuff where you need to wait for the GPU to finish before you can do anything else.
		
        init->disp.freeCommandBuffers(commandPool, static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());
        commandBuffers.clear();
    }

    int get_queues() {
        auto gq = init->device.get_queue(vkb::QueueType::transfer);
        if (!gq.has_value()) {
            std::cout << "failed to get queue: " << gq.error().message() << "\n";
            return -1;
        }
        allocQueue = gq.value();
        return 0;
    }

    void create_command_pool() {
        VkCommandPoolCreateInfo pool_info = {};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.queueFamilyIndex = init->device.get_queue_index(vkb::QueueType::transfer).value();
        init->disp.createCommandPool(&pool_info, nullptr, &commandPool);

        VkCommandBufferAllocateInfo allocate_info = {};
        allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocate_info.commandPool = commandPool;
        allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocate_info.commandBufferCount = commandBuffers.size();
        //init->disp.allocateCommandBuffers(&allocate_info, commandBuffers.data());
    }
};

// Like std::vector. Has its own VkDeviceMemory. Uses a staging buffer to copy memory from host to GPU high-performance memory
template<typename T>
struct StandaloneBuffer {
    VkBuffer buffer{};
    VkDeviceMemory bufferMemory{};

	VkDeviceAddress bufferAddress{}; // Address of the buffer in GPU memory.

    VkBuffer stagingBuffer{};
	VkDeviceMemory stagingBufferMemory{}; // Staging memory on CPU (same size as buffer and memory on GPU)

    VkDeviceSize alignment;
    VkDeviceSize capacity;
	uint32_t numElements = 0; // Number of elements in this buffer

    // stuff you need to send to pipeline creation:
    VkDescriptorSetLayoutBinding binding{};
    VkWriteDescriptorSet wrt_desc_set{};
    VkDescriptorBufferInfo desc_buf_info{};
    uint32_t bindingIndex;

	Allocator* allocator;

	VkShaderStageFlagBits flags = VK_SHADER_STAGE_COMPUTE_BIT;

    void* memMap;

	// Signal to all resources using this buffer. The current descriptor set is invalid and needs to be updated and this varieable needs to be set to false
	Signal<10> descUpdateQueued;
    
    // Copy assignment operator
    /*
    StandaloneBuffer& operator=(const StandaloneBuffer& other) {
        if (this != &other) {
            buffer = other.buffer;
            bufferMemory = other.bufferMemory;
            stagingBuffer = other.stagingBuffer;
            stagingBufferMemory = other.stagingBufferMemory;
            alignment = other.alignment;
            capacity = other.capacity;
            numElements = other.numElements;
            binding = other.binding;
            wrt_desc_set = other.wrt_desc_set;
            desc_buf_info = other.desc_buf_info;
            bindingIndex = other.bindingIndex;
            allocator = other.allocator;
            flags = other.flags;
            memMap = other.memMap;
            descUpdateQueued = other.descUpdateQueued;
        }
        return *this;
    }
    */

    // Copy constructor
    StandaloneBuffer(const StandaloneBuffer& other)
        : buffer(other.buffer),
        bufferMemory(other.bufferMemory),
        stagingBuffer(other.stagingBuffer),
        stagingBufferMemory(other.stagingBufferMemory),
        alignment(other.alignment),
        capacity(other.capacity),
        numElements(other.numElements),
        binding(other.binding),
        wrt_desc_set(other.wrt_desc_set),
        desc_buf_info(other.desc_buf_info),
        bindingIndex(other.bindingIndex),
        allocator(other.allocator),
        flags(other.flags),
        memMap(other.memMap),
        descUpdateQueued(other.descUpdateQueued),
        bufferAddress(other.bufferAddress){
    }

    ~StandaloneBuffer(){
        allocator->killMemory(buffer, bufferMemory);
        allocator->killMemory(stagingBuffer,  stagingBufferMemory);
    }

    StandaloneBuffer(size_t numElements, Allocator* allocator, VkShaderStageFlagBits flags = VK_SHADER_STAGE_COMPUTE_BIT) : allocator(allocator), flags(flags), numElements(numElements) {
        init();
    }

    StandaloneBuffer(std::vector<T>& data, Allocator* allocator, VkShaderStageFlagBits flags = VK_SHADER_STAGE_COMPUTE_BIT) : allocator(allocator), flags(flags) {
		numElements = static_cast<uint32_t>(data.size());
        init();
        alloc(data);
    }

    StandaloneBuffer() : allocator(nullptr) {}

    void init() {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(allocator->init->device.physical_device, &props);
        alignment = props.limits.minStorageBufferOffsetAlignment;

        capacity = numElements * sizeof(T);
        capacity = (capacity + alignment - 1) & ~(alignment - 1);

        // Create the staging buffer
        auto stageBuff = allocator->createBuffer(capacity, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        stagingBuffer = stageBuff.first;
        stagingBufferMemory = stageBuff.second;

        // Create the buffer
        auto buff = allocator->createBuffer(capacity, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        buffer = buff.first;
        bufferMemory = buff.second;
        allocator->init->disp.mapMemory(stagingBufferMemory, 0, capacity, 0, &memMap);

		VkBufferDeviceAddressInfoEXT bufferInfo{};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_EXT;
		bufferInfo.buffer = buffer;
		bufferInfo.pNext = nullptr;

		bufferAddress = allocator->init->disp.getBufferDeviceAddress(&bufferInfo);
    }

    template<typename U>
    void addListener(U* listener){
		descUpdateQueued.addListener(listener);
    }

    void set(int idx, const T& data) {  
        if (memMap == nullptr) {  
            std::cerr << "memMap is null, cannot write data to buffer\n";  
            return;  
        }  
        if (idx < 0 || idx * sizeof(T) >= capacity) {
            std::cerr << "Index out of bounds: " << idx << "\n";  
            return;  
        }  
        std::memcpy(memMap, &data, sizeof(T));  

        allocator->copyBuffer(stagingBuffer, buffer, sizeof(T), 0, idx * sizeof(T), true);  

        std::memset(memMap, 0, sizeof(T));  
    }

	T get(const uint32_t idx) const {
		if (memMap == nullptr) {
			std::cerr << "memMap is null, cannot read data from buffer\n";
			return T{};
		}
		if (idx < 0 || idx >= numElements) {
			std::cerr << "Index out of bounds: " << idx << "\n";
			return T{};
		}
		// download data from gpu
		allocator->copyBuffer(buffer, stagingBuffer, sizeof(T), idx * sizeof(T), 0, true);
        auto data = *reinterpret_cast<T*>(static_cast<char*>(memMap));
        std::memset(memMap, 0, capacity);
        return data;
	}

    void push_back(const T& data) {
        if (numElements * sizeof(T) >= capacity) {
            grow(2); // double the capacity
        }
        set(numElements, data);
        numElements++;
    }
    
	T pop_back() {
		if (numElements == 0) {
			std::cerr << "Buffer is empty, cannot pop back\n";
			return T{};
		}
		numElements--;
		T data = get(numElements);
		set(numElements, T{}); // clear the last element
		return data;
	}

	void erase(int idx) {
		if (idx < 0 || idx >= numElements) {
			std::cerr << "Index out of bounds: " << idx << "\n";
			return;
		}
		allocator->freeMemory(buffer, idx * sizeof(T), sizeof(T));
		numElements--;
	}

	void insert(int idx, T& data) {
		if (idx < 0 || idx > numElements) {
			std::cerr << "Index out of bounds: " << idx << "\n";
			return;
		}
		if (numElements * sizeof(T) >= capacity) {
			grow(2); // double the capacity
		}

		std::memcpy(memMap, &data, sizeof(T));
        allocator->insertMemory(buffer, stagingBuffer, idx * sizeof(T), sizeof(T));
		std::memset(memMap, 0, sizeof(T));
		numElements++;
	}

    void replace(int idx, T& data) {
        if (idx < 0 || idx >= numElements) {
            std::cerr << "Index out of bounds: " << idx << "\n";
            return;
        }
		set(idx, data);
    }

    // Allocate data into the staging buffer only. Does not copy to GPU buffer.
    void alloc_cpu(const std::vector<T>& data){
        auto sizeOfData = static_cast<uint32_t>(sizeof(T) * data.size());
        VkMemoryRequirements memRequirements{};
        allocator->init->disp.getBufferMemoryRequirements(stagingBuffer, &memRequirements);
        auto stagingBufferSize = memRequirements.size;
        if( stagingBufferSize < sizeOfData) {
            allocator->init->disp.unmapMemory(stagingBufferMemory);
            allocator->killMemory(stagingBuffer, stagingBufferMemory);
            auto stageBuff = allocator->createBuffer(sizeOfData, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            stagingBuffer = stageBuff.first;
            stagingBufferMemory = stageBuff.second;
            allocator->init->disp.mapMemory(stagingBufferMemory, 0, sizeOfData, 0, &memMap);
        }
        std::memcpy(memMap, data.data(), sizeOfData);
    }

    // Allocate data into the staging buffer only. Does not copy to GPU buffer.
    void alloc_cpu(const T* data, size_t size){
        auto sizeOfData = sizeof(T) * size;
        VkMemoryRequirements memRequirements{};
        allocator->init->disp.getBufferMemoryRequirements(stagingBuffer, &memRequirements);
        auto stagingBufferSize = memRequirements.size;
        if( stagingBufferSize < sizeOfData) {
            allocator->init->disp.unmapMemory(stagingBufferMemory);
            allocator->killMemory(stagingBuffer, stagingBufferMemory);
            auto stageBuff = allocator->createBuffer(sizeOfData, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            stagingBuffer = stageBuff.first;
            stagingBufferMemory = stageBuff.second;
            allocator->init->disp.mapMemory(stagingBufferMemory, 0, sizeOfData, 0, &memMap);
        }
        std::memcpy(memMap, data, sizeOfData);
    }

    // Allocate data into the buffer. Each alloc will overwrite the previous data in the buffer.
    void alloc(const std::vector<T>& data) {
        auto sizeOfData = static_cast<uint32_t>(sizeof(T) * data.size());
		if (sizeOfData > capacity) {
			std::cout << "Data size exceeds buffer capacity. Growing buffer...\n";
			growUntil(2, data.size());
		}

        VkMemoryRequirements memRequirements{};
        allocator->init->disp.getBufferMemoryRequirements(stagingBuffer, &memRequirements);
        auto stagingBufferSize = memRequirements.size;

        // Copy Data to Staging Buffer
        if (stagingBufferSize < sizeOfData) {
            allocator->init->disp.unmapMemory(stagingBufferMemory);
            allocator->killMemory(stagingBuffer, stagingBufferMemory);
            auto stageBuff = allocator->createBuffer(sizeOfData, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            stagingBuffer = stageBuff.first;
            stagingBufferMemory = stageBuff.second;
            allocator->init->disp.mapMemory(stagingBufferMemory, 0, sizeOfData, 0, &memMap);
        }

        std::memcpy(memMap, data.data(), sizeOfData);
        allocator->copyBuffer(stagingBuffer, buffer, sizeOfData, 0, 0, true);
		std::memset(memMap, 0, sizeOfData);

		numElements = static_cast<uint32_t>(data.size());
    }

    void alloc(const T* data, size_t size){
        if (size > capacity / sizeof(T)) {
            std::cout << "Data size exceeds buffer capacity. Growing buffer...\n";
            growUntil(2, static_cast<uint32_t>(size));
        }
        VkMemoryRequirements memRequirements{};
        allocator->init->disp.getBufferMemoryRequirements(stagingBuffer, &memRequirements);
        auto stagingBufferSize = memRequirements.size;
        // Copy Data to Staging Buffer
        auto sizeOfData = sizeof(T) * size;
        if (stagingBufferSize < sizeOfData) {
            allocator->init->disp.unmapMemory(stagingBufferMemory);
            allocator->killMemory(stagingBuffer, stagingBufferMemory);
            auto stageBuff = allocator->createBuffer(sizeOfData, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            stagingBuffer = stageBuff.first;
            stagingBufferMemory = stageBuff.second;
            allocator->init->disp.mapMemory(stagingBufferMemory, 0, sizeOfData, 0, &memMap);
        }
        std::memcpy(memMap, data, sizeOfData);
        allocator->copyBuffer(stagingBuffer, buffer, sizeOfData, 0, 0, true);
        std::memset(memMap, 0, sizeOfData);
        numElements = static_cast<uint32_t>(size);
    }
	
	// Allocate data into the buffer. Each alloc will overwrite the previous data in the buffer.
    void alloc(T& data) {
        auto sizeOfData = sizeof(T);
        if (sizeOfData > capacity) {
            std::cout << "Data size exceeds buffer capacity. Growing buffer...\n";
            growUntil(2, 1);
        }
        VkMemoryRequirements memRequirements{};
        allocator->init->disp.getBufferMemoryRequirements(stagingBuffer, &memRequirements);
        auto stagingBufferSize = memRequirements.size;
        // Copy Data to Staging Buffer
        if (stagingBufferSize < sizeOfData) {
            allocator->init->disp.unmapMemory(stagingBufferMemory);
            allocator->killMemory(stagingBuffer, stagingBufferMemory);
            auto stageBuff = allocator->createBuffer(sizeOfData, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            stagingBuffer = stageBuff.first;
            stagingBufferMemory = stageBuff.second;
            allocator->init->disp.mapMemory(stagingBufferMemory, 0, sizeOfData, 0, &memMap);
        }
        std::memcpy(memMap, &data, sizeOfData);
        allocator->copyBuffer(stagingBuffer, buffer, sizeOfData, 0, 0, true);
    }

    // Copy whatever is in the staging buffer to the GPU buffer. Overwrites all data in the GPU buffer.
    void to_gpu() {
        // check if staging buffer size = buffer size
        VkMemoryRequirements memRequirements{};
        allocator->init->disp.getBufferMemoryRequirements(stagingBuffer, &memRequirements);
        auto stagingBufferSize = memRequirements.size;
        if (stagingBufferSize < capacity) {
            auto newStageBuff = allocator->createBuffer(capacity, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            allocator->copyBuffer(stagingBuffer, newStageBuff.first, stagingBufferSize, 0, 0, true);
            allocator->init->disp.unmapMemory(stagingBufferMemory);
            allocator->killMemory(stagingBuffer, stagingBufferMemory);
            stagingBuffer = newStageBuff.first;
            stagingBufferMemory = newStageBuff.second;
            allocator->init->disp.mapMemory(stagingBufferMemory, 0, capacity, 0, &memMap);
        }

        allocator->copyBuffer(stagingBuffer, buffer, capacity, 0, 0, true);
    }

    void fill_buffer(T data){
        allocator->fillBuffer(buffer, bufferMemory, data, 0, capacity);
    }

    VkDeviceAddress getBufferAddress() {
        VkBufferDeviceAddressInfoEXT bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_EXT;
        bufferInfo.buffer = buffer;
        bufferInfo.pNext = nullptr;

        bufferAddress = allocator->init->disp.getBufferDeviceAddress(&bufferInfo);
        return bufferAddress;
	}

    void createDescriptors(int idx, uint32_t stage) {
        binding.binding = stage;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binding.descriptorCount = 1;
        binding.stageFlags = stage;
        binding.pImmutableSamplers = nullptr;
    }

    void createDescriptors() {
        binding.binding = bindingIndex;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binding.descriptorCount = 1;
        binding.stageFlags = flags;
        binding.pImmutableSamplers = nullptr;
    }

    void updateDescriptorSet(VkDescriptorSet& set) {
        desc_buf_info.buffer = buffer;
        desc_buf_info.offset = 0;
        desc_buf_info.range = VK_WHOLE_SIZE;

        wrt_desc_set.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wrt_desc_set.dstSet = set;
        wrt_desc_set.dstBinding = bindingIndex;
        wrt_desc_set.dstArrayElement = 0;
        wrt_desc_set.descriptorCount = 1;
        wrt_desc_set.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wrt_desc_set.pBufferInfo = &desc_buf_info;
    }

    void updateDescriptorSet(VkDescriptorSet& set, int arrayElement, int idx) {
        desc_buf_info.buffer = buffer;
        desc_buf_info.offset = 0;
        desc_buf_info.range = VK_WHOLE_SIZE;

        wrt_desc_set.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wrt_desc_set.dstSet = set;
        wrt_desc_set.dstBinding = idx;
        wrt_desc_set.dstArrayElement = arrayElement;
        wrt_desc_set.descriptorCount = 1;
        wrt_desc_set.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wrt_desc_set.pBufferInfo = &desc_buf_info;
    }

    void clearBuffer() {
        if (allocator != nullptr) {
            allocator->fillBuffer(buffer, bufferMemory, 0, 0, capacity);
        }
    }

    void grow(int factor) {
		auto oldCapacity = capacity;
        capacity = factor * sizeof(T);
		capacity = (capacity + alignment - 1) & ~(alignment - 1);

		auto [buf, mem] = allocator->createBuffer(capacity, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

		allocator->copyBuffer(buffer, buf, oldCapacity, 0, 0, true);
		allocator->killMemory(buffer, bufferMemory);
		buffer = buf;
		bufferMemory = mem;

		getBufferAddress();

        // the internal handles were changed. We need descriptor updates
        descUpdateQueued.trigger();
    }

    // newSize is element size, not byte size
    void growUntil(int factor, uint32_t newSize) {
        auto oldCapacity = capacity;
        while (capacity < newSize * sizeof(T)) {
            capacity *= factor;
            capacity = (capacity + alignment - 1) & ~(alignment - 1);
        }
		auto [buf, mem] = allocator->createBuffer(capacity, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, true);
		
        allocator->copyBuffer(buffer, buf, oldCapacity, 0, 0, true);
		allocator->killMemory(buffer, bufferMemory);
		buffer = buf;
		bufferMemory = mem;

		getBufferAddress();
		// the internal handles were changed. We need descriptor updates
		descUpdateQueued.trigger();
    }

    // newSize is size of buffer in elements, NOT bytes
    void resize(size_t newSize) {
        auto prevCapacity = capacity;
		capacity = newSize * sizeof(T);
		capacity = (capacity + alignment - 1) & ~(alignment - 1);
        
		auto [buf, mem] = allocator->createBuffer(capacity, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, true);
        allocator->copyBuffer(buffer, buf, prevCapacity, 0, 0, true);
		allocator->killMemory(buffer, bufferMemory);
        
		auto [stagingBuf, stagingMem] = allocator->createBuffer(capacity, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, true);
		allocator->init->disp.unmapMemory(stagingBufferMemory);
		allocator->killMemory(stagingBuffer, stagingBufferMemory);

		stagingBuffer = stagingBuf;
		stagingBufferMemory = stagingMem;
		allocator->init->disp.mapMemory(stagingBufferMemory, 0, capacity, 0, &memMap);

		buffer = buf;
		bufferMemory = mem;
		getBufferAddress();
		// the internal handles were changed. We need descriptor updates
		descUpdateQueued.trigger();
    }

    size_t size() const {
        return capacity / sizeof(T);
    }

    std::vector<T> downloadBuffer() {
        
		VkMemoryRequirements memRequirements;
		allocator->init->disp.getBufferMemoryRequirements(buffer, &memRequirements);
		auto bufferSize = memRequirements.size;
		auto numElements = static_cast<uint32_t>(bufferSize / sizeof(T));
        allocator->copyBuffer(buffer, stagingBuffer, bufferSize, 0, 0, true);

		std::vector<T> data; 
        data.resize(numElements);
        std::memcpy(data.data(), memMap, bufferSize);
		std::memset(memMap, 0, bufferSize);
		return data;
    }

    /*
    void operator=(const StandaloneBuffer& other) const {
        if (this != &other) {
            allocator->init->disp.unmapMemory(stagingBufferMemory);
            allocator->killMemory(buffer, bufferMemory);
            allocator->killMemory(stagingBuffer, stagingBufferMemory);
            buffer = other.buffer;
            bufferMemory = other.bufferMemory;
            stagingBuffer = other.stagingBuffer;
            stagingBufferMemory = other.stagingBufferMemory;
            alignment = other.alignment;
            capacity = other.capacity;
            numElements = other.numElements;
            binding = other.binding;
            wrt_desc_set = other.wrt_desc_set;
            desc_buf_info = other.desc_buf_info;
            bindingIndex = other.bindingIndex;
            allocator = other.allocator;
            flags = other.flags;
            memMap = other.memMap;
        }
    }
    */
};

// Bindless descriptor array of standaloneBuffers
template<typename bufferType>
struct BufferArray {
    std::vector<StandaloneBuffer<bufferType>> buffers;

    uint32_t numBuffers = 1000;             // by default, 1000 buffers are supported

	VkDescriptorSet descSet;
	VkDescriptorSetLayout descSetLayout;
	VkDescriptorPool descPool;
	Allocator* allocator;

    uint32_t bindingIndex = 0;

	BufferArray(Allocator* allocator, uint32_t bindingIndex) : allocator(allocator), bindingIndex(bindingIndex) {
        createDescriptorPool();
		createDescSetLayout();
		allocateDescSet();
        allocator->init->addObject(this);
	}

    BufferArray(Allocator* allocator, uint32_t bindingIndex, VkDescriptorPool sharedPool) : allocator(allocator), bindingIndex(bindingIndex), descPool(sharedPool) {
        createDescSetLayout();
		allocateDescSet();
    };

    BufferArray() {};

    ~BufferArray() {
        allocator->init->disp.destroyDescriptorSetLayout(descSetLayout, nullptr);
        allocator->init->disp.destroyDescriptorPool(descPool, nullptr);
    }

	void push_back(StandaloneBuffer<bufferType>& buffer) {
		if (buffers.size() >= numBuffers) {
			std::cout << "BufferArray: reached max number of buffers. Cannot add more.\n";
			return;
		}
		buffers.push_back(std::move(buffer));
	}

	void push_back(std::vector<bufferType>& data) {
		if (buffers.size() >= numBuffers) {
			std::cout << "BufferArray: reached max number of buffers. Cannot add more.\n";
			return;
		}
		buffers.emplace_back(data, allocator);
	}

	void erase(uint32_t idx) {
		if (idx < 0 || idx >= buffers.size()) {
			std::cout << "BufferArray: index out of range. Cannot erase.\n";
			return;
		}
		allocator->init->disp.destroyBuffer(buffers[idx].buffer, nullptr);
		allocator->init->disp.freeMemory(buffers[idx].bufferMemory, nullptr);
		buffers.erase(buffers.begin() + idx);
	}

	size_t size() const {
		return buffers.size();
	}

    void createDescriptorPool() {
        VkDescriptorPoolSize pool_sizes_bindless[] =
        {
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, numBuffers }
        };

        VkDescriptorPoolCreateInfo poolInfo = {};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = pool_sizes_bindless;
        poolInfo.maxSets = 1;
        poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
        allocator->init->disp.createDescriptorPool(&poolInfo, nullptr, &descPool);
    }

    void createDescSetLayout() {
        VkDescriptorBindingFlags bindless_flags = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT_EXT | VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT_EXT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT_EXT;

        VkDescriptorSetLayoutBinding vk_binding;
        vk_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        vk_binding.descriptorCount = numBuffers;
        vk_binding.binding = 0;

        vk_binding.stageFlags = VK_SHADER_STAGE_ALL;
        vk_binding.pImmutableSamplers = nullptr;

        VkDescriptorSetLayoutCreateInfo layout_info = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        layout_info.bindingCount = 1;
        layout_info.pBindings = &vk_binding;
        layout_info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT_EXT;

        VkDescriptorSetLayoutBindingFlagsCreateInfoEXT extended_info{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO_EXT, nullptr };
        extended_info.bindingCount = 1;
        extended_info.pBindingFlags = &bindless_flags;

        layout_info.pNext = &extended_info;

        allocator->init->disp.createDescriptorSetLayout(&layout_info, nullptr, &descSetLayout);
    }

    void allocateDescSet() {
        VkDescriptorSetAllocateInfo alloc_info{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        alloc_info.descriptorPool = descPool;
        alloc_info.descriptorSetCount = 1;
        alloc_info.pSetLayouts = &descSetLayout;

        VkDescriptorSetVariableDescriptorCountAllocateInfoEXT count_info{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO_EXT };
        uint32_t max_binding = numBuffers - 1;
        count_info.descriptorSetCount = 1;
        // This number is the max allocatable count
        count_info.pDescriptorCounts = &max_binding;
        alloc_info.pNext = &count_info;

        allocator->init->disp.allocateDescriptorSets(&alloc_info, &descSet);
    }

    void updateDescriptorSets() {
        std::vector<VkWriteDescriptorSet> writes(buffers.size());
        for (size_t i = 0; i < buffers.size(); ++i) {
            buffers[i].createDescriptors(0, VK_SHADER_STAGE_ALL);
            buffers[i].updateDescriptorSet(descSet, i, bindingIndex);
            writes[i] = buffers[i].wrt_desc_set;
        }
        allocator->init->disp.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }
};

// an element inside the MemPool.
template<typename T>
struct Buffer {
    VkBuffer buffer;                      // Points to the MemPool's buffer
    VkDeviceSize offset = 0;                  // Byte Offset within the buffer
	uint32_t elementOffset = 0;               // Element Offset within the buffer
    uint32_t numElements = 0;                 // Number of elements in this buffer
	VkDeviceSize alignedSize(VkDeviceSize alignment) { return (numElements * sizeof(T) + alignment - 1) & ~(alignment - 1); } // Return the aligned byte size of this buffer element
	VkDeviceSize size() { return numElements * sizeof(T); } // Return the byte size of this buffer element

    // Descriptor set members (used when MemPool is not using bypassDescriptors)
    uint32_t bindingIndex = 0;
    VkDescriptorSetLayoutBinding binding{};
    VkWriteDescriptorSet wrt_desc_set{};
    VkDescriptorBufferInfo desc_buf_info{};

    void createDescriptors(uint32_t bindingIdx, VkShaderStageFlags flags= VK_SHADER_STAGE_COMPUTE_BIT) {
        bindingIndex = bindingIdx;
        binding.binding = bindingIndex;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binding.descriptorCount = 1;
        binding.stageFlags = flags;
        binding.pImmutableSamplers = nullptr;
    }

    void updateDescriptorSet(VkDescriptorSet& set) {
        desc_buf_info.buffer = buffer;
        desc_buf_info.offset = offset;
        desc_buf_info.range = numElements * sizeof(T);

        wrt_desc_set.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wrt_desc_set.dstSet = set;
        wrt_desc_set.dstBinding = bindingIndex;
        wrt_desc_set.dstArrayElement = 0;
        wrt_desc_set.descriptorCount = 1;
        wrt_desc_set.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wrt_desc_set.pBufferInfo = &desc_buf_info;
    }
};

// like std::vector<std::vector<T>>, but worse. And for the GPU. Only handles Storage buffers. For Uniform buffers, use StandaloneBuffer
// Each input is a seperate descriptor binding, but same set.
// Can update buffer offsets and even buffer handles if defragmented or resized. Shoots a Signal struct to all using resources. Resources using this pool must register themselves as a 
// signal listener and submit their onSignal() function ptr to the pool's Signal. When internal descriptors are updated, you need to pause your pipeline's execution and update descriptors
// This is useful for dynamic memory allocation where all scene mesh data is allocated together, in one MemPool (as shown in VkCalcium.hpp). 
// You shouldn't need to use any defragmentation or resize operations if you're using this for training Machine Learning tensors.
template<typename T>
struct MemPool {
    VkBuffer buffer;               // Single buffer for all allocations on GPU
    VkDeviceMemory memory;         // Backing memory on GPU
	VkDeviceAddress poolAddress;   // Address of the buffer in GPU memory.

    // persistent staging buffer for efficiency. We should avoid allocating new memory whenever we can.
    // Not really the best solution, cause for every MemPool, 
    // we now have two memories of the same size that's taking up space in the computer... but eeh
    VkBuffer stagingBuffer;                  // Staging buffer on CPU (same size as buffer and memory on GPU)
    VkDeviceMemory stagingMemory;            // Staging memory on CPU (same size as buffer and memory on GPU)

    VkDeviceAddress bufferDeviceAddress; // Actual pointer to the memory inside of the GPU. used when bypassDescriptors are activated

    void* mapped = nullptr;        // Mapped pointer to the staging buffer. This stays mapped until MemPool is destroyed.

    VkDeviceSize alignment;        // Buffer alignment requirement
    VkDeviceSize capacity;         // Total capacity in bytes
    VkDeviceSize offset = 0;       // Current allocation byte offset
	uint32_t elementOffset = 0;    // Current element offset (in number of elements)
    VkDeviceSize occupied = 0;     // The amount of occupied memory in the buffer (its value of the same as current allocation offset, but I kept it seperate for readability purposes)

    std::vector<Buffer<T>> buffers; // Track all allocated buffers
    VkDeviceSize deadMemory = 0;
    std::vector<std::pair<VkDeviceSize, VkDeviceSize>> gaps; // track all gaps in the memory

    Allocator* allocator;           // The allocator has all the vulkan context stuff and it handles buffer creation, destruction, manipulation, etc.

    VkShaderStageFlagBits flags = VK_SHADER_STAGE_COMPUTE_BIT;

    Signal<10> descUpdateQueued;    // by default, max 10 listeners. Meaning, 10 user resources can register to listen for descriptor updates

    // maxElements means number of individual elements of type T inside of all the buffers. 
    // Meaning, if maxElements is set to 10, pool can sustain 10 buffers with 1 float each (if T = float).
    // But only 2 buffers with 5 floats each.
    MemPool(uint32_t maxElements, Allocator* allocator = nullptr, VkShaderStageFlagBits flags = VK_SHADER_STAGE_COMPUTE_BIT) : allocator(allocator), flags(flags) {
        // Query alignment requirement for storage buffers
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(allocator->init->device.physical_device, &props);
        alignment = props.limits.minStorageBufferOffsetAlignment;

        if (maxElements == 0) {
            std::cerr << "maxElements cannot be 0" << std::endl;
            return;
        }

        // Calculate total size with alignment
        capacity = maxElements * sizeof(T);
        capacity = (capacity + alignment - 1) & ~(alignment - 1);

        // Create CPU staging buffer (CPU visible, low-performance)
        auto stageBuff = allocator->createBuffer(capacity, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        stagingBuffer = stageBuff.first;
        stagingMemory = stageBuff.second;

        // Map the staging buffer memory
        allocator->init->disp.mapMemory(stagingMemory, 0, capacity, 0, &mapped);

        auto buff = allocator->createBuffer(capacity,
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        buffer = buff.first;
        memory = buff.second;

		VkBufferDeviceAddressInfoEXT bufferInfo{};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_EXT;
		bufferInfo.buffer = buffer;
		bufferInfo.pNext = nullptr;
		poolAddress = allocator->init->disp.getBufferDeviceAddress(&bufferInfo);
    }

    ~MemPool(){
        allocator->killMemory(buffer, memory);
        allocator->killMemory(stagingBuffer, stagingMemory);
    }

    MemPool() {};

    size_t size() {
        return buffers.size();
    }
    
    template<typename U>
    void addListener(U* listener) {
		descUpdateQueued.addListener(listener);
    }

	VkDeviceAddress getBufferAddress() {
		VkBufferDeviceAddressInfoEXT bufferInfo{};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_EXT;
		bufferInfo.buffer = buffer;
		bufferInfo.pNext = nullptr;
		poolAddress = allocator->init->disp.getBufferDeviceAddress(&bufferInfo);
		return poolAddress;
	}

    // simple push_back operation. Quite expensive, use with caution.
    bool push_back(const std::vector<T>& data, bool autoBind = true) {
        auto bindingIndex = buffers.size();
        const VkDeviceSize dataSize = data.size() * sizeof(T);
        const VkDeviceSize alignedSize = (dataSize + alignment - 1) & ~(alignment - 1);
        occupied += alignedSize;
        
        // try to find a gap in which we could insert this buffer into, then either update that gap or destroy it
        uint32_t allocationOffset = offset;
        bool used_gap = false;
        for (uint32_t i = 0; i < gaps.size(); i++){
            auto& [ofse, siz] = gaps[i];
            if (siz >= dataSize){
                used_gap = true;
                allocationOffset = ofse;
                uint32_t remaining_size = dataSize - siz;
                deadMemory -= dataSize;
                if(remaining_size > 0){
                    gaps[i].first = ofse + dataSize;
                    gaps[i].second = remaining_size;
                }else {
                    gaps.erase(gaps.begin() + i);
                }
                break;
            }
        }

        // Check if there's enough space
        if (offset + alignedSize > capacity && !used_gap) {
            growUntil(2, offset + alignedSize);
        }

        VkMemoryRequirements memRequirements{};
        allocator->init->disp.getBufferMemoryRequirements(stagingBuffer, &memRequirements);
		auto stagingBufferSize = memRequirements.size;

        // Copy Data to Staging Buffer
		if (stagingBufferSize < alignedSize) {
            allocator->init->disp.unmapMemory(stagingMemory);
			allocator->killMemory(stagingBuffer, stagingMemory);
			auto stageBuff = allocator->createBuffer(alignedSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
			stagingBuffer = stageBuff.first;
			stagingMemory = stageBuff.second;
			allocator->init->disp.mapMemory(stagingMemory, 0, alignedSize, 0, &mapped);
		}
        std::memcpy(mapped, data.data(), dataSize);

        // Copy Staging Buffer to GPU Buffer
        allocator->copyBuffer(stagingBuffer, buffer, alignedSize, 0, allocationOffset, true);
        // Clear the staging buffer memory
        std::memset(mapped, 0, dataSize);

        // Create a Buffer entry (Created on the heap, because stack memory apparently overflows)
        auto newBuffer = new Buffer<T>();
        newBuffer->buffer = buffer;
        newBuffer->offset = allocationOffset;
		newBuffer->elementOffset = static_cast<uint32_t>(allocationOffset / sizeof(T));
        newBuffer->numElements = static_cast<uint32_t>(data.size());
        newBuffer->createDescriptors(bindingIndex, flags);
        // dereference ptr and push copy into the vector
        buffers.push_back(*newBuffer);
		delete newBuffer; // Free the heap memory

        // Update offset for next allocation, if we didn't allocate inside a gap
        if(!used_gap){
            offset += alignedSize;
            elementOffset += static_cast<uint32_t>(data.size());
        }

        return true;
    }

    // the standaloneBuffer still lives after this. If you want to dispose of it, you'll have to call its destructor manually
    bool push_back(StandaloneBuffer<T>& data) {
        auto bindingIndex = buffers.size();
        const VkDeviceSize dataSize = data.capacity;
        const VkDeviceSize alignedSize = (dataSize + alignment - 1) & ~(alignment - 1);
        occupied += alignedSize;
        
        // try to find a gap in which we could insert this buffer into, then either update that gap or destroy it
        uint32_t allocationOffset = offset;
        bool used_gap = false;
        for (uint32_t i = 0; i < gaps.size(); i++){
            auto& [ofse, siz] = gaps[i];
            if (siz >= dataSize){
                used_gap = true;
                allocationOffset = ofse;
                uint32_t remaining_size = siz - dataSize;
                deadMemory -= dataSize;
                if(remaining_size > 0){
                    gaps[i].first = ofse + dataSize;
                    gaps[i].second = remaining_size;
                }else {
                    gaps.erase(gaps.begin() + i);
                }
                break;
            }
        }

        // Check if there's enough space
        if (offset + alignedSize > capacity && !used_gap) {
            growUntil(2, offset + alignedSize);
        }

        // Copy given buffer to GPU Buffer
        allocator->copyBuffer(data.buffer, buffer, dataSize, 0, allocationOffset, true);

        // Create a Buffer entry
        auto newBuffer = new Buffer<T>();
        newBuffer->buffer = buffer;
        newBuffer->offset = allocationOffset;
        newBuffer->elementOffset = static_cast<uint32_t>(allocationOffset / sizeof(T));
        newBuffer->numElements = static_cast<uint32_t>(data.capacity / sizeof(T));
        newBuffer->createDescriptors(bindingIndex, flags);
        buffers.push_back(*newBuffer);
        delete newBuffer; // Free the heap memory
        
        // Update offset for next allocation, if we didn't allocate inside a gap
        if(!used_gap){
            offset += alignedSize;
            elementOffset += static_cast<uint32_t>(data.capacity / sizeof(T));
        }
        
        return true;
    }

    // Retrieve data from GPU
    std::vector<T> operator[](size_t index) {
        if (index >= buffers.size() || index < 0) {
            throw std::out_of_range("Index out of range");
        }

        Buffer<T>& buf = buffers[index];
        VkDeviceSize dataSize = buf.numElements * sizeof(T);
        const VkDeviceSize alignedSize = (dataSize + alignment - 1) & ~(alignment - 1);

        // Copy Data from GPU Buffer to Staging Buffer
        allocator->copyBuffer(buffer, stagingBuffer, dataSize, buf.offset, 0);

        std::vector<T> output(buf.numElements);
        std::memcpy(output.data(), mapped, dataSize);
        std::memset(mapped, 0, alignedSize);

        return output;
    }

    // resizes the memPool to the given newSize. This doesn't destroy data in the original buffer.
    void resize(int newSize) {
        // Calculate total size with alignment
        auto oldCapacity = capacity;
        capacity = newSize * sizeof(T);
        capacity = (capacity + alignment - 1) & ~(alignment - 1);

        auto newBuffer = allocator->createBuffer(capacity,
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        allocator->copyBuffer(buffer, newBuffer.first, oldCapacity, 0, 0, true);

		allocator->killMemory(buffer, memory);

        buffer = newBuffer.first;
        memory = newBuffer.second;

        for (auto& buffer : buffers) {
            buffer.buffer = newBuffer.first;
        }
		getBufferAddress();
        // Internal handles were changed. We need descriptor updates
        descUpdateQueued.trigger();
    }

    // Grow the buffer to a new size. This will not change the data inside of the buffer.
    // Very expensive Operation. Try to set the MemPool's size at the beginning of it's life, that way there's no need for growth.
    void grow(int factor) {
        // Calculate total size with alignment
        auto oldCapacity = capacity;

        capacity *= factor;
        capacity = (capacity + alignment - 1) & ~(alignment - 1);

        // Notice how we don't grow the staging buffer. This can cause problems when trying to download the buffer to host, but eeh
        auto newBuffer = allocator->createBuffer(capacity,
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        allocator->copyBuffer(buffer, newBuffer.first, oldCapacity, 0, 0, true);
		allocator->killMemory(buffer, memory);

        buffer = newBuffer.first;
        memory = newBuffer.second;

        for (auto& buffer : buffers) {
            buffer.buffer = newBuffer.first;
        }
        getBufferAddress();

        // Internal handles were changed. We need descriptor updates
        descUpdateQueued.trigger();
    }

    void growUntil(int factor, VkDeviceSize finalSize) {
        auto oldCapacity = capacity;
        
        while (capacity < finalSize) {
            capacity *= factor;
            capacity = (capacity + alignment - 1) & ~(alignment - 1);
        }
        
        // Notice how we don't grow the staging buffer. This can cause problems when trying to download the buffer to host, but eeh
        auto newBuffer = allocator->createBuffer(capacity,
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        allocator->copyBuffer(buffer, newBuffer.first, oldCapacity, 0, 0, true);

		allocator->killMemory(buffer, memory);

        buffer = newBuffer.first;
        memory = newBuffer.second;
        
        for (auto& buffer : buffers) {
            buffer.buffer = newBuffer.first;
        }
        getBufferAddress();
        descUpdateQueued.trigger();
    }

    // ez operation.
    std::vector<T> pop_back() {
        if (buffers.empty()) {
            throw std::out_of_range("No buffers to pop");
        }
        auto buf = buffers.back();
        auto dataSize = buf.size();
        auto alignedSize = buf.alignedSize(alignment);

        allocator->copyBuffer(buffer, stagingBuffer, dataSize, buf.offset, 0);

        auto data = std::vector<T>(buf.numElements);
        std::memcpy(data.data(), mapped, dataSize);
        std::memset(mapped, 0, dataSize);
        buffers.pop_back();
        offset -= alignedSize;
        occupied -= alignedSize;
        return data;
    }

    // expensive operation if instaClean is turned on. Use with caution.
    void erase(uint32_t index, bool instaClean = true) {
        if (index >= buffers.size()) {
            throw std::out_of_range("Index out of range");
        }
        if (buffers.size() <= 0) {
            throw std::runtime_error("MemPool has nothing left to erase.");
        }
        auto& buf = buffers[index];
        auto alignedSize = buf.alignedSize(alignment);
		auto elementSize = buf.numElements;
        auto dataSize = buf.size();

        // free buffer memory and manage offsets accordingly
        if (instaClean) {
            // free the blob of erased memory, push the remaining memory towards the left and zero out the tail of straggling memory
            allocator->freeMemory(buffer, buf.offset, alignedSize);
            // Update offset for next allocation
            offset -= alignedSize;
            occupied -= alignedSize;
            // Update the offset of the remaining buffers
            for (size_t i = index; i < buffers.size(); ++i) {
                buffers[i].offset -= alignedSize;
				buffers[i].elementOffset -= elementSize;
            }
        }
        // just record the gap and leave it there. The record of the gaps can be cleaned up later. Kind of like a garbage collector.
        else {
            deadMemory += alignedSize;
            gaps.push_back({buffers[index].offset, alignedSize});
        }
        // Remove the buffer from the vector
        buffers.erase(buffers.begin() + index);

        // update binding index of the remaining buffers. You need to recreate descriptor sets after this. erase is a trash operation, I know.
        for (size_t in = index; in < buffers.size(); ++in) {
            buffers[in].createDescriptors(buffers[in].bindingIndex - 1, flags);
        }

        descUpdateQueued.trigger();
    }

    // garbage collector (kinda)
    void cleanGaps() {

        if (deadMemory <= 0) { throw std::runtime_error("MemPool is already clean."); }

        // record the good stuff (aligned to vulkan's reqs)
        // offset, range format. both need to be aligned to vulkan's reqs. Automatically done by the MemPool
        std::vector<std::pair<VkDeviceSize, VkDeviceSize>> alive;
        for (auto& buffer : buffers) {
            auto alignedSize = buffer.alignedSize(alignment);
            alive.push_back({ buffer.offset, alignedSize });
        }
        // defragment the memory
        allocator->defragment(buffer, memory, alive);
        gaps.clear();
        // update the offset of the remaining buffers
        VkDeviceSize runningOffset = 0;
        uint32_t runningElementOffset = 0;
        for (auto& buffer : buffers) {
            auto alignedSize = buffer.alignedSize(alignment);
            buffer.offset = runningOffset;
			buffer.elementOffset = runningElementOffset;
            runningOffset += alignedSize;
			runningElementOffset += buffer.numElements;
        }

        // need to notify all user resources to update their descriptor sets
        descUpdateQueued.trigger();

        // decrement the offset by the amount of memory we've cleared
        offset -= deadMemory;
        occupied -= deadMemory;
    }

    // expensive operation. Use with caution. replaces element[index] with given element
    void replace(uint32_t index, const std::vector<T>& data) {
        if (index > buffers.size()) {
            throw std::out_of_range("Index out of range");
        }

        const VkDeviceSize dataSize = data.size() * sizeof(T);
        const VkDeviceSize alignedSize = (dataSize + alignment - 1) & ~(alignment - 1);
        const VkDeviceSize oldAlignedSize = buffers[index].alignedSize(alignment);
        const auto oldOffset = buffers[index].offset;
        const auto oldNumElements = buffers[index].numElements;
        const VkDeviceSize sizeDelta = alignedSize - oldAlignedSize;

        // Check if there's enough space
        if ((offset + sizeDelta) > capacity) {
            growUntil(2, offset + sizeDelta);
        }

        // Update gaps that lay beyond the buffer being replaced
        for (auto& [ofse, rnge] : gaps) {
            if (ofse > oldOffset) {
                ofse += sizeDelta;
            }
        }

        // Adjust offset and occupied by the size change
        offset += sizeDelta;
        occupied += sizeDelta;

        // Prepare staging buffer
        VkMemoryRequirements memRequirements{};
        allocator->init->disp.getBufferMemoryRequirements(stagingBuffer, &memRequirements);
        auto stagingBufferSize = memRequirements.size;

        if (stagingBufferSize < alignedSize) {
            allocator->init->disp.unmapMemory(stagingMemory);
            allocator->killMemory(stagingBuffer, stagingMemory);
            auto stageBuff = allocator->createBuffer(alignedSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            stagingBuffer = stageBuff.first;
            stagingMemory = stageBuff.second;
            allocator->init->disp.mapMemory(stagingMemory, 0, alignedSize, 0, &mapped);
        }

        std::memcpy(mapped, data.data(), dataSize);
        allocator->replaceMemory(buffer, memory, stagingBuffer, alignedSize, oldOffset);
        std::memset(mapped, 0, dataSize);

        // Edit the buffer element to reflect the new data it now represents
        buffers[index].numElements = static_cast<uint32_t>(data.size());

        // Update offsets of all subsequent buffers
        for (size_t i = index + 1; i < buffers.size(); ++i) {
            buffers[i].offset += sizeDelta;
            buffers[i].elementOffset += static_cast<uint32_t>((alignedSize - oldAlignedSize) / sizeof(T));
            buffers[i].createDescriptors(static_cast<uint32_t>(i), flags);
        }

        // internal offsets were changed. We need descriptor updates
        descUpdateQueued.trigger();
    }

    void replace(uint32_t index, const StandaloneBuffer<T>& data) {
        if (index > buffers.size()) {
            throw std::out_of_range("Index out of range");
        }

        const VkDeviceSize alignedSize = data.capacity;
        const VkDeviceSize oldAlignedSize = buffers[index].alignedSize(alignment);
        const auto oldOffset = buffers[index].offset;
        const auto oldNumElements = buffers[index].numElements;
        const VkDeviceSize sizeDelta = alignedSize - oldAlignedSize;

        // Check if there's enough space
        if ((offset + sizeDelta) > capacity) {
            growUntil(2, offset + sizeDelta);
        }

        // Update gaps that lay beyond the buffer being replaced
        for (auto& [ofse, rnge] : gaps) {
            if (ofse > oldOffset) {
                ofse += sizeDelta;
            }
        }

        // Adjust offset and occupied by the size change
        offset += sizeDelta;
        occupied += sizeDelta;

        allocator->replaceMemory(buffer, memory, data.buffer, alignedSize, oldOffset);

        // Edit the buffer element to reflect the new data it now represents
        buffers[index].numElements = static_cast<uint32_t>(data.capacity / sizeof(T));

        // Update offsets of all subsequent buffers
        for (size_t i = index + 1; i < buffers.size(); ++i) {
            buffers[i].offset += sizeDelta;
            buffers[i].elementOffset += static_cast<uint32_t>((alignedSize - oldAlignedSize) / sizeof(T));
            buffers[i].createDescriptors(static_cast<uint32_t>(i), flags);
        }

        // internal offsets were changed. We need descriptor updates
        descUpdateQueued.trigger();
    }

    void insert(uint32_t index, std::vector<T>& data) {

        if (index > buffers.size() || index < 0) {
            throw std::out_of_range("Index out of range");
        }

        auto alignedSize = (data.size() * sizeof(T) + alignment - 1) & ~(alignment - 1);

        if (capacity < occupied + alignedSize) {
            growUntil(2, occupied + alignedSize);
        }

        // we're using the staging buffer to hold the toInsert memory
        std::memcpy(mapped, data.data(), data.size() * sizeof(T));
        allocator->insertMemory(buffer, stagingBuffer, buffers[index].offset, alignedSize);
        std::memset(mapped, 0, data.size() * sizeof(T));
        
        // update all the gaps whose offset is greater than the offset at which we just inserted the data
        for (uint32_t i = 0; i < gaps.size(); i++){
            if(gaps[i].first >= buffers[index].offset){
                auto& [ofse, siz] = gaps[i];
                ofse += alignedSize;
            }
        }
        
        // Create a Buffer entry
        auto newBuffer = new Buffer<T>();
        newBuffer->buffer = buffer;
        newBuffer->offset = buffers[index].offset;
        newBuffer->numElements = static_cast<uint32_t>(data.size());
		newBuffer->elementOffset = buffers[index - 1].elementOffset + buffers[index - 1].numElements;
        newBuffer->createDescriptors(index, flags);
        buffers.insert(buffers.begin() + index, *newBuffer);
		delete newBuffer; // Free the heap memory

        // Update offset for next allocation
        offset += alignedSize;
        occupied += alignedSize;
        for (size_t i = index + 1; i < buffers.size(); ++i) {
            buffers[i].offset += alignedSize;
			buffers[i].elementOffset += static_cast<uint32_t>(data.size());
            buffers[i].createDescriptors(static_cast<uint32_t>(i), flags); // assign correct binding
        }

        // internal offsets were changed. We need descriptor updates
        descUpdateQueued.trigger();
    }

    void insert(uint32_t index, StandaloneBuffer<T>& data) {
        if (index > buffers.size() || index < 0) {
            throw std::out_of_range("Index out of range");
        }
        auto alignedSize = data.capacity;
        if (capacity < occupied + alignedSize) {
            growUntil(2, occupied + alignedSize);
        }
        
        allocator->insertMemory(buffer, data.buffer, buffers[index].offset, alignedSize);
        
        // update all the gaps whose offset is greater than the offset at which we just inserted the data
        for (uint32_t i = 0; i < gaps.size(); i++){
            if(gaps[i].first >= buffers[index].offset){
                auto& [ofse, siz] = gaps[i];
                ofse += alignedSize;
            }
        }
        
        // Create a Buffer entry
        auto newBuffer = new Buffer<T>();
        newBuffer->buffer = buffer;
        newBuffer->offset = buffers[index].offset;
        newBuffer->numElements = static_cast<uint32_t>(data.capacity / sizeof(T));
        newBuffer->elementOffset = buffers[index - 1].elementOffset + buffers[index - 1].numElements;
        newBuffer->createDescriptors(index, flags);
        buffers.insert(buffers.begin() + index, *newBuffer);
        delete newBuffer; // Free the heap memory
        
        // Update offset for next allocation
        offset += alignedSize;
        occupied += alignedSize;
        
        for (size_t i = index + 1; i < buffers.size(); ++i) {
            buffers[i].offset += alignedSize;
            buffers[i].elementOffset += static_cast<uint32_t>(data.capacity / sizeof(T));
            buffers[i].createDescriptors(static_cast<uint32_t>(i), flags); // assign correct binding
        }
        
        // internal offsets were changed. We need descriptor updates
        descUpdateQueued.trigger();
    }

    std::vector<T> downloadPool() {
        allocator->copyBuffer(buffer, stagingBuffer, capacity, 0, 0, true);
        std::vector<T> data;
        data.resize(capacity / sizeof(T));
        std::memcpy(data.data(), mapped, capacity);
        std::memset(mapped, 0, capacity);
        return data;
    }

    // Daggers in my eyes, but it's not that important, so not gonna waste time on it
    std::vector<std::vector<T>> downloadBuffers() {
        std::vector<std::vector<T>> bufs{};
        for (auto& buf : buffers) {
            allocator->copyBuffer(buffer, stagingBuffer, buf.size(), buf.offset, 0, true);
            std::vector<T> data(buf.numElements);
            std::memcpy(data.data(), mapped, buf.size());
            std::memset(mapped, 0, buf.size());
            bufs.push_back(data);
        }

        return bufs;
    }

    void printPool() {  
        auto allBufs = downloadBuffers();  

        for (int i = 0; i < allBufs.size(); i++) {  
            std::cout << "buffer number " << i << "\n\t";  
            for (auto& ele : allBufs[i]) {  
                std::cout << ele << "\n";  
            }  
            std::cout << "\n";  
        }  
    }
};

// "V" is your vertex struct with all the attribs
template<typename V>
struct VertexBuffer {
    Allocator* allocator;
    VkBuffer vBuffer;
    VkDeviceMemory vMem;
    VkDeviceAddress address;

    VkBuffer stageBuf;
    VkDeviceMemory stageMem;

    void* map;

    VertexBuffer() {};

    bool uploaded = false;

    VertexBuffer(Allocator* allocator, size_t numVertex) : allocator(allocator) {
        auto [buf, mem] = allocator->createBuffer(numVertex * sizeof(V), VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        vBuffer = buf;
        vMem = mem;

        auto [sBuf, sMem] = allocator->createBuffer(numVertex * sizeof(V), VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        stageBuf = sBuf;
        stageMem = sMem;

        allocator->init->disp.mapMemory(stageMem, 0, numVertex * sizeof(V), 0, &map);
        uploaded = true;

		VkBufferDeviceAddressInfoEXT bufferInfo{};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_EXT;
		bufferInfo.buffer = vBuffer;
		bufferInfo.pNext = nullptr;
        address = allocator->init->disp.getBufferDeviceAddress(&bufferInfo);
    };

    void upload(std::vector<V>& vertices) {
        std::memcpy(map, vertices.data(), vertices.size() * sizeof(V));
        allocator->copyBuffer(stageBuf, vBuffer, vertices.size() * sizeof(V), 0, 0, true);
        std::memset(map, 0, vertices.size() * sizeof(V));
        uploaded = true;
    }

    ~VertexBuffer() {
        allocator->init->disp.unmapMemory(stageMem);
        allocator->killMemory(vBuffer, vMem);
        allocator->killMemory(stageBuf, stageMem);
    }
};

template<typename T>
struct UniformBuffer {
    VkBuffer buffer;
    VkDeviceMemory memory;

    VkDeviceAddress address;
    VkBuffer stageBuffer;
    VkDeviceMemory stageMemory;

	VkDeviceSize capacity;

    void* map;
    Allocator* allocator;
    UniformBuffer() {};

    uint32_t bindingIndex;
    VkDescriptorSetLayoutBinding binding{};
    VkWriteDescriptorSet wrt_desc_set{};
    VkDescriptorBufferInfo desc_buf_info{};

    UniformBuffer(Allocator* allocator) : allocator(allocator) {
        auto [buf, mem] = allocator->createBuffer(sizeof(T), VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        buffer = buf;
        memory = mem;
        auto [sBuf, sMem] = allocator->createBuffer(sizeof(T), VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        stageBuffer = sBuf;
        stageMemory = sMem;
		capacity = sizeof(T);
        allocator->init->disp.mapMemory(stageMemory, 0, sizeof(T), 0, &map);
    };
	void upload(T data) {
		std::memcpy(map, &data, sizeof(T));
		allocator->copyBuffer(stageBuffer, buffer, sizeof(T), 0, 0, true);
		std::memset(map, 0, sizeof(T));
	}

	VkDeviceAddress getBufferAddress() {
		VkBufferDeviceAddressInfoEXT bufferInfo{};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_EXT;
		bufferInfo.buffer = buffer;
		bufferInfo.pNext = nullptr;
		address = allocator->init->disp.getBufferDeviceAddress(&bufferInfo);
		return address;
	}

    void createDescriptors(uint32_t bindingIdx, VkShaderStageFlags flags = VK_SHADER_STAGE_ALL) {
        bindingIndex = bindingIdx;
        binding.binding = bindingIndex;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        binding.descriptorCount = 1;
        binding.stageFlags = flags;
        binding.pImmutableSamplers = nullptr;
    }

    void updateDescriptorSet(VkDescriptorSet& set) {
        desc_buf_info.buffer = buffer;
        desc_buf_info.offset = 0;
        desc_buf_info.range = capacity;

        wrt_desc_set.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wrt_desc_set.dstSet = set;
        wrt_desc_set.dstBinding = bindingIndex;
        wrt_desc_set.dstArrayElement = 0;
        wrt_desc_set.descriptorCount = 1;
        wrt_desc_set.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        wrt_desc_set.pBufferInfo = &desc_buf_info;
    }

	~UniformBuffer() {
		allocator->init->disp.unmapMemory(stageMemory);
        allocator->killMemory(buffer, memory);
        allocator->killMemory(stageBuffer, stageMemory);
	}
};

typedef struct ImageDesc {
    uint32_t bindingIndex, width, height, mipLevels;
    VkFormat format;
    VkImageUsageFlags usage;
    VkImageLayout layout;
    VkSampleCountFlagBits samples;
    VkShaderStageFlags stage;
} ImageDesc;

struct Image2D {
	VkImage image;
	VkImageLayout imageLayout;
	VkDeviceMemory memory;
	VkImageView imageView;
	VkSampler sampler;

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingMemory;

	VkFormat format;
	VkImageUsageFlags usage;
	VkSampleCountFlagBits samples;
	uint32_t mipLevels;

    Allocator* alloc;

    VkDescriptorImageInfo imageInfo;
    uint32_t bindingIndex;
    VkDescriptorSetLayoutBinding binding{};
    VkWriteDescriptorSet wrt_desc_set{};

    int width, height;

    Image2D(int width, int height, int mipLevels, VkFormat format, VkImageUsageFlags usage, Allocator* allocator, VkImageLayout layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT) : format(format), usage(usage), samples(samples), mipLevels(mipLevels), width(width), height(height), alloc(allocator) {
        imageLayout = layout; // need to change this to be customizable
        image = VK_NULL_HANDLE;
        memory = VK_NULL_HANDLE;
        imageView = VK_NULL_HANDLE;
        sampler = VK_NULL_HANDLE;
        auto image = allocator->createImage(width, height, mipLevels, usage | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_TYPE_2D, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, format, samples);
		this->image = image.first;
		this->memory = image.second;

		allocator->transitionImageLayout(image.first, format, VK_IMAGE_LAYOUT_UNDEFINED, imageLayout, mipLevels);

        auto [buf, mem] = allocator->createBuffer(static_cast<VkDeviceSize>(width * height) * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        stagingBuffer = buf;
        stagingMemory = mem;

		if (format == VK_FORMAT_D32_SFLOAT) {
			//imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
			createImageView(true);
		}
		else if (format == VK_FORMAT_D24_UNORM_S8_UINT) {
			//imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
			createImageView(true);
		}

        else { createImageView(); }
        createSampler();
    };

    Image2D(ImageDesc desc, Allocator* allocator) : bindingIndex(desc.bindingIndex), format(desc.format), usage(desc.usage), samples(desc.samples), mipLevels(desc.mipLevels), width(desc.width), height(desc.height), alloc(allocator) {
        imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;  // need to change this to be customizable
        image = VK_NULL_HANDLE;
        memory = VK_NULL_HANDLE;
        imageView = VK_NULL_HANDLE;
        sampler = VK_NULL_HANDLE;
        auto image = allocator->createImage(width, height, mipLevels, usage, VK_IMAGE_TYPE_2D, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, format, samples);
        this->image = image.first;
        this->memory = image.second;

        auto [buf, mem] = allocator->createBuffer(static_cast<VkDeviceSize>(width * height) * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        stagingBuffer = buf;
        stagingMemory = mem;

        createImageView();
        createSampler();
    };

    Image2D(const std::string& path, Allocator* allocator) : alloc(allocator) {
        imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        image = VK_NULL_HANDLE;
        memory = VK_NULL_HANDLE;
        imageView = VK_NULL_HANDLE;
        sampler = VK_NULL_HANDLE;

		uploadTexture(path, allocator);
    }

    Image2D() {};

    ~Image2D() {
        alloc->killMemory(stagingBuffer, stagingMemory);
        alloc->killImage(image, memory);
        alloc->init->disp.destroyImageView(imageView, nullptr);
        alloc->init->disp.destroySampler(sampler, nullptr);
    }

    void createDescriptors(uint32_t bindingIdx, VkShaderStageFlags flags = VK_SHADER_STAGE_FRAGMENT_BIT, VkDescriptorType descType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
        bindingIndex = bindingIdx;
        binding.binding = bindingIndex;
        binding.descriptorType = descType;
        binding.descriptorCount = 1;
        binding.stageFlags = flags;
        binding.pImmutableSamplers = nullptr;
    }

    // call after uploadTexture has setup the layout and assigned the memory
    void updateDescriptorSet(VkDescriptorSet& set, size_t arrayElement = 0) {
        imageInfo.imageLayout = imageLayout;
        imageInfo.imageView = imageView;
        imageInfo.sampler = sampler;

        wrt_desc_set.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wrt_desc_set.dstSet = set;
        wrt_desc_set.dstBinding = bindingIndex;
        wrt_desc_set.dstArrayElement = arrayElement;
        wrt_desc_set.descriptorCount = 1;
        wrt_desc_set.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        wrt_desc_set.pImageInfo = &imageInfo;
    }

    void generateMipmaps() {
        auto cmd = alloc->getSingleTimeCmd(true);
        VkImageMemoryBarrier barrier{};
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.image = image;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;  // because only color images can have mipmaps. because why in the world would a depth image need mipaps.
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount = 1;
		barrier.subresourceRange.levelCount = 1;
        
		int32_t mipWidth = width;
		int32_t mipHeight = height;

        for (uint32_t i = 1; i < mipLevels; i++) {
			barrier.subresourceRange.baseMipLevel = i - 1;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
			vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 0, nullptr, 0, nullptr, 1, &barrier);
        
            VkImageBlit blit{};
            blit.srcOffsets[0] = { 0, 0, 0 };
            blit.srcOffsets[1] = { mipWidth, mipHeight, 1 };
            blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            blit.srcSubresource.mipLevel = i - 1;
            blit.srcSubresource.baseArrayLayer = 0;
            blit.srcSubresource.layerCount = 1;
            blit.dstOffsets[0] = { 0, 0, 0 };
            blit.dstOffsets[1] = { mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1 };
            blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            blit.dstSubresource.mipLevel = i;
            blit.dstSubresource.baseArrayLayer = 0;
            blit.dstSubresource.layerCount = 1;
            vkCmdBlitImage(cmd,
                image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1, &blit,
                VK_FILTER_LINEAR);

            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                0, nullptr,
                0, nullptr,
                1, &barrier);

            if (mipWidth > 1) mipWidth /= 2;
            if (mipHeight > 1) mipHeight /= 2;
        }
        
        barrier.subresourceRange.baseMipLevel = mipLevels - 1;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
            0, nullptr,
            0, nullptr,
            1, &barrier);

        alloc->init->disp.endCommandBuffer(cmd);
		alloc->submitSingleTimeCmd(cmd, false, true);
    }

    void uploadTexture(const std::string& path, Allocator* allocator) {
        
        alloc = allocator;

        int texWidth, texHeight, texChannels;
        stbi_uc* pixels = stbi_load(path.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        VkDeviceSize imageSize = texWidth * texHeight * 4;

        width = texWidth;
        height = texHeight;

        mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1;

        auto im = allocator->createImage(width, height, mipLevels, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_TYPE_2D, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_FORMAT_R8G8B8A8_SRGB, VK_SAMPLE_COUNT_1_BIT);
        this->image = im.first;
        this->memory = im.second;

        auto [buf, mem] = allocator->createBuffer(static_cast<VkDeviceSize>(width * height * 4), VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        stagingBuffer = buf;
        stagingMemory = mem;

		format = VK_FORMAT_R8G8B8A8_SRGB;
		usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
		samples = VK_SAMPLE_COUNT_1_BIT;
		imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        createSampler();
        createImageView();

        void* data;
        allocator->init->disp.mapMemory(stagingMemory, 0, imageSize, 0, &data);
        std::memcpy(data, pixels, imageSize);
        allocator->init->disp.unmapMemory(stagingMemory);

        stbi_image_free(pixels);
        
        allocator->transitionImageLayout(image, format, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, mipLevels);
        allocator->copyBufferToImage2D(stagingBuffer, image, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
        generateMipmaps(); // handles all the image layout transitions too
    }

    void createImageView(bool depthOnly = false) {
        // Create Image View
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = format;
        if (!depthOnly) {
            viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		}
		else {
			viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
		}
        
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = mipLevels;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;
        viewInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        viewInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        viewInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        viewInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        if (alloc->init->disp.createImageView(&viewInfo, nullptr, &imageView) != VK_SUCCESS) {
            throw std::runtime_error("could not create image view");
        }
    }

    void createSampler() {
        // Create Sampler
        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.anisotropyEnable = VK_FALSE;
        samplerInfo.maxAnisotropy = 0;// alloc->init->device.physical_device.properties.limits.maxSamplerAnisotropy;;
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerInfo.mipLodBias = 0.0f;
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = static_cast<float>(mipLevels);
        if (alloc->init->disp.createSampler(&samplerInfo, nullptr, &sampler) != VK_SUCCESS) {
            throw std::runtime_error("could not create sampler");
        }          
    }

    std::unique_ptr<Image2D> deepcpy() {
		auto newImage = std::make_unique<Image2D>(width, height, mipLevels, format, usage, alloc, imageLayout, samples);
		newImage->imageLayout = imageLayout;
		newImage->bindingIndex = bindingIndex;
		newImage->createDescriptors(bindingIndex);
		return newImage;
    }
};

// uses bindless descriptors for Image2D access in shaders.
struct ImageArray {
    std::vector<std::shared_ptr<Image2D>> images;

    VkDescriptorSet descSet = VK_NULL_HANDLE;
    VkDescriptorSetLayout descSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool descPool = VK_NULL_HANDLE;

    uint32_t numImages = 100;
    uint32_t bindingIndex = 0;
    Allocator* allocator;

    ImageArray(uint32_t numImages, Allocator* allocator) : numImages(numImages), allocator(allocator) {
		if (numImages == 0) {
			std::cerr << "numImages cannot be 0" << std::endl;
			return;
		}
		createDescriptorPool();
		createDescSetLayout();  // for now, the set = 1 is reserved for bindless textures via ImageArray
		allocateDescSet();
    };

    // when the ImageArray is a member of a larger pool
    ImageArray(uint32_t numImages, Allocator* allocator, uint32_t bindingIdx) : numImages(numImages), allocator(allocator), bindingIndex(bindingIdx) {}

	ImageArray() {};

    ~ImageArray() {
        if (descSetLayout != VK_NULL_HANDLE) {
            allocator->init->disp.destroyDescriptorSetLayout(descSetLayout, nullptr);
        }
		if (descPool != VK_NULL_HANDLE) {
			allocator->init->disp.freeDescriptorSets(descPool, 1, &descSet);
            allocator->init->disp.destroyDescriptorPool(descPool, nullptr);
		}
    }

    // creates new image and pushes it back into the array and descriptor set
    void push_back(const std::string& path) {
        auto im = std::make_shared<Image2D>(path, allocator);
        images.push_back(im);
    }

	void erase(uint32_t index) {
		if (index >= images.size()) {
			throw std::out_of_range("Index out of range");
		}
		images.erase(images.begin() + index);
	}

    void createDescriptorPool() {
        VkDescriptorPoolSize pool_sizes_bindless[] =
        {
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, numImages }
        };

        VkDescriptorPoolCreateInfo poolInfo = {};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = pool_sizes_bindless;
        poolInfo.maxSets = 1;
        poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
		allocator->init->disp.createDescriptorPool(&poolInfo, nullptr, &descPool);
    }

    void createDescSetLayout() {
        VkDescriptorBindingFlags bindless_flags = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT_EXT | VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT_EXT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT_EXT;

        VkDescriptorSetLayoutBinding vk_binding;
        vk_binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        vk_binding.descriptorCount = numImages;
        vk_binding.binding = 0;

        vk_binding.stageFlags = VK_SHADER_STAGE_ALL;
        vk_binding.pImmutableSamplers = nullptr;

        VkDescriptorSetLayoutCreateInfo layout_info = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        layout_info.bindingCount = 1;
        layout_info.pBindings = &vk_binding;
        layout_info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT_EXT;

        VkDescriptorSetLayoutBindingFlagsCreateInfoEXT extended_info{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO_EXT, nullptr };
        extended_info.bindingCount = 1;
        extended_info.pBindingFlags = &bindless_flags;

        layout_info.pNext = &extended_info;

        allocator->init->disp.createDescriptorSetLayout(&layout_info, nullptr, &descSetLayout);
    }

    void allocateDescSet() {
        VkDescriptorSetAllocateInfo alloc_info{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        alloc_info.descriptorPool = descPool;
        alloc_info.descriptorSetCount = 1;
        alloc_info.pSetLayouts = &descSetLayout;

        VkDescriptorSetVariableDescriptorCountAllocateInfoEXT count_info{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO_EXT };
        uint32_t max_binding = 100 - 1;
        count_info.descriptorSetCount = 1;
        // This number is the max allocatable count
        count_info.pDescriptorCounts = &max_binding;
        alloc_info.pNext = &count_info;

        allocator->init->disp.allocateDescriptorSets(&alloc_info, &descSet);
    }

    void updateDescriptorSets() {
        std::vector<VkWriteDescriptorSet> writes(images.size());
        for (size_t i = 0; i < images.size(); ++i) {
            images[i]->createDescriptors(bindingIndex, VK_SHADER_STAGE_ALL);
			images[i]->updateDescriptorSet(descSet, i);
            writes[i] = images[i]->wrt_desc_set;
        }
		allocator->init->disp.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }
};

// for multiple image arrays in the desc pool
struct ImagePool {
    Allocator* allocator;
    std::vector<std::unique_ptr<ImageArray>> arrays;
	std::vector<VkDescriptorSet> descSets;
    uint32_t numArrays = 0;

    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;

    uint32_t totalImages = 1000;

    ImagePool(uint32_t numArrays = 10, uint32_t arraySizes = 100, Allocator* allocator = nullptr) : totalImages(numArrays* arraySizes), allocator(allocator), numArrays(numArrays) {
        createDescriptorPool();
    }

    ~ImagePool() {
		if (pool != VK_NULL_HANDLE) {
			allocator->init->disp.freeDescriptorSets(pool, 1, &set);
			allocator->init->disp.destroyDescriptorPool(pool, nullptr);
		}
		if (layout != VK_NULL_HANDLE) {
			allocator->init->disp.destroyDescriptorSetLayout(layout, nullptr);
		}
    }

    void addArray(std::unique_ptr<ImageArray>& array) {
        array->descPool = pool;
		array->createDescSetLayout();
		array->allocateDescSet();
        descSets.push_back(array->descSet);
        arrays.push_back(std::move(array));
    }

    void createDescriptorPool() {
        VkDescriptorPoolSize pool_sizes_bindless[] =
        {
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, totalImages }
        };

        VkDescriptorPoolCreateInfo poolInfo = {};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = pool_sizes_bindless;
        poolInfo.maxSets = numArrays;
        poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
        allocator->init->disp.createDescriptorPool(&poolInfo, nullptr, &pool);
    }

    void updateDescriptorSets() {
        for (auto& arr : arrays) {
            arr->updateDescriptorSets();
        }
    }
};

// a singular framebuffer attachment
struct FBAttachment {
    std::unique_ptr<Image2D> image;
    VkImageLayout initialLayout;
    VkImageLayout finalLayout;
    VkFormat format;

	VkAttachmentDescription attachmentDescription{};
	VkAttachmentReference attachmentReference{};

    FBAttachment() {};
    FBAttachment(int width, int height, int bindingIndex, VkFormat format, VkImageUsageFlags usage, VkImageLayout initialLayout, VkImageLayout finalLayout, VkImageLayout attachmentLayout, Allocator* allocator) : 
        image(std::make_unique<Image2D>(width, height, 1, format, usage, allocator, finalLayout)), 
        initialLayout(initialLayout), finalLayout(finalLayout), format(format) {

        image->createDescriptors(bindingIndex, VK_SHADER_STAGE_ALL);

        attachmentDescription.flags = 0;
        attachmentDescription.format = format;
        attachmentDescription.samples = VK_SAMPLE_COUNT_1_BIT;
        attachmentDescription.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachmentDescription.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachmentDescription.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachmentDescription.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachmentDescription.initialLayout = initialLayout;
        attachmentDescription.finalLayout = finalLayout;

		attachmentReference.attachment = bindingIndex;
		attachmentReference.layout = attachmentLayout;
    }

    std::shared_ptr<FBAttachment> deepcpy() {
		auto newAttachment = std::make_unique<FBAttachment>(image->width, image->height, attachmentReference.attachment, format, image->usage, initialLayout, finalLayout, attachmentReference.layout, image->alloc);
		return newAttachment;
    }
};
// a framebuffer object. Can be used for offscreen rendering, or as a render pass attachment
// This specific implementation now works for multiple framebuffers in flight, each with the identical attachments.
struct Framebuffer {
    std::vector<VkFramebuffer> framebuffers;
    VkRenderPass renderPass;

    struct Attachments {
        std::vector<std::shared_ptr<FBAttachment>> attachments;
    };
	std::vector<Attachments> attachments;

    Allocator* allocator;

    VkSubpassDescription subpassDesc;
	VkSubpassDependency subpassDependency;

    int width, height, numAttachments, MAX_FRAMES;
    bool hasDepthStencil = false;

    Framebuffer() {};

	// MAX_FRAMES is used for multiple frames in flight. Generates MAX_FRAMES framebuffers, each with the same attachments. But of couese, the images are deep copied.
    Framebuffer(int width, int height, int numAttachments, int maxFrames, VkRenderPass renderPass, Allocator* allocator) : width(width), height(height), renderPass(renderPass), allocator(allocator), numAttachments(numAttachments), MAX_FRAMES(maxFrames) {
		attachments.resize(MAX_FRAMES);
		framebuffers.resize(MAX_FRAMES);
    }

	void addAttachment(VkFormat format, VkImageUsageFlags usage, VkImageLayout initialLayout, VkImageLayout attachmentLayout, VkImageLayout finalLayout) {
		int idx = attachments[0].attachments.size();
        for (int i = 0; i < MAX_FRAMES; i++) {
            attachments[i].attachments.push_back(std::make_shared<FBAttachment>(width, height, idx, format, usage, initialLayout, finalLayout, attachmentLayout, allocator));
        }
	}

    void addAttachment(std::shared_ptr<FBAttachment>& attachment) {
		auto it = std::find_if(attachments[0].attachments.begin(), attachments[0].attachments.end(),
			[&attachment](const std::shared_ptr<FBAttachment>& a) { return a->attachmentReference.attachment == attachment->attachmentReference.attachment; });
		if (it != attachments[0].attachments.end()) {
			return;
		}
        for (int i = 0; i < MAX_FRAMES; i++) {
            attachments[i].attachments.push_back(attachment->deepcpy());
        }
    }

	void init() {
        
        for (int i = 0; i < MAX_FRAMES; i++) {
            std::vector<VkImageView> views(numAttachments);
		    for (int j = 0; j < numAttachments; j++) {
		    	views[j] = attachments[i].attachments[j]->image->imageView;
		    }
            VkFramebufferCreateInfo framebufferInfo{};
		    framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		    framebufferInfo.renderPass = renderPass;
		    framebufferInfo.attachmentCount = views.size();
		    framebufferInfo.pAttachments = views.data();
		    framebufferInfo.width = width;
		    framebufferInfo.height = height;
		    framebufferInfo.layers = 1;
		    if (allocator->init->disp.createFramebuffer(&framebufferInfo, nullptr, &framebuffers[i]) != VK_SUCCESS) {
			    throw std::runtime_error("failed to create framebuffer!");
		    }
		    std::cout << "Framebuffer created with " << numAttachments << " attachments." << std::endl;
        }
    }

	std::vector<VkAttachmentDescription> getAttachmentDescriptions() {
		std::vector<VkAttachmentDescription> descs;
        // just take the first attachment set, because each set is identical.
		for (auto& attachment : attachments[0].attachments) {
			descs.push_back(attachment->attachmentDescription);
		}
		return descs;
	}

	std::vector<VkAttachmentReference> getAttachmentReferences() {
		std::vector<VkAttachmentReference> refs;
		for (auto& attachment : attachments[0].attachments) {
			refs.push_back(attachment->attachmentReference);
		}
		return refs;
	}

    ~Framebuffer() {
        for (int i = 0; i < framebuffers.size(); i++) {
            allocator->init->disp.destroyFramebuffer(framebuffers[i], nullptr);
        }
    }
};