#pragma once


#include <vulkan/vulkan_core.h>
#include "VkBootstrap.h"
#include <glm/glm.hpp>
#include <string>
#include <sstream>

struct ShaderFunction {

};

enum class DescriptorType {
	SAMPLER2D,
	SINT16_BUFFER,
	UINT16_BUFFER,
	SINT32_BUFFER,
	UINT32_BUFFER,
	SFLOAT16_BUFFER,
	UFLOAT16_BUFFER,
	SFLOAT32_BUFFER,
	UFLOAT32_BUFFER,
	VEC2_BUFFER,
	VEC3_BUFFER,
	VEC4_BUFFER,
	REFERENCE_BUFFER_SCALAR,
	REFERENCE_BUFFER_FLOAT,
	REFERENCE_BUFFER_INT,
	REFERENCE_BUFFER_VEC2,
	REFERENCE_BUFFER_VEC4
};

struct Descriptor {
	std::stringstream code;

	Descriptor(const DescriptorType& type, const std::string& name, int bindingIndex, int set = 0) {
		switch (type) {
		case DescriptorType::SAMPLER2D:
			code << "layout(set = " << set << ", binding = " << bindingIndex << ") uniform sampler2D " << name << ";";
			break;

		case DescriptorType::SINT16_BUFFER:
			code << "layout(set = " << set << ", binding = " << bindingIndex << ") buffer " << name << " { int16_t data[]; };";
			break;

		case DescriptorType::UINT16_BUFFER:
			code << "layout(set = " << set << ", binding = " << bindingIndex << ") buffer " << name << " { uint16_t data[]; };";
			break;

		case DescriptorType::SINT32_BUFFER:
			code << "layout(set = " << set << ", binding = " << bindingIndex << ") buffer " << name << " { int data[]; };";
			break;

		case DescriptorType::UINT32_BUFFER:
			code << "layout(set = " << set << ", binding = " << bindingIndex << ") buffer " << name << " { uint data[]; };";
			break;

		case DescriptorType::SFLOAT16_BUFFER:
			code << "layout(set = " << set << ", binding = " << bindingIndex << ") buffer " << name << " { float16_t data[]; };";
			break;

		case DescriptorType::UFLOAT16_BUFFER:
			code << "layout(set = " << set << ", binding = " << bindingIndex << ") buffer " << name << " { float16_t data[]; };";
			break;

		case DescriptorType::SFLOAT32_BUFFER:
			code << "layout(set = " << set << ", binding = " << bindingIndex << ") buffer " << name << " { float data[]; };";
			break;

		case DescriptorType::UFLOAT32_BUFFER:
			code << "layout(set = " << set << ", binding = " << bindingIndex << ") buffer " << name << " { float data[]; };";
			break;

		case DescriptorType::VEC2_BUFFER:
			code << "layout(set = " << set << ", binding = " << bindingIndex << ") buffer " << name << " { vec2 data[]; };";
			break;

		case DescriptorType::VEC3_BUFFER:
			code << "layout(set = " << set << ", binding = " << bindingIndex << ") buffer " << name << " { vec3 data[]; };";
			break;

		case DescriptorType::VEC4_BUFFER:
			code << "layout(set = " << set << ", binding = " << bindingIndex << ") buffer " << name << " { vec4 data[]; };";
			break;

		case DescriptorType::REFERENCE_BUFFER_SCALAR:
			code << "layout(buffer_reference, scalar, buffer_reference_align = 16) buffer " << name << " { vec4 data[]; };";
			break;

		case DescriptorType::REFERENCE_BUFFER_FLOAT:
			code << "layout(buffer_reference, std430, buffer_reference_align = 16) buffer " << name << " { float data[]; };";
			break;

		case DescriptorType::REFERENCE_BUFFER_INT:
			code << "layout(buffer_reference, std430, buffer_reference_align = 16) buffer " << name << " { int data[]; };";
			break;

		case DescriptorType::REFERENCE_BUFFER_VEC2:
			code << "layout(buffer_reference, std430, buffer_reference_align = 16) buffer " << name << " { vec2 data[]; };";
			break;

		case DescriptorType::REFERENCE_BUFFER_VEC4:
			code << "layout(buffer_reference, std430, buffer_reference_align = 16) buffer " << name << " { vec4 data[]; };";
			break;
		}
	}
};

struct Push {

};

class Shader {
public:

private:
	std::vector<std::string> extensions;
	std::vector<Descriptor> descriptors;
};