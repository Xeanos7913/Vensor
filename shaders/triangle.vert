#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_buffer_reference_uvec2 : enable
#extension GL_EXT_scalar_block_layout : enable

// Outputs
layout(location = 0) out vec3 outColor;
layout(location = 1) out vec2 outUV;
layout(location = 2) out int mIndex;
layout(location = 3) out int frameIndex;
layout(location = 4) out int numLights;

struct PackedIndex {
    uint pos;
    uint norm;
    uint uv;
    uint pad;
};

// Buffer references
layout(buffer_reference, std430, buffer_reference_align = 16) buffer Positions {
    vec3 positions[];
};
layout(buffer_reference, std430, buffer_reference_align = 16) buffer Normals {
    vec3 normals[];
};
layout(buffer_reference, std430, buffer_reference_align = 16) buffer UVs {
    vec2 uvs[];
};
layout(buffer_reference, scalar, buffer_reference_align = 16) buffer PackedIndexBuffer {
    PackedIndex indices[];
};

struct UniformBuf {
    Positions posBuf;
    Normals normalBuf;
    UVs uvBuf;
    PackedIndexBuffer indexBuf;
};

layout(buffer_reference, std430, buffer_reference_align = 16) buffer UniformBuffer {
    UniformBuf data;
};

// Push constants
layout(push_constant) uniform PushConstant {
    mat4 model;
    UniformBuffer uniforms;
    int materialIndex;
    int positionOffset; // offsets aren't byte offsets, they're element offsets
    int uvOffset;
	int normalOffset;
    int indexOffset;    // this is index of the first element in the index buffer of the current mesh
    int frameIndex;
    int nLights;
} pushConstant;

void main() {
    uint idx = gl_VertexIndex;

    PackedIndex packed = pushConstant.uniforms.data.indexBuf.indices[idx + pushConstant.indexOffset];

    vec3 pos = pushConstant.uniforms.data.posBuf.positions[packed.pos + pushConstant.positionOffset];
    vec3 normal = pushConstant.uniforms.data.normalBuf.normals[packed.norm + pushConstant.normalOffset];
    vec2 uv = pushConstant.uniforms.data.uvBuf.uvs[packed.uv + pushConstant.uvOffset];

    gl_Position = pushConstant.model * vec4(pos, 1.0);

    frameIndex = pushConstant.frameIndex;
    outColor = normal;
    outUV = uv;
    mIndex = pushConstant.materialIndex;
    numLights = pushConstant.nLights;
}