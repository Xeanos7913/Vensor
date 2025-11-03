#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_buffer_reference_uvec2 : enable
#extension GL_EXT_scalar_block_layout : enable

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
layout(buffer_reference, scalar, buffer_reference_align = 16) buffer PackedIndexBuffer {
    PackedIndex indices[];
};

// Push constants
layout(push_constant) uniform PushConstant {
    mat4 model;
    Positions posBuf;
    PackedIndexBuffer indexBuf;
    int positionOffset; // offsets aren't byte offsets, they're element offsets
    int indexOffset;    // this is index of the first element in the index buffer of the current mesh
} pushConstant;

void main() {
    uint idx = gl_VertexIndex;

    PackedIndex packed = pushConstant.indexBuf.indices[idx + pushConstant.indexOffset];

    vec3 pos = pushConstant.posBuf.positions[packed.pos + pushConstant.positionOffset];

    gl_Position = pushConstant.model * vec4(pos, 1.0);

}