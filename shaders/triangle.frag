#version 450  
#extension GL_ARB_separate_shader_objects : enable  
#extension GL_KHR_vulkan_glsl : enable  
#extension GL_EXT_nonuniform_qualifier : enable  
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_buffer_reference_uvec2 : enable
#extension GL_EXT_scalar_block_layout : enable

// this has the size of two vec4's
struct PointLight{
    vec3 position;
    vec3 colour;
    float intensity;
    int pad;
};

layout (location = 0) in vec3 fragColor;  
layout (location = 1) in vec2 fragTexCoord;  
layout (location = 2) in flat int fragTexIndex;  
layout (location = 3) in flat int frameIndex;
layout (location = 4) in flat int numLights;

layout (location = 0) out vec4 outColor;  

layout(set = 0, binding = 0) uniform sampler2D depthTex1;  
layout (set = 0, binding = 1) uniform sampler2D depthTex2;  
layout(set = 0, binding = 2) uniform sampler2D depthTex3;  

layout(set = 1, binding = 0) uniform sampler2D textures[];

layout(buffer_reference, scalar, buffer_reference_align = 16) buffer LightBuffer {
    PointLight lights[];
};

void main() {  
    vec2 screenUV = gl_FragCoord.xy / vec2(1920, 1080);  

    float sceneDepth;  
    if (frameIndex == 0) {  
        sceneDepth = texture(depthTex1, screenUV).r;  
    } else if (frameIndex == 1) {  
        sceneDepth = texture(depthTex2, screenUV).r;  
    } else if (frameIndex == 2) {  
        sceneDepth = texture(depthTex3, screenUV).r;  
    } else {  
        discard; // Invalid frameIndex  
    }  

    float fragDepth = gl_FragCoord.z;  

    // Adjust depth comparison for reverse-z technique  
    if (fragDepth < sceneDepth)  
        discard;  

    outColor = texture(textures[fragTexIndex], fragTexCoord); // mesh UVs here are fine  
}
