/*
    This is a work in progress shader builder Domain Specific Language inspired by Triron. It currently works somewhat. 
    I plan to use this to rewrite all the shaders in this project.
    And yes, the name of this project is Livia.

    Look at main.cpp for an example shader built using Livia. It doesn't do anything useful but it showcases what Livia is capable of as of now.
*/

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdint>
#include <unordered_map>
#include <array>
#include <memory>
#include <stdexcept>
enum class types{
    UINT64,
    INT64,
    FLOAT32,
    INT32,
    UINT32,
    TENSOR
};

constexpr std::unordered_map<types, std::string> type_to_string(){
    std::unordered_map<types, std::string> map;
    map[types::UINT32] = "uint";
    map[types::INT32] = "int";
    map[types::FLOAT32] = "float";
    map[types::TENSOR] = "TensorImpl";
    map[types::UINT64] = "uint64";

    return map;
}

constexpr std::unordered_map<types, uint64_t> type_to_byte_ranges(){
    std::unordered_map<types, uint64_t> map;
    map[types::UINT32] = 4;
    map[types::UINT64] = 8;
    map[types::FLOAT32] = 4;
    map[types::INT32] = 4;
    map[types::INT64] = 8;
    
    return map;
}

struct SPIRVModule {
    std::vector<uint32_t> binary;
    uint32_t next_id = 1;
    
    // Separate sections for organization
    std::vector<uint32_t> capabilities;
    std::vector<uint32_t> extensions;
    std::vector<uint32_t> ext_inst_imports;
    std::vector<uint32_t> memory_model;
    std::vector<uint32_t> entry_points;
    std::vector<uint32_t> execution_modes;
    std::vector<uint32_t> debug_strings;
    std::vector<uint32_t> annotations;
    std::vector<uint32_t> types;
    std::vector<uint32_t> constants;
    std::vector<uint32_t> variables;
    std::vector<uint32_t> functions;
    
    uint32_t allocate_id() { return next_id++; }
    
    void emit_header() {
        binary.push_back(0x07230203);  // Magic number
        binary.push_back(0x00010000);  // Version 1.0
        binary.push_back(0);           // Generator (0 = unknown)
        binary.push_back(next_id - 2);     // Bound (highest ID + 1)
        binary.push_back(0);           // Schema (must be 0)
    }

    void assemble() {
        emit_header();
        
        // Append sections in correct order
        binary.insert(binary.end(), capabilities.begin(), capabilities.end());
        binary.insert(binary.end(), extensions.begin(), extensions.end());
        binary.insert(binary.end(), ext_inst_imports.begin(), ext_inst_imports.end());
        binary.insert(binary.end(), memory_model.begin(), memory_model.end());
        binary.insert(binary.end(), entry_points.begin(), entry_points.end());
        binary.insert(binary.end(), execution_modes.begin(), execution_modes.end());
        binary.insert(binary.end(), debug_strings.begin(), debug_strings.end());
        binary.insert(binary.end(), annotations.begin(), annotations.end());
        binary.insert(binary.end(), types.begin(), types.end());
        binary.insert(binary.end(), constants.begin(), constants.end());
        binary.insert(binary.end(), variables.begin(), variables.end());
        binary.insert(binary.end(), functions.begin(), functions.end());
    }
};

class SPIRVTypeEmitter {
    SPIRVModule& module;
    std::unordered_map<std::string, uint32_t> type_cache;
    
public:
    SPIRVTypeEmitter(SPIRVModule& m) : module(m) {}
    
    // OpTypeVoid
    uint32_t emit_void() {
        if (auto it = type_cache.find("void"); it != type_cache.end()) {
            return it->second;
        }
        uint32_t id = module.allocate_id();
        emit_instruction(module.types, 2, 19, {id});  // OpTypeVoid
        type_cache["void"] = id;
        return id;
    }
    
    // OpTypeInt
    uint32_t emit_int(uint32_t width, bool is_signed) {
        std::string key = (is_signed ? "i" : "u") + std::to_string(width);
        if (auto it = type_cache.find(key); it != type_cache.end()) {
            return it->second;
        }
        
        uint32_t id = module.allocate_id();
        emit_instruction(module.types, 4, 21, {id, width, is_signed ? 1u : 0u});
        type_cache[key] = id;
        return id;
    }
    
    // OpTypeFloat
    uint32_t emit_float(uint32_t width) {
        std::string key = "f" + std::to_string(width);
        if (auto it = type_cache.find(key); it != type_cache.end()) {
            return it->second;
        }
        
        uint32_t id = module.allocate_id();
        emit_instruction(module.types, 3, 22, {id, width});
        type_cache[key] = id;
        return id;
    }
    
    // OpTypePointer
    uint32_t emit_pointer(uint32_t storage_class, uint32_t type_id) {
        std::string key = "ptr_" + std::to_string(storage_class) + "_" + std::to_string(type_id);
        if (auto it = type_cache.find(key); it != type_cache.end()) {
            return it->second;
        }
        
        uint32_t id = module.allocate_id();
        emit_instruction(module.types, 4, 32, {id, storage_class, type_id});
        type_cache[key] = id;
        return id;
    }
    
    // OpTypeRuntimeArray (unsized array)
    uint32_t emit_runtime_array(uint32_t element_type) {
        std::string key = "rtarray_" + std::to_string(element_type);
        if (auto it = type_cache.find(key); it != type_cache.end()) {
            return it->second;
        }
        
        uint32_t id = module.allocate_id();
        emit_instruction(module.types, 3, 29, {id, element_type});
        type_cache[key] = id;
        return id;
    }
    
    // OpTypeStruct
    uint32_t emit_struct(const std::vector<uint32_t>& member_types) {
        uint32_t id = module.allocate_id();
        std::vector<uint32_t> operands = {id};
        operands.insert(operands.end(), member_types.begin(), member_types.end());
        emit_instruction(module.types, 2 + member_types.size(), 30, operands);
        return id;
    }
    
    uint32_t get_type_for(types t) {
        switch(t) {
            case types::UINT32: return emit_int(32, false);
            case types::INT32: return emit_int(32, true);
            case types::UINT64: return emit_int(64, false);
            case types::INT64: return emit_int(64, true);
            case types::FLOAT32: return emit_float(32);
            default: throw std::runtime_error("Type not handled");
        }
    }
    
private:
    void emit_instruction(std::vector<uint32_t>& target, uint32_t word_count, 
                         uint32_t opcode, const std::vector<uint32_t>& operands) {
        target.push_back((word_count << 16) | opcode);
        target.insert(target.end(), operands.begin(), operands.end());
    }
};

inline void emit_instruction(std::vector<uint32_t>& target, 
                            uint32_t word_count,
                            uint32_t opcode, 
                            const std::vector<uint32_t>& operands) {
    target.push_back((word_count << 16) | opcode);
    target.insert(target.end(), operands.begin(), operands.end());
}

inline std::vector<uint32_t> string_to_words(const std::string& s) {
    std::vector<uint32_t> words;
    size_t i = 0;
    while (i < s.length()) {
        uint32_t word = 0;
        for (int j = 0; j < 4 && i < s.length(); j++, i++) {
            word |= (static_cast<uint32_t>(s[i]) << (8 * j));
        }
        words.push_back(word);
    }
    // Ensure null termination
    if (s.length() % 4 != 0 || words.empty()) {
        if (words.empty() || (words.back() & 0xFF000000) != 0) {
            words.push_back(0);
        }
    }
    return words;
}

inline void setup_buffer_reference_module(SPIRVModule& module) {
    // OpCapability Shader (1)
    emit_instruction(module.capabilities, 2, 17, {1});
    
    // OpCapability PhysicalStorageBufferAddresses (5347)
    emit_instruction(module.capabilities, 2, 17, {5347});
    
    // OpExtension "SPV_KHR_physical_storage_buffer"
    std::string ext = "SPV_KHR_physical_storage_buffer";
    std::vector<uint32_t> words = string_to_words(ext);
    emit_instruction(module.extensions, 1 + words.size(), 10, words);
    
    // OpMemoryModel Physical64 GLSL450 (2, 1)
    emit_instruction(module.memory_model, 3, 14, {5348, 1});  // 2 = Physical64
}

class SPIRVEntryPoint {
    SPIRVModule& module;
    
public:
    SPIRVEntryPoint(SPIRVModule& m) : module(m) {}
    
    // OpEntryPoint GLCompute %main "main" %gl_GlobalInvocationID ...
    uint32_t emit_compute_entry(const std::string& name, uint32_t function_id,
                                const std::vector<uint32_t>& interface_vars) {
        std::vector<uint32_t> operands;
        operands.push_back(5);  // Execution Model: GLCompute
        operands.push_back(function_id);
        
        // Add name as words
        auto name_words = string_to_words(name);
        operands.insert(operands.end(), name_words.begin(), name_words.end());
        
        // Add interface variables
        operands.insert(operands.end(), interface_vars.begin(), interface_vars.end());
        
        emit_instruction(module.entry_points, 1 + operands.size(), 15, operands);
        return function_id;
    }
    
    // OpExecutionMode %main LocalSize x y z
    void emit_local_size(uint32_t entry_point, uint32_t x, uint32_t y, uint32_t z) {
        emit_instruction(module.execution_modes, 6, 16, 
                        {entry_point, 17, x, y, z});  // 17 = LocalSize
    }
    
    // OpExecutionModeId %main LocalSizeId %x %y %z (for spec constants)
    void emit_local_size_id(uint32_t entry_point, uint32_t x_id, uint32_t y_id, uint32_t z_id) {
        emit_instruction(module.execution_modes, 5, 279,
                        {entry_point, 38, x_id, y_id, z_id});  // 38 = LocalSizeId
    }
};

class SPIRVBuiltins {
    SPIRVModule& module;
    SPIRVTypeEmitter& type_emitter;
    std::unordered_map<std::string, uint32_t> builtin_cache;
    
public:
    SPIRVBuiltins(SPIRVModule& m, SPIRVTypeEmitter& te) 
        : module(m), type_emitter(te) {}
    
    uint32_t get_global_invocation_id() {
        if (auto it = builtin_cache.find("GlobalInvocationId"); it != builtin_cache.end()) {
            return it->second;
        }
        
        // Create type: uvec3 (vector of 3 uint32s)
        uint32_t uint_type = type_emitter.emit_int(32, false);
        uint32_t uvec3_type = emit_vector_type(uint_type, 3);
        
        // Create pointer to Input storage class
        uint32_t ptr_type = type_emitter.emit_pointer(1, uvec3_type);  // 1 = Input
        
        // OpVariable
        uint32_t var_id = module.allocate_id();
        emit_instruction(module.variables, 4, 59, {ptr_type, var_id, 1});  // OpVariable
        
        // OpDecorate %var BuiltIn GlobalInvocationId (28)
        emit_instruction(module.annotations, 4, 71, {var_id, 11, 28});  // 11=BuiltIn, 28=GlobalInvocationId
        
        builtin_cache["GlobalInvocationId"] = var_id;
        return var_id;
    }
    
    uint32_t get_local_invocation_id() {
        if (auto it = builtin_cache.find("LocalInvocationId"); it != builtin_cache.end()) {
            return it->second;
        }
        
        uint32_t uint_type = type_emitter.emit_int(32, false);
        uint32_t uvec3_type = emit_vector_type(uint_type, 3);
        uint32_t ptr_type = type_emitter.emit_pointer(1, uvec3_type);
        
        uint32_t var_id = module.allocate_id();
        emit_instruction(module.variables, 4, 59, {ptr_type, var_id, 1});
        emit_instruction(module.annotations, 4, 71, {var_id, 11, 27});  // 27=LocalInvocationId
        
        builtin_cache["LocalInvocationId"] = var_id;
        return var_id;
    }
    
    uint32_t get_workgroup_id() {
        if (auto it = builtin_cache.find("WorkgroupId"); it != builtin_cache.end()) {
            return it->second;
        }
        
        uint32_t uint_type = type_emitter.emit_int(32, false);
        uint32_t uvec3_type = emit_vector_type(uint_type, 3);
        uint32_t ptr_type = type_emitter.emit_pointer(1, uvec3_type);
        
        uint32_t var_id = module.allocate_id();
        emit_instruction(module.variables, 4, 59, {ptr_type, var_id, 1});
        emit_instruction(module.annotations, 4, 71, {var_id, 11, 26});  // 26=WorkgroupId
        
        builtin_cache["WorkgroupId"] = var_id;
        return var_id;
    }
    
private:
    uint32_t emit_vector_type(uint32_t component_type, uint32_t count) {
        std::string key = "vec" + std::to_string(count) + "_" + std::to_string(component_type);
        
        // Check type emitter cache first
        // For simplicity, creating directly:
        uint32_t id = module.allocate_id();
        emit_instruction(module.types, 4, 23, {id, component_type, count});  // OpTypeVector
        return id;
    }
};

struct uniform{
        struct entry{
            types type;
            std::string name;
            uint64_t offset, range;
            bool custom_type;
            std::string type_name; // only valid if custom_type == true;
        };
        std::vector<entry> items; // items in the uniform buffer

        void add_item(entry e){
            items.push_back(e);
        }

        void serialize_uniform(std::string& glsl_codestream){
            glsl_codestream += "struct uniform_buffer { \n";
            for (auto& e : items){
                if(!e.custom_type){
                    glsl_codestream += "   " + type_to_string()[e.type] + " " + e.name + "; \n";
                }else{
                    glsl_codestream += "   " + e.type_name + " " + e.name + "; \n";
                }
            }
            glsl_codestream += "};\n";
        };

        entry find(const std::string& name) {
            for (auto& e : items){
                if (e.name == name) return e;
            }
            throw std::runtime_error("no entry with name " + name + " exists in shader's uniform buffer. Did you forget to name it?");
        }
    };

class SPIRVUniformBuffer {
    SPIRVModule& module;
    SPIRVTypeEmitter& type_emitter;
    
public:
    SPIRVUniformBuffer(SPIRVModule& m, SPIRVTypeEmitter& te)
        : module(m), type_emitter(te) {}
    
    // Create the uniform buffer struct type
    uint32_t emit_uniform_buffer_struct(const uniform& uniform_buffer) {
        std::vector<uint32_t> member_types;
        
        // Collect all member types
        for (const auto& entry : uniform_buffer.items) {
            uint32_t member_type;
            
            if (!entry.custom_type) {
                // Standard type
                member_type = type_emitter.get_type_for(entry.type);
            } else {
                // Custom type (buffer reference)
                // This should be a uint64 (physical pointer)
                uint32_t element_type_id = type_emitter.get_type_for(types::FLOAT32);
                uint32_t array_id = type_emitter.emit_runtime_array(element_type_id);
                uint32_t struct_id = type_emitter.emit_struct({array_id});
                member_type = type_emitter.emit_pointer(5349, struct_id);
                emit_instruction(module.types, 3, 39, {member_type, 5349});
            }
            
            member_types.push_back(member_type);
        }
        
        // OpTypeStruct
        uint32_t struct_id = type_emitter.emit_struct(member_types);
        uint32_t wrapper_struct_id = type_emitter.emit_struct({struct_id});
        
        // Add decorations for the struct
        decorate_uniform_struct(struct_id, uniform_buffer);
        
        return wrapper_struct_id;
    }
    
    void decorate_uniform_struct(uint32_t struct_id, const uniform& uniform_buffer) {
        // OpDecorate %struct Block
        emit_instruction(module.annotations, 3, 71, {struct_id, 2});  // 2 = Block
        
        // Decorate each member with offset
        for (size_t i = 0; i < uniform_buffer.items.size(); i++) {
            const auto& entry = uniform_buffer.items[i];
            
            // OpMemberDecorate %struct member Offset offset
            emit_instruction(module.annotations, 5, 72, 
                           {struct_id, static_cast<uint32_t>(i), 35, 
                            static_cast<uint32_t>(entry.offset)});  // 35 = Offset
        }
    }
    
    // Create the uniform buffer variable
    uint32_t emit_uniform_buffer_variable(uint32_t struct_type_id, 
                                          uint32_t binding = 0, 
                                          uint32_t set = 0) {
        // Create pointer to struct in PhysicalStorageBuffer storage class
        uint32_t ptr_type = type_emitter.emit_pointer(5349, struct_type_id);  // 5349 = PhysicalStorageBuffer
        
        // Create pointer to the pointer in PushConstant storage class (9)
        // This is how we pass the buffer reference in
        uint32_t push_ptr_type = type_emitter.emit_pointer(9, ptr_type);
        
        // OpVariable
        uint32_t var_id = module.allocate_id();
        emit_instruction(module.variables, 4, 59, {push_ptr_type, var_id, 9});  // OpVariable in PushConstant
        
        return var_id;
    }
    
    // Alternative: Using descriptor set binding
    uint32_t emit_uniform_buffer_variable_descriptor(uint32_t struct_type_id,
                                                     uint32_t binding = 0,
                                                     uint32_t set = 0) {
        // Wrap struct in a runtime array or use directly
        // Create pointer in Uniform storage class (2)
        uint32_t ptr_type = type_emitter.emit_pointer(2, struct_type_id);
        
        // OpVariable
        uint32_t var_id = module.allocate_id();
        emit_instruction(module.variables, 4, 59, {ptr_type, var_id, 2});
        
        // OpDecorate %var DescriptorSet set
        emit_instruction(module.annotations, 4, 71, {var_id, 34, set});  // 34 = DescriptorSet
        
        // OpDecorate %var Binding binding
        emit_instruction(module.annotations, 4, 71, {var_id, 33, binding});  // 33 = Binding
        
        return var_id;
    }
};

class SPIRVBufferReference {
    SPIRVModule& module;
    SPIRVTypeEmitter& type_emitter;
    std::unordered_map<std::string, uint32_t> buffer_struct_cache;
    
public:
    SPIRVBufferReference(SPIRVModule& m, SPIRVTypeEmitter& te)
        : module(m), type_emitter(te) {}
    
    uint32_t create_buffer_struct(types element_type, const std::string& name) {
        if (auto it = buffer_struct_cache.find(name); it != buffer_struct_cache.end()) {
            return it->second;
        }
        
        // Get base element type
        uint32_t elem_type_id = type_emitter.get_type_for(element_type);
        
        // Create runtime array: T data[]
        uint32_t array_id = type_emitter.emit_runtime_array(elem_type_id);
        
        // Decorate array with ArrayStride
        uint32_t stride = type_to_byte_ranges()[element_type];
        emit_instruction(module.annotations, 4, 71, {array_id, 6, stride});  // 6 = ArrayStride
        
        // Create struct { T data[]; }
        uint32_t struct_id = type_emitter.emit_struct({array_id});
        
        // Decorate struct
        emit_instruction(module.annotations, 3, 71, {struct_id, 2});  // Block decoration
        
        // Decorate member 0 with Offset 0
        emit_instruction(module.annotations, 5, 72, {struct_id, 0, 35, 0});  // Offset
        
        // OpName for debugging
        emit_debug_name(struct_id, name);
        
        buffer_struct_cache[name] = struct_id;
        return struct_id;
    }
    
    uint32_t create_buffer_pointer(uint32_t struct_id) {
        // Physical storage buffer pointer
        return type_emitter.emit_pointer(5349, struct_id);
    }
    
private:
    void emit_debug_name(uint32_t target_id, const std::string& name) {
        std::vector<uint32_t> operands = {target_id};
        auto name_words = string_to_words(name);
        operands.insert(operands.end(), name_words.begin(), name_words.end());
        
        emit_instruction(module.debug_strings, 1 + operands.size(), 5, operands);  // OpName
    }
};

class SPIRVFunctionBuilder {
    SPIRVModule& module;
    SPIRVTypeEmitter& type_emitter;
    
    uint32_t current_label_id = 0;
    std::vector<uint32_t> current_function_body;
    
public:
    SPIRVFunctionBuilder(SPIRVModule& m, SPIRVTypeEmitter& te)
        : module(m), type_emitter(te) {}
    
    // Create void main() function type
    uint32_t emit_main_function_type() {
        uint32_t void_type = type_emitter.emit_void();
        
        // OpTypeFunction %void
        uint32_t func_type_id = module.allocate_id();
        std::vector<uint32_t> operands = {func_type_id, void_type};
        emit_instruction(module.types, 2 + operands.size() - 1, 33, operands);  // OpTypeFunction
        
        return func_type_id;
    }
    
    // Begin the main function
    uint32_t begin_function(uint32_t function_id, uint32_t function_type_id) {
        uint32_t void_type = type_emitter.emit_void();
        
        // OpFunction %void None %func_type
        std::vector<uint32_t> operands = {void_type, function_id, 0, function_type_id};
        emit_instruction(current_function_body, 5, 54, operands);  // OpFunction, control=0 (None)
        
        // OpLabel - start of function body
        current_label_id = module.allocate_id();
        emit_instruction(current_function_body, 2, 248, {current_label_id});  // OpLabel
        
        return current_label_id;
    }
    
    // End the function
    void end_function() {
        // OpReturn
        emit_instruction(current_function_body, 1, 253, {});  // OpReturn
        
        // OpFunctionEnd
        emit_instruction(current_function_body, 1, 56, {});  // OpFunctionEnd
        
        // Move function body to module
        module.functions.insert(module.functions.end(), 
                               current_function_body.begin(), 
                               current_function_body.end());
        current_function_body.clear();
    }
    
    // Get current function body for adding instructions
    std::vector<uint32_t>& get_function_body() {
        return current_function_body;
    }
};

class SPIRVUniformAccess {
    SPIRVModule& module;
    SPIRVTypeEmitter& type_emitter;
    std::vector<uint32_t>& function_body;
    
public:
    SPIRVUniformAccess(SPIRVModule& m, SPIRVTypeEmitter& te, std::vector<uint32_t>& fb)
        : module(m), type_emitter(te), function_body(fb) {}
    
    // Load uniform buffer pointer from push constant
    // Returns the ID of the loaded uniform buffer struct pointer
    uint32_t load_uniform_buffer_from_push(uint32_t push_constant_var_id,
                                           uint32_t uniform_struct_type_id) {
        // The push constant variable is a pointer to a pointer
        // First, we need to load the actual pointer value
        
        // Get the pointer type (PhysicalStorageBuffer pointer to uniform struct)
        uint32_t uniform_ptr_type = type_emitter.emit_pointer(5349, uniform_struct_type_id);
        
        // OpLoad to get the pointer from push constant
        uint32_t loaded_ptr_id = module.allocate_id();
        emit_instruction(function_body, 4, 61, {uniform_ptr_type, loaded_ptr_id, push_constant_var_id});  // OpLoad
        
        return loaded_ptr_id;
    }
    
    // Access a member of the uniform buffer
    // Returns a pointer to the member
    uint32_t access_uniform_member(uint32_t uniform_ptr_id,
                                   uint32_t uniform_struct_type_id,
                                   uint32_t member_index,
                                   uint32_t member_type_id) {
        // Create pointer type for the member in PhysicalStorageBuffer
        uint32_t member_ptr_type = type_emitter.emit_pointer(5349, member_type_id);
        
        // OpAccessChain or OpInBoundsAccessChain
        // We need to use OpAccessChain with PhysicalStorageBuffer
        uint32_t access_chain_id = module.allocate_id();
        
        // Create constant for member index
        uint32_t int_type = type_emitter.emit_int(32, false);
        uint32_t index_constant = emit_constant_uint32(member_index);
        
        // OpAccessChain %member_ptr_type %result %uniform_ptr %index
        std::vector<uint32_t> operands = {
            member_ptr_type,
            access_chain_id,
            uniform_ptr_id,
            index_constant
        };
        emit_instruction(function_body, 3 + operands.size() - 2, 65, operands);  // OpAccessChain
        
        return access_chain_id;
    }
    
    // Load a value from uniform buffer member
    uint32_t load_uniform_member_value(uint32_t member_ptr_id, uint32_t value_type_id) {
        uint32_t loaded_value_id = module.allocate_id();
        emit_instruction(function_body, 4, 61, {value_type_id, loaded_value_id, member_ptr_id});  // OpLoad
        
        return loaded_value_id;
    }
    
private:
    uint32_t emit_constant_uint32(uint32_t value) {
        // Check if constant already exists in module constants
        // For simplicity, creating new one each time
        
        uint32_t uint_type = type_emitter.emit_int(32, false);
        uint32_t const_id = module.allocate_id();
        
        emit_instruction(module.constants, 4, 43, {uint_type, const_id, value});  // OpConstant
        
        return const_id;
    }
};

struct Shader {

    struct global_mem{
        types type;
        std::string name;
        std::string uniform_name;
    };

    struct index_macro;

    struct shared_mem{
        types type;
        uint64_t size;
        std::string name;

        std::string at(index_macro idx);
    };

    
    std::vector<global_mem> declared_buffers;
    std::vector<shared_mem> SRAM_buffers;
    uint32_t version;
    std::vector<std::string> extensions_requested;
    std::array<uint32_t, 3> block_layout_size;
    uniform uniform_buffer;
    
    SPIRVModule spirv_module;
    SPIRVTypeEmitter type_emitter{spirv_module};
    SPIRVEntryPoint entry_point_emitter{spirv_module};
    SPIRVBuiltins builtins_emitter{spirv_module, type_emitter};
    SPIRVUniformBuffer uniform_emitter{spirv_module, type_emitter};
    SPIRVBufferReference buffer_ref_emitter{spirv_module, type_emitter};
    SPIRVFunctionBuilder function_builder{spirv_module, type_emitter};
    
    uint32_t main_function_id;
    uint32_t uniform_buffer_var_id;
    uint32_t uniform_struct_type_id;
    std::vector<uint32_t> interface_variables;
    
    // Map buffer names to their indices in the uniform buffer
    std::unordered_map<std::string, uint32_t> buffer_to_uniform_index;

    void begin_spirv_generation() {
        // Setup capabilities and extensions
        setup_buffer_reference_module(spirv_module);
        
        // Add required capabilities
        uint32_t glsl_std_450_id = spirv_module.allocate_id();
        emit_instruction(spirv_module.capabilities, 6, 11, {glsl_std_450_id,          // result-id
                                                            0x4C534C47,
                                                            0x6474732E,
                                                            0x3035342E,
                                                            0x00000000});
    }

    void declare_buffer_spirv(const global_mem& buffer, size_t index) {
        // Create the buffer struct type
        uint32_t struct_id = buffer_ref_emitter.create_buffer_struct(
            buffer.type, buffer.name);
        
        // Store the mapping from buffer name to uniform index
        buffer_to_uniform_index[buffer.name] = index;
    }

    void create_uniform_buffer_spirv() {
        // First, declare all buffer struct types
        for (size_t i = 0; i < declared_buffers.size(); i++) {
            declare_buffer_spirv(declared_buffers[i], i);
        }
        
        // Create the uniform buffer struct type
        uniform_struct_type_id = uniform_emitter.emit_uniform_buffer_struct(uniform_buffer);
        
        // Create the push constant variable that holds pointer to uniform buffer
        uniform_buffer_var_id = create_push_constant_for_uniform(uniform_struct_type_id);
    }

    uint32_t create_push_constant_for_uniform(uint32_t uniform_struct_id) {
        // Create pointer to uniform struct in PhysicalStorageBuffer
        uint32_t uniform_ptr_type = type_emitter.emit_pointer(5349, uniform_struct_id);
        
        // Wrap in a struct for push constant
        uint32_t push_struct_id = type_emitter.emit_struct({uniform_ptr_type});
        
        emit_instruction(spirv_module.types, 3, 39, {uniform_ptr_type, 5349});

        // Decorate push constant struct
        emit_instruction(spirv_module.annotations, 3, 71, {push_struct_id, 2});  // Block
        emit_instruction(spirv_module.annotations, 5, 72, {push_struct_id, 0, 35, 0});  // Offset 0
        
        // Create pointer to push constant struct
        uint32_t push_ptr_type = type_emitter.emit_pointer(9, push_struct_id);  // 9 = PushConstant
        
        // OpVariable
        uint32_t push_var_id = spirv_module.allocate_id();
        emit_instruction(spirv_module.variables, 4, 59, {push_ptr_type, push_var_id, 9});
        
        return push_var_id;
    }

    void create_entry_point() {
        // Allocate ID for main function
        main_function_id = spirv_module.allocate_id();
        
        // Get built-in variables
        uint32_t global_invocation_id = builtins_emitter.get_global_invocation_id();
        interface_variables.push_back(global_invocation_id);
        
        // Create entry point
        entry_point_emitter.emit_compute_entry("main", main_function_id, interface_variables);
        
        // Set local size
        entry_point_emitter.emit_local_size(main_function_id, 
                                           block_layout_size[0],
                                           block_layout_size[1],
                                           block_layout_size[2]);
    }

    void create_main_function() {
        // Create function type (void -> void)
        uint32_t func_type_id = function_builder.emit_main_function_type();
        
        // Begin function
        function_builder.begin_function(main_function_id, func_type_id);
        
        // Get function body reference
        auto& body = function_builder.get_function_body();
        
        // Create uniform access helper
        SPIRVUniformAccess uniform_access(spirv_module, type_emitter, body);
        
        uint32_t uniform_loaded_ptr_id = uniform_access.load_uniform_buffer_from_push(uniform_buffer_var_id, uniform_struct_type_id);
        
        // Store ctx_ptr_id for later use in load/store operations
        // For now, this is all we need - we've loaded the uniform buffer pointer
        
        // TODO: Add actual compute work here later
        
        // End function
        function_builder.end_function();
    }

    uint32_t get_push_constant_struct_type() {
        // This should match what we created in create_push_constant_for_uniform
        uint32_t uniform_ptr_type = type_emitter.emit_pointer(5349, uniform_struct_type_id);
        return type_emitter.emit_struct({uniform_ptr_type});
    }
    
    uint32_t emit_constant_uint32(uint32_t value) {
        uint32_t uint_type = type_emitter.emit_int(32, false);
        uint32_t const_id = spirv_module.allocate_id();
        
        emit_instruction(spirv_module.constants, 4, 43, {uint_type, const_id, value});
        
        return const_id;
    }

    void generate_spirv_module() {
        begin_spirv_generation();
        
        // Create uniform buffer (this will also declare all buffers)
        create_uniform_buffer_spirv();
        
        // Create entry point
        create_entry_point();
        
        // Create main function
        create_main_function();
        
        // Finalize
        finalize_spirv("output.spv");
    }
    
    void finalize_spirv(const std::string& output_file) {
        spirv_module.assemble();
        
        std::ofstream out(output_file, std::ios::binary);
        out.write(reinterpret_cast<const char*>(spirv_module.binary.data()),
                  spirv_module.binary.size() * sizeof(uint32_t));
    }

    void add_attr_to_uniform(uniform::entry e){
        uniform_buffer.add_item(e);
    }

    void request_extention(const std::string& extention){
        extensions_requested.push_back(extention);
    }

    void serialize_extentions(std::string& glsl_codestream){
        glsl_codestream += "#version " + std::to_string(version) + "\n";
        for (auto& e : extensions_requested){
            glsl_codestream += "#extension " + e + " : enable\n";
        }
    }

    void serialize_block_layout(std::string& glsl_codestream){
        glsl_codestream += "layout(";
        for (int i = 0; i < 3; i++){
            switch(i){
                case(0):
                    glsl_codestream += "local_size_x = " + std::to_string(block_layout_size[i]) + ", ";
                    break;
                case(1):
                    glsl_codestream += "local_size_y = " + std::to_string(block_layout_size[i]) + ", ";
                    break;
                case(2):
                    glsl_codestream += "local_size_z = " + std::to_string(block_layout_size[i]);
                    break;
                default:
                    break;
            }
        }
        glsl_codestream += ") in;\n";
    }

    void set_block_size(uint32_t x, uint32_t y, uint32_t z) {
        block_layout_size[0] = x;
        block_layout_size[1] = y;
        block_layout_size[2] = z;
    }
    
    global_mem declare_buffer(const types type, const std::string& name){
        uniform::entry e;
        e.name = "buffer" + std::to_string(uniform_buffer.items.size());
        declared_buffers.push_back({type, name, "ctx." + e.name});
        e.offset = uniform_buffer.items.empty() ? 0 : uniform_buffer.items.back().offset + uniform_buffer.items.back().range;
        e.range = 8; // because this is a 64 bit integer pointer
        e.type = type;
        e.custom_type = true;
        e.type_name = name;
        uniform_buffer.add_item(e);
        return {type, name, "ctx." + e.name}; // returns an exact copy
    }

    shared_mem declare_shared(const types type, const std::string& name, uint64_t size){
        SRAM_buffers.push_back({type, size, name});
        return {type, size, name}; // returns an exact copy
    }

    void serialize_uniform(std::string& glsl_codestream){
        uniform_buffer.serialize_uniform(glsl_codestream);
        glsl_codestream += "layout(buffer_reference, std430, scalar) buffer UniformBuffer{ uniform_buffer ub; };\n";
    }

    void serialize_declared_buffers(std::string& glsl_codestream){
        for (auto& buffer : declared_buffers){
            switch (buffer.type)
            {
            case types::FLOAT32:
                glsl_codestream += "layout(buffer_reference, std430, scalar) buffer " + buffer.name + "{ float data[]; };\n";
                break;
            case types::INT32:
                glsl_codestream += "layout(buffer_reference, std430, scalar) buffer " + buffer.name + "{ int data[]; };\n";
                break;
            default:
                throw std::runtime_error("type not supported yet");
                break;
            }
        }
    }

    void serialize_declared_SRAM_buffers(std::string& glsl_codestream){
        for (auto& buffer : SRAM_buffers){
            switch (buffer.type){
            case types::FLOAT32:
                glsl_codestream += "shared float " + buffer.name + "[" + std::to_string(buffer.size) + "];\n";
                break;
            case types::INT32:
                glsl_codestream += "shared int " + buffer.name + "[" + std::to_string(buffer.size) + "];\n";
                break;
            default:
                throw std::runtime_error("type not supported yet");
                break;
            }
        }
    }

    struct index_macro {
        enum class IDs {
            PROGRAM_ID_X,
            PROGRAM_ID_Y,
            PROGRAM_ID_Z,
            RANGE,
            SCALAR,
            EXPRESSION,
            UNIFORM
        };

        struct range_desc {
            std::shared_ptr<index_macro> start;    // base expression
            std::shared_ptr<index_macro> stride;   // stride expression
            std::shared_ptr<index_macro> count;    // number of iterations
            std::string lane_var = "i";
        };

        IDs type = IDs::SCALAR;
        std::string expr;     // scalar / base expression
        range_desc range;     // valid iff type == RANGE

        /* ===================== Factories ===================== */

        static index_macro program_id(int id) {
            index_macro m;
            switch (id) {
                case 0: m.type = IDs::PROGRAM_ID_X; m.expr = "uint(gl_WorkGroupID.x)"; break;
                case 1: m.type = IDs::PROGRAM_ID_Y; m.expr = "uint(gl_WorkGroupID.y)"; break;
                case 2: m.type = IDs::PROGRAM_ID_Z; m.expr = "uint(gl_WorkGroupID.z)"; break;
                default: throw std::runtime_error("program_id must be 0,1,2");
            }
            return m;
        }

        static index_macro scalar(const std::string& e) {
            return index_macro(e);
        }

        static index_macro arange(index_macro start,
                          index_macro count,
                          index_macro stride = index_macro("1"))
        {
            index_macro m;
            m.type = IDs::RANGE;
            m.range.start  = std::make_shared<index_macro>(start);
            m.range.count  = std::make_shared<index_macro>(count);
            m.range.stride = std::make_shared<index_macro>(stride);

            // Canonical base expression
            m.expr = "(" + start.expr + " + " +
                    m.range.lane_var + " * " +
                    stride.expr + ")";

            return m;
        }

        // Backward-compatible constant arange
        static index_macro arange(int start, int end, int stride = 1) {
            if (stride <= 0 || end <= start)
                throw std::runtime_error("Invalid arange");

            int count = (end - start + stride - 1) / stride;
            return arange(
                index_macro(std::to_string(start)),
                index_macro(std::to_string(count)),
                index_macro(std::to_string(stride))
            );
        }

        static index_macro from_uniform(uniform& u, const std::string& name) {
            auto entry = u.find(name);
            index_macro m;
            m.type = IDs::UNIFORM;
            m.expr = "ctx." + entry.name;
            return m;
        }

        explicit index_macro(const std::string& name)
            : type(IDs::SCALAR), expr(name) {}

        index_macro() = default;

        /* ===================== Validation ===================== */

        static void validate_combine(const index_macro& a,
                                    const index_macro& b)
        {
            if (a.type == IDs::RANGE && b.type == IDs::RANGE)
                throw std::runtime_error("Cannot combine two RANGE indices");
        }

        /* ===================== Combine ===================== */

        static index_macro combine(const index_macro& a,
                                const index_macro& b,
                                const std::string& e)
        {
            validate_combine(a, b);

            index_macro m;
            m.expr = e;

            if (a.type == IDs::RANGE || b.type == IDs::RANGE) {
                m.type  = IDs::RANGE;
                m.range = (a.type == IDs::RANGE) ? a.range : b.range;
            } else {
                m.type = IDs::EXPRESSION;
            }
            return m;
        }

        index_macro operator+(const index_macro& o) const {
            return combine(*this, o, "(" + expr + " + " + o.expr + ")");
        }
        index_macro operator*(const index_macro& o) const {
            return combine(*this, o, "(" + expr + " * " + o.expr + ")");
        }
        index_macro operator/(const index_macro& o) const {
            return combine(*this, o, "(" + expr + " / " + o.expr + ")");
        }
        index_macro operator%(const index_macro& o) const {
            return combine(*this, o, "(" + expr + " % " + o.expr + ")");
        }

        /* ===================== Codegen ===================== */

        std::string emit_index_expr() const {
            return expr;
        }

        std::string emit_loop_header() const {
            return "for (uint " + range.lane_var + " = 0; " +
                range.lane_var + " < " + range.count->expr + "; ++" +
                range.lane_var + ")";
        }

        /* ===================== Loads / Stores ===================== */

        std::string generate_load_vec(const std::string& buffer,
                                    const std::string& load_to,
                                    const std::string& mask_cond) const
        {
            if (type != IDs::RANGE)
                throw std::runtime_error("Vector load requires RANGE index");

            std::string g;
            g += "// Triton-style vectorized load\n";
            g += emit_loop_header() + " {\n";
            g += "    uint idx = " + emit_index_expr() + ";\n";

            if (mask_cond.empty()) {
                g += "    " + load_to + "[" + range.lane_var + "] = " +
                    buffer + ".data[idx];\n";
            } else {
                g += "    if (" + mask_cond + ") {\n";
                g += "        " + load_to + "[" + range.lane_var + "] = " +
                    buffer + ".data[idx];\n";
                g += "    } else {\n";
                g += "        " + load_to + "[" + range.lane_var + "] = 0;\n";
                g += "    }\n";
            }

            g += "}\n";
            return g;
        }

        std::string generate_store_vec(const std::string& store_from,
                                    const std::string& buffer,
                                    const std::string& mask_cond) const
        {
            if (type != IDs::RANGE)
                throw std::runtime_error("Vector store requires RANGE index");

            std::string g;
            g += "// Triton-style vectorized store\n";
            g += emit_loop_header() + " {\n";
            g += "    uint idx = " + emit_index_expr() + ";\n";

            if (mask_cond.empty()) {
                g += "    " + buffer + ".data[idx] = " +
                    store_from + "[" + range.lane_var + "];\n";
            } else {
                g += "    if (" + mask_cond + ") {\n";
                g += "        " + buffer + ".data[idx] = " +
                    store_from + "[" + range.lane_var + "];\n";
                g += "    }\n";
            }

            g += "}\n";
            return g;
        }

        std::string generate_load_scalar(const std::string& buffer,
                                        const std::string& load_to,
                                        const std::string& mask_cond) const
        {
            std::string g;
            g += "// Triton-style scalar load\n";
            g += "uint idx = " + expr + ";\n";

            if (mask_cond.empty()) {
                g += load_to + " = " + buffer + ".data[idx];\n";
            } else {
                g += "if (" + mask_cond + ") {\n";
                g += "    " + load_to + " = " + buffer + ".data[idx];\n";
                g += "} else {\n";
                g += "    " + load_to + " = 0;\n";
                g += "}\n";
            }
            return g;
        }

        std::string generate_store_scalar(const std::string& store_from,
                                        const std::string& buffer,
                                        const std::string& mask_cond) const
        {
            std::string g;
            g += "// Triton-style scalar store\n";
            g += "uint idx = " + expr + ";\n";

            if (mask_cond.empty()) {
                g += buffer + ".data[idx] = " + store_from + ";\n";
            } else {
                g += "if (" + mask_cond + ") {\n";
                g += "    " + buffer + ".data[idx] = " + store_from + ";\n";
                g += "}\n";
            }
            return g;
        }
    };

    struct for_loop {
        index_macro iterator;   // scalar loop variable (e.g. "i")
        index_macro indx;       // RANGE index_macro
        std::string body;

        for_loop(const index_macro& iter, const index_macro& idx)
            : iterator(iter), indx(idx)
        {
            if (idx.type != index_macro::IDs::RANGE)
                throw std::runtime_error("for_loop requires RANGE index_macro");

            if (iter.type != index_macro::IDs::SCALAR &&
                iter.type != index_macro::IDs::EXPRESSION)
                throw std::runtime_error("for_loop iterator must be SCALAR");

            // Bind iterator symbol to range lane variable
            indx.range.lane_var = iterator.expr;

            // Emit loop header immediately
            body += indx.emit_loop_header();
            body += " {\n";
        }

        void add_instruction(const std::string& instruction) {
            body += "    " + instruction;
            if (!instruction.empty() && instruction.back() != '\n')
                body += "\n";
        }

        void end_for_loop() {
            body += "}\n";
        }

        void serialize_for_loop(std::string& glsl_codestream) const {
            glsl_codestream += body;
        }

        std::string get_loop_code() const {
            return body;
        }
    };

    struct local_var {
        std::string name;
        types type;
        bool isArray = false;
        index_macro size;   // valid iff isArray

        /* ===================== Ctors ===================== */

        local_var(const std::string& n, types t)
            : name(n), type(t) {}

        local_var(const std::string& n, types t, const index_macro& sz)
            : name(n), type(t), isArray(true), size(sz) {}

    private:
        /* ===================== Helpers ===================== */

        std::string lhs(const index_macro* idx = nullptr) const {
            if (idx) {
                if (!isArray)
                    throw std::runtime_error("Scalar local_var cannot be indexed");
                return name + "[" + idx->expr + "]";
            }
            return name;
        }

        static std::string bin(const std::string& a,
                            const std::string& op,
                            const std::string& b)
        {
            return "(" + a + " " + op + " " + b + ")";
        }

        static std::string stmt(const std::string& a,
                                const std::string& op,
                                const std::string& b)
        {
            return a + " " + op + " " + b + ";\n";
        }

    public:
        /* ===================== Declaration ===================== */

        std::string decl() const {
            if (isArray) {
                return type_to_string()[type] + " " + name +
                    "[" + size.expr + "];\n";
            }
            return type_to_string()[type] + " " + name + ";\n";
        }

        std::string decl_assign(const std::string& rhs) const {
            if (isArray)
                throw std::runtime_error("Cannot assign array variable");
            return type_to_string()[type] + " " + name + " = " + rhs + ";\n";
        }

        /* ===================== Indexing ===================== */

        std::string at(const index_macro& idx) const {
            return lhs(&idx);
        }

        /* ===================== Pure expressions ===================== */

        // scalar ⨉ scalar
        std::string add(const local_var& o) const { return bin(lhs(), "+", o.lhs()); }
        std::string sub(const local_var& o) const { return bin(lhs(), "-", o.lhs()); }
        std::string mul(const local_var& o) const { return bin(lhs(), "*", o.lhs()); }
        std::string div(const local_var& o) const { return bin(lhs(), "/", o.lhs()); }

        // array element ⨉ expression
        std::string add(const index_macro& idx, const std::string& rhs) const {
            return bin(lhs(&idx), "+", rhs);
        }
        std::string sub(const index_macro& idx, const std::string& rhs) const {
            return bin(lhs(&idx), "-", rhs);
        }
        std::string mul(const index_macro& idx, const std::string& rhs) const {
            return bin(lhs(&idx), "*", rhs);
        }
        std::string div(const index_macro& idx, const std::string& rhs) const {
            return bin(lhs(&idx), "/", rhs);
        }

        // array element ⨉ local_var
        std::string add(const index_macro& idx, const local_var& o) const {
            return add(idx, o.lhs());
        }
        std::string mul(const index_macro& idx, const local_var& o) const {
            return mul(idx, o.lhs());
        }

        /* ===================== Assignments ===================== */

        // scalar
        std::string assign(const std::string& rhs) const {
            if (isArray)
                throw std::runtime_error("Use indexed assign for arrays");
            return stmt(lhs(), "=", rhs);
        }

        std::string add_assign(const std::string& rhs) const {
            if (isArray)
                throw std::runtime_error("Use indexed add_assign for arrays");
            return stmt(lhs(), "+=", rhs);
        }

        std::string mul_assign(const std::string& rhs) const {
            if (isArray)
                throw std::runtime_error("Use indexed mul_assign for arrays");
            return stmt(lhs(), "*=", rhs);
        }

        /* ===================== Indexed assignments ===================== */

        std::string assign(const index_macro& idx,
                        const std::string& rhs) const
        {
            return stmt(lhs(&idx), "=", rhs);
        }

        std::string add_assign(const index_macro& idx,
                            const std::string& rhs) const
        {
            return stmt(lhs(&idx), "+=", rhs);
        }

        std::string mul_assign(const index_macro& idx,
                            const std::string& rhs) const
        {
            return stmt(lhs(&idx), "*=", rhs);
        }
    };

    void load(global_mem& buffer, shared_mem& load_to, index_macro idx, std::string& glsl_codestream) {
        if(idx.type == index_macro::IDs::RANGE){
            glsl_codestream += idx.generate_load_vec(buffer.uniform_name, load_to.name, "");
        }else glsl_codestream += idx.generate_load_scalar(buffer.uniform_name, load_to.name, "");
    }

    std::string load(global_mem& buffer, shared_mem& load_to, index_macro idx) {
        if(idx.type == index_macro::IDs::RANGE){
            return idx.generate_load_vec(buffer.uniform_name, load_to.name, "");
        }else return idx.generate_load_scalar(buffer.uniform_name, load_to.name, "");
    }

    void store(shared_mem& store_from, global_mem& buffer, index_macro idx, std::string& glsl_codestream) {
        if (idx.type == index_macro::IDs::RANGE) {
            glsl_codestream += idx.generate_store_vec(
                store_from.name, buffer.uniform_name, "");
        } else {
            glsl_codestream += idx.generate_store_scalar(
                store_from.name, buffer.uniform_name, "");
        }
    }

    std::string store(shared_mem& store_from, global_mem& buffer, index_macro idx) {
        if (idx.type == index_macro::IDs::RANGE) {
            return idx.generate_store_vec(
                store_from.name, buffer.uniform_name, "");
        } else {
            return idx.generate_store_scalar(
                store_from.name, buffer.uniform_name, "");
        }
    }

    void dot(shared_mem rhs, shared_mem lhs, shared_mem result, index_macro idx){

    }

    void decl_main(std::string& glsl_codestream) {
        serialize_extentions(glsl_codestream);
        serialize_block_layout(glsl_codestream);
        serialize_declared_buffers(glsl_codestream);
        serialize_uniform(glsl_codestream);

        glsl_codestream += "layout(push_constant) uniform PushConstants { UniformBuffer ctx; } push;\n";

        serialize_declared_SRAM_buffers(glsl_codestream);

        glsl_codestream += "void main(){ \n";
        glsl_codestream += "uniform_buffer ctx = push.ctx.ub;\n";
    }
};
using idx = Shader::index_macro;
using loop = Shader::for_loop;
using var = Shader::local_var;

std::string Shader::shared_mem::at(idx index){
    return name + "[" + index.expr + "]";
}
