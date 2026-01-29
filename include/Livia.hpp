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

    struct uniform{
        struct entry{
            types type;
            std::string name;
            uint64_t offset, range;
        };
        std::vector<entry> items; // items in the uniform buffer

        void add_item(entry e){
            items.push_back(e);
        }

        void serialize_uniform(std::string& glsl_codestream){
            glsl_codestream += "struct uniform_buffer { \n";
            for (auto& e : items){
                glsl_codestream += "   " + type_to_string()[e.type] + " " + e.name + "; \n";
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

    std::vector<global_mem> declared_buffers;
    std::vector<shared_mem> SRAM_buffers;
    uint32_t version;
    std::vector<std::string> extensions_requested;
    std::array<uint32_t, 3> block_layout_size;

    uniform uniform_buffer;

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
        e.range = type_to_byte_ranges()[type];
        e.type = type;
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