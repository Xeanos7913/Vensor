# Compiler
CXX = g++

# Debug/Release flags
DEBUG_FLAGS = -std=c++20 -g -O0 -Iinclude -I/usr/include/glm -Wno-shadow-template -DDEBUG
RELEASE_FLAGS = -std=c++20 -O2 -Iinclude -I/usr/include/glm -Wno-shadow-template

# Default to release
CXXFLAGS = $(RELEASE_FLAGS)

# Libraries
LIBS = -lglfw -lvulkan

# Shader compiler
GLSLANG = glslangValidator

# Source files
SRC_DIR = src
CPP_SRC = $(wildcard $(SRC_DIR)/*.cpp)
C_SRC = $(wildcard $(SRC_DIR)/*.c)
SRC = $(CPP_SRC) $(C_SRC)
OBJ = $(SRC:.cpp=.o)
OBJ := $(OBJ:.c=.o)

# Shaders
SHADER_DIR = compiled_shaders
SHADERS_VERT = $(wildcard shaders/*.vert)
SHADERS_FRAG = $(wildcard shaders/*.frag)
SHADERS_COMP = $(wildcard shaders/*.comp)

SPV_VERT = $(patsubst shaders/%.vert, $(SHADER_DIR)/%.vert.spv, $(SHADERS_VERT))
SPV_FRAG = $(patsubst shaders/%.frag, $(SHADER_DIR)/%.frag.spv, $(SHADERS_FRAG))
SPV_COMP = $(patsubst shaders/%.comp, $(SHADER_DIR)/%.comp.spv, $(SHADERS_COMP))

SPV = $(SPV_VERT) $(SPV_FRAG) $(SPV_COMP)

# Target
TARGET = app

all: $(SPV) $(TARGET)

# Debug build
debug: CXXFLAGS = $(DEBUG_FLAGS)
debug: clean $(SPV) $(TARGET)

# Ensure shader folder exists
$(SHADER_DIR):
	mkdir -p $(SHADER_DIR)

# Compile shaders - separate rules for each type
$(SHADER_DIR)/%.vert.spv: shaders/%.vert | $(SHADER_DIR)
	$(GLSLANG) -V --target-env vulkan1.1 $< -o $@

$(SHADER_DIR)/%.frag.spv: shaders/%.frag | $(SHADER_DIR)
	$(GLSLANG) -V --target-env vulkan1.1 $< -o $@

$(SHADER_DIR)/%.comp.spv: shaders/%.comp | $(SHADER_DIR)
	$(GLSLANG) -V --target-env vulkan1.1 $< -o $@

clean:
	rm -f $(OBJ) $(TARGET) $(SHADER_DIR)/*.spv

.PHONY: all debug clean