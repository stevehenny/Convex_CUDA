# Compiler and flags
NVCC = nvcc
CXXFLAGS = -Isrc/ -lstdc++ -std=c++17 -Xcompiler=-Wno-deprecated -g -G --extended-lambda 


CXXFLAGS_OPT = -O3 -Isrc/ -lstdc++ -std=c++17 -Xcompiler=-Wno-deprecated -g -G --extended-lambda 

# Source files
SRC_FILES = src/main.cu src/convex_hull_general.cpp src/convex_hull_serial.cpp src/convex_hull_cuda.cu

# Output directory and executable
OUTPUT_DIR = bin
VISUAL_OUTPUT = $(OUTPUT_DIR)/visual.o
NO_VISUAL_OUTPUT = $(OUTPUT_DIR)/no_visual.o
VISUAL_OPTIMIZED = $(OUTPUT_DIR)/visual_optimized.o 
NO_VISUAL_OPTIMIZED = $(OUTPUT_DIR)/no_visual_optimized.o 

# Targets
all: visual no_visual visual_optimized no_visual_optimized 

# 'visual' target will set USE_OPENGL=1 and include OpenGL-specific flags
visual:
	@mkdir -p $(OUTPUT_DIR)        # Create the bin directory if it doesn't exist
	$(eval VISUAL_DEFINES = -DUSE_OPENGL) # Define USE_OPENGL for conditional compilation
	$(eval VISUAL_CXXFLAGS = $(CXXFLAGS) -lglut -lGL -lGLU) # Add OpenGL libraries for visual
	$(NVCC) $(SRC_FILES) $(VISUAL_CXXFLAGS) $(VISUAL_DEFINES) -o $(VISUAL_OUTPUT)

# 'no_visual' target will set USE_OPENGL=0 and disable OpenGL-specific flags
no_visual:
	@mkdir -p $(OUTPUT_DIR)        # Create the bin directory if it doesn't exist
	$(eval NO_VISUAL_DEFINES = -DNO_OPENGL) # Define NO_OPENGL for conditional compilation
	$(eval NO_VISUAL_CXXFLAGS = $(CXXFLAGS)) # No OpenGL libraries for no_visual
	$(NVCC) $(SRC_FILES) $(NO_VISUAL_CXXFLAGS) $(NO_VISUAL_DEFINES) -o $(NO_VISUAL_OUTPUT)

visual_optimized:
	@mkdir -p $(OUTPUT_DIR)        # Create the bin directory if it doesn't exist
	$(eval VISUAL_DEFINES = -DUSE_OPENGL) # Define USE_OPENGL for conditional compilation
	$(eval VISUAL_CXXFLAGS_OPT = $(CXXFLAGS_OPT) -lglut -lGL -lGLU) # Add OpenGL libraries for visual 
	$(NVCC) $(SRC_FILES) $(VISUAL_CXXFLAGS_OPT) $(VISUAL_DEFINES) -o $(VISUAL_OPTIMIZED)

no_visual_optimized:
	@mkdir -p $(OUTPUT_DIR)        # Create the bin directory if it doesn't exist
	$(eval NO_VISUAL_DEFINES = -DNO_OPENGL) # Define NO_OPENGL for conditional compilation
	$(eval NO_VISUAL_CXXFLAGS_OPT = $(CXXFLAGS_OPT)) # No OpenGL libraries for no_visual
	$(NVCC) $(SRC_FILES) $(NO_VISUAL_CXXFLAGS_OPT) $(NO_VISUAL_DEFINES) -o $(NO_VISUAL_OPTIMIZED)


clean:
	rm -rf $(OUTPUT_DIR)

