# Compiler and Flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -I./src -I./src/googletest/googletest/include
LDFLAGS = -L./src/googletest/build/lib -lgtest -lgtest_main -pthread -lgmp -lglut -lGL -lGLU

# Directories
SRC_DIR = src
TEST_DIR = src/tests
BUILD_DIR = build

# Files and Targets
MAIN_TARGET = build/main
MAIN_SOURCES = $(SRC_DIR)/main.cpp $(SRC_DIR)/convex_hull_serial.cpp
TARGET = $(BUILD_DIR)/convex_hull_serial
TEST_TARGET = $(BUILD_DIR)/serial_tests
SOURCES = $(SRC_DIR)/convex_hull_serial.cpp
HEADERS = $(SRC_DIR)/convex_hull_serial.h
TEST_SOURCES = $(TEST_DIR)/serial_tests.cpp
OBJECTS = $(SOURCES:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)
MAIN_OBJECTS = $(MAIN_SOURCES:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)
TEST_OBJECTS = $(TEST_SOURCES:$(TEST_DIR)/%.cpp=$(BUILD_DIR)/%.o)

# Create build directory if it doesn't exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Build main convex hull program
$(TARGET): $(BUILD_DIR) $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $@

# Build the main program
$(MAIN_TARGET): $(BUILD_DIR) $(MAIN_OBJECTS)
	$(CXX) $(MAIN_OBJECTS) $(LDFLAGS) -o $@

# Build unit tests
$(TEST_TARGET): $(BUILD_DIR) $(OBJECTS) $(TEST_OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) $(TEST_OBJECTS) -o $@ $(LDFLAGS)

# Object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(TEST_DIR)/%.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Run tests
test: $(TEST_TARGET)
	./$(TEST_TARGET)

#build main
main: $(MAIN_TARGET)
	./$(MAIN_TARGET)

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)

# Phony targets
.PHONY: all test clean
