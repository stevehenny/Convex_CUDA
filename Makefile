# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -I./src -I./src/googletest/googletest/include
LDFLAGS = ./src/googletest/build/lib/libgtest.a ./src/googletest/build/lib/libgtest_main.a -pthread

# Directories
SRC_DIR = src
TEST_DIR = src/tests

# Targets and sources
TARGET = $(SRC_DIR)/convex_hull_serial
TEST_TARGET = $(TEST_DIR)/serial_tests
SOURCES = $(SRC_DIR)/convex_hull_serial.cpp
TEST_SOURCES = $(TEST_DIR)/serial_tests.cpp

# All target
all: $(TARGET) $(TEST_TARGET)

# Compile main convex hull program
$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES)

# Compile unit tests
$(TEST_TARGET): $(SOURCES) $(TEST_SOURCES)
	$(CXX) $(CXXFLAGS) $(SOURCES) $(TEST_SOURCES) -o $(TEST_TARGET) $(LDFLAGS)

# Run tests
test: $(TEST_TARGET)
	./$(TEST_TARGET)

# Clean
clean:
	rm -f $(TARGET) $(TEST_TARGET) $(SRC_DIR)/*.o $(TEST_DIR)/*.o

