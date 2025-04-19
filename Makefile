CC = gcc
CFLAGS = -Wall -Wextra -Wpedantic -std=c99 -O2

NVCC = nvcc
NVCCFLAGS = -Wno-deprecated-gpu-targets -arch=sm_61 -O2

INCLUDE = -Iinclude
LIB = -lm

CPU_SRC = src/prs-cpu.c src/prs.c src/utils.c
CPU_OBJ = $(patsubst src/%.c, build/%.o, $(CPU_SRC))

CUDA_SRC = src/prs-cuda.cu
CUDA_OBJ = build/prs-cuda.o

TARGET_CPU = bin/prs-cpu
TARGET_CUDA = bin/prs-cuda

TEST_DIR = tests
TEST_SRC = $(wildcard $(TEST_DIR)/*.c)
TEST_BIN = $(patsubst $(TEST_DIR)/%.c, bin/test/%, $(TEST_SRC))
SRC_NO_MAIN = $(filter-out src/prs-cpu.c, $(SRC))
SRC_NO_MAIN_OBJ = $(patsubst src/%.c, build/%.o, $(SRC_NO_MAIN))

all: $(TARGET_CPU) $(TARGET_CUDA)

$(TARGET_CPU): $(CPU_OBJ)
	@mkdir -p bin
	$(CC) $(CFLAGS) $(CPU_OBJ) -o $@ $(LIB)
	@echo "[#] CPU Build complete."

build/%.o: src/%.c
	@mkdir -p build
	$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@

$(CUDA_OBJ): $(CUDA_SRC)
	@mkdir -p build
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(TARGET_CUDA): $(CUDA_OBJ)
	@mkdir -p bin
	$(NVCC) $(NVCCFLAGS) $< -o $@
	@echo "[#] CUDA Build complete."


test: $(TEST_BIN)
	@for t in $(TEST_BIN); do \
		echo "[#] Running $$t..."; \
		./$$t; \
	done

bin/test/%: tests/%.c $(SRC_NO_MAIN_OBJ)
	@mkdir -p bin/test
	$(CC) $(CFLAGS) $(INCLUDE) $< $(SRC_NO_MAIN_OBJ) -o $@ $(LIB)

clean:
	rm -rf build bin