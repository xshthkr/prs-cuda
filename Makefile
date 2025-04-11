CC      = gcc
NVCC    = nvcc

CFLAGS      = -Wall -Wextra -Wpedantic -std=c99 -O2
NVCCFLAGS   = -arch=sm_61 -O2 -Wno-deprecated-gpu-targets -Xcompiler "-Wall -Wextra"
INCLUDE     = -Iinclude

SRC_DIR     = src
BUILD_DIR   = build
BIN_DIR     = bin
TEST_DIR    = tests

CPU_MAIN    = $(SRC_DIR)/prs-cpu.c
CUDA_MAIN   = $(SRC_DIR)/prs-cuda.cu
C_SRCS      = $(filter-out $(CPU_MAIN), $(wildcard $(SRC_DIR)/*.c))

CPU_MAIN_OBJ = $(BUILD_DIR)/$(notdir $(CPU_MAIN:.c=.o))
C_OBJS       = $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o, $(C_SRCS))

CPU_BIN     = $(BIN_DIR)/prs-cpu
CUDA_BIN    = $(BIN_DIR)/prs-cuda

TEST_SRCS   = $(wildcard $(TEST_DIR)/*.c)
TEST_BINS   = $(patsubst $(TEST_DIR)/%.c, $(BIN_DIR)/test/%, $(TEST_SRCS))

all: $(CPU_BIN) $(CUDA_BIN)

$(CPU_BIN): $(CPU_MAIN_OBJ) $(C_OBJS)
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $^ -o $@
	@echo "[#] Built $@ with gcc"

$(CUDA_BIN): $(CUDA_MAIN)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@
	@echo "[#] Built $@ with nvcc"

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@

test: $(TEST_BINS)
	@for t in $(TEST_BINS); do \
		echo "[#] Running $$t..."; \
		./$$t; \
	done

$(BIN_DIR)/test/%: $(TEST_DIR)/%.c $(C_OBJS)
	@mkdir -p $(BIN_DIR)/test
	$(CC) $(CFLAGS) $(INCLUDE) $< $(C_OBJS) -o $@

compare:
	@echo "[#] Comparing CPU and CUDA results..."
	@echo "[#] Running CPU version..."
	./$(CPU_BIN)
	@echo "[#] Running CUDA version..."
	./$(CUDA_BIN)

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)
