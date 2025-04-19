#include <cuda.h>

#include <stdio.h>

void getCudaStats() {
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        printf("Number of CUDA devices: %d\n", deviceCount);
        
        for (int i = 0; i < deviceCount; i++) {
                cudaDeviceProp prop;
                cudaGetDeviceProperties(&prop, i);
                printf("Device %d: %s\n", i, prop.name);
                printf("  Total global memory: %zu bytes\n", prop.totalGlobalMem);
                printf("  Multiprocessor count: %d\n", prop.multiProcessorCount);
                printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
        }
        
        return;
}

int main() {
        getCudaStats();
        return 0;
}
