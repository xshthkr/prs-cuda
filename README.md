
# Prism Refraction Search (PRS) - CUDA Implementation

A Novel Physics-Based Metaheuristic Algorithm

[Springer, The Journal of Supercomputing, Published: 04 January 2024](https://link.springer.com/article/10.1007/s11227-023-05790-3)

## Project Overview  

This project implements the **Prism Refraction Search (PRS) Algorithm** using **CUDA C** to leverage parallel computation for improved performance. The PRS algorithm, inspired by the principles of light refraction through a prism, is a novel metaheuristic optimization method designed for solving global optimization problems.

This project was submitted as part of the final project for **CS326 - High Performance Computing** at **NIT Surat** in 2025.

## Objectives  

- Implement the **sequential PRS algorithm** based on the original research paper.  
- Develop a **parallel CUDA implementation** to enhance computational efficiency.  
- Compare the **performance of sequential and parallel implementations** on benchmark optimization problems.  

## Features  

- Implementation of the **sequential PRS algorithm** in C++.  
- Development of a **parallelized CUDA version** to accelerate computation.  
- Performance benchmarking on standard optimization problems.  
- Comparative analysis of execution time and scalability.  

## Installation and Usage

Clone the project repository and compile source files:

```bash
git clone https://github.com/xshthkr/prs-cuda.git
cd prs-cuda
make
```

Compare naive sequential and CUDA parallel execution:

```bash
make compare
```

## References  

- **Original Paper**: [Prism Refraction Search Algorithm](https://link.springer.com/article/10.1007/s11227-023-05790-3)  
- **CUDA Documentation**: [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)  
