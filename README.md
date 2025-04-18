
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

- Implementation of the **sequential PRS algorithm** in C.  
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

**Required**: Nvidia GPU (supporting cuda 6.1 or higher)

Compare naive sequential and CUDA parallel execution:

```bash
make compare
```

## CPU-based Prism Refraction Search

```txt
1:  Initialize population of N solutions

2:  Initialize prism angle A₀

3:  For t = 1 to MaxIter do

4:      For each solution i in 1 to N do
5:          Calculate deviation δₜ[i] = f(iₜ[i])     // Fitness
6:          If δₜ[i] < BestScore:
7:              BestScore = δₜ[i]
8:              BestSolution = iₜ[i]
9:          End if
10:     End for

11:     Calculate refractive index μₘ:
            μₘ = sin((A₀ + δₜ) / 2 ) / sin(A₀ / 2)   // Refractive index Eq.10

12:     For each solution i in 1 to N do
13:         For each dimension j in 1 to D do
14:             Eₜ[i][j] = δₜ[i] - iₜ[i][j] + Aₜ     // Emergent angle Eq.9
15:             r₁ = random number in [-1, 1]
16:             iₜ₊₁[i][j] = asin(...)               // Incidence angle Eq.11
17:             Ensure iₜ₊₁[i][j] is within bounds
18:         End for
19:     End for

20:     Update prism angle Aₜ₊₁:
            Aₜ₊₁ = Aₜ × ((alpha - t) / MaxIter)      // Prism angle Eq.12

21: End for

22: Return BestSolution, BestScore
```

Running this on the *3-dimensional [Rastrigin function](https://en.wikipedia.org/wiki/Rastrigin_function)*, which has many local minimas, over 1000 iterations and 500 population size, the cpu-based prism refraction search algorithm converged on the global minima in *just 7 seconds*.

```txt
Parameters:
  Dimension: 3
  Max Iterations: 1000
  Population size: 500
  Alpha: 0.090000
Best solution:
  -0.014345
  -0.019591
  -0.006583
Best score: 0.125446
Elapsed time: 6.569873 seconds
```

## Requirements

- Linux
- Dependencies: GNU math library (math.h)

## References  

- **Original Paper**: [Prism Refraction Search Algorithm](https://link.springer.com/article/10.1007/s11227-023-05790-3)  
- **CUDA Documentation**: [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)  
