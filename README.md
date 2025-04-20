
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

Running this on the **3-dimensional [Rastrigin function](https://en.wikipedia.org/wiki/Rastrigin_function)**, which has many local minimas, over 1000 iterations and 500 population size, the cpu-based prism refraction search algorithm converged on the global minima in **just 7 seconds**.

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

To compare the CPU-based optimizer's performance with the CUDA-based optimizer, lets increasing the dimensions to 10, population size to 1000, and iterations to 5000. The following is the performance of the CPU-based optimizer:

```txt
Parameters:
  Dimension: 10
  Max Iterations: 5000
  Population size: 1000
  Alpha: 0.009000
Best solution:
  -0.014345
  -0.001092
  0.012114
  -0.033002
  0.028386
  0.036972
  0.001129
  0.041132
  0.021138
  0.028532
Best score: 1.298488
Elapsed time: 118.681122 seconds
```

Lets compare this result with the CUDA-based PRS Optimizer.

## CUDA-based Prism Refraction Search

### Kernels

- `prs_init_incident_angles_kernel` - one thread per dimension/solution (depending on mem coalesence)
- `prs_eval_fitness_kernel` - one thread per individual in population
- `prs_reduce_delta_kernel` - reduce `delta[i]` to averate `delta_t`
- `prs_update_emergent_kernel` - one thread per (individual, dimension) pair
- `prs_update_incident_kernel` - one thread per (individual, dimension) pair

### Memory Allocation on GPU and CPU

| Data | Location | Notes |
| ---- | -------- | ----- |
| `incident_angles` (pop_size × dim) | GPU | Updated every iteration |
| `emergent_angles` (pop_size × dim) | GPU | Fully device-side |
| `solution` (dim) | GPU scratch per thread (or shared memory) | Used for converting angles to solutions |
| `best_solution` (dim) | Host + GPU | Tracked both sides; updated if a better one found |
| `best_score` | Host + GPU | Same as above |
| `lowerbound`, `upperbound` (dim) | Copied to GPU | Read-only constants for mapping angles to real values |
| `delta_t` | GPU (per iteration) | Global sum or average – use atomic or reduction |
| `prism_angle` | Host + GPU | Updated on host, copied to GPU (or done in kernel if needed) |
| `params` | GPU constant memory | Accessed frequently, rarely changes |

### Pseudocode

```txt
1:  Kernel to initialize population of N solutions

2:  Initialize prism angle A₀

3:  For t = 1 to MaxIter do

4:      Kernel to calculate delta for every solution

5:      Kernel to reduce delta to average delta_t

6:      Calculate refractive index μₘ:
            μₘ = sin((A₀ + δₜ) / 2 ) / sin(A₀ / 2)   // Refractive index Eq.10

7:      Kernel to calculate emergent angle           // Eq. 9
            and update incident angles               // Eq. 11

8:      Update prism angle Aₜ₊₁:
            Aₜ₊₁ = Aₜ × ((alpha - t) / MaxIter)      // Prism angle Eq.12

9:      Kernel to update BestSolution and BestScore

10: End for

11: Return BestSolution, BestScore
```

## Requirements

- Linux
- Dependencies: GNU math library (math.h)
- Nvidia GPU (cuda 6.1 or higher)

## References  

- [Kundu, R., Chattopadhyay, S., Nag, S. et al. **"Prism refraction search: a novel physics-based metaheuristic algorithm."** J Supercomput 80, 10746–10795 (2024).](https://doi.org/10.1007/s11227-023-05790-3)
- [Rastrigin, L. A. **"Systems of extremal control."** Mir, Moscow (1974).](https://en.wikipedia.org/wiki/Rastrigin_function)
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)  
