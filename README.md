# useless-cuda
Experiment with CUDA and to test my GPU speed

* collective-test: Output variable numbers of items from each input item.
* cuda-related: Try to know something about GPU hardware, but not actually run CUDA kernel.
* experiment: Well, experiments...
* fft: Fast Fourier transform
* gameoflife: Conway's Game of Life simulation
* haltkernel: Stop GPU kernels from host side to get the intermediate result. Warning! These programs may hang your computer.
* hamilton: Counting Hamiltonian cycles of a directed graph.
* lightout: Counting the solutions of n*m Lights Out game. Actually this game can be solved by Gausssian elimination, but I choose to brute force it by GPU.😃
* memspeed: Test memory speed of various cases.
* nqueen: Counting solutions to N-Queens puzzle. I have many kinds of implementations: AVX2, ARM64 NEON, CUDA, and Metal.
* sparse-subgraph: Compute the number of edges of sparsest k-subgraph.
* persist.cu: I wrote this because one of my lab computer is slow at initializing CUDA. (It took about 0.5 seconds) I found out that CUDA initialization will be faster if some process is already using GPU, so all the program does is initialize CUDA context and then sleep indefinitely.
