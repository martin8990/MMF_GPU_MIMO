# MMF_GPU_MIMO
GPU Accelerated MIMO for lab experiments concerning Multimode Fiber systems. For more information please see the included paper.

This Repository contains a GPU Python Implementation of the Multiple input, Multiple output adaptive equalizer required for Multimode fiber systems.

# Requirements

Python 3.x (3.6.3)
Pyculib (1.0.2)
Numba (0.35.0)
Numpy (1.13.3)
Matplotlib (2.1.0)
Cuda Toolkit (9)

Inside the repository several Python files can be Found.
# Main.py
Sample usage of the MIMO.
# Transmitter.py
Randomly generates a signal.
# Disturbance.py
Distorts signals using a delay or crosstalk.
# MIMO_CPU.py
CPU implementation of the MIMO-equalizer.
# MIMO.py
General GPU implementation of the MIMO-equalizer.
# ALL_IN_MIMO.py
Optimzed GPU implementation of the MIMO-equalizer.
Optimization by:
Merging several GPU Kernels.
Doing the First "input" FFT for every required cycle at the same time in the beginning.

# MIMO_Kernels.py
All the GPU code for the MIMO

The rest of the files contain performance tests of the MIMO.


