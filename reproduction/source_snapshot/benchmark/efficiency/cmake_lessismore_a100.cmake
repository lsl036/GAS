# The upstream project hardcodes sm_89.  Override target architecture for A100.
set(CMAKE_CUDA_ARCHITECTURES 80)
