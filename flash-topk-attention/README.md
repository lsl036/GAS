# Flash TopK Attention


## Installation

Create environment

```
conda create -n ftka python=3.10
conda activate ftka
pip install flashinfer-python==0.2.0.post1
pip install torch==2.5.0
pip install numpy==2.1.2
```

Install package


```
cd flash-topk-attention
pip install -e .
```

All CUDA kernels are automatically JIT compiled (Default cached in `~/.cache/flashinfer`) for convient development and deployment.