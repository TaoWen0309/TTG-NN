# TTG-NN
The PyTorch implementation of TTG-NN (**Tensor-view Topological Graph Neural Network**) published @ AISTATS-24.
[\[arXiv\]](https://arxiv.org/pdf/2301.08243.pdf) 

## Method
This work proposes two tensor-based graph representation learning schemes, i.e., Tensor-view Topological Convolutional Layers (TT-CL) and Tensor-view Graph Convolutional Layers (TG-CL). It first produces topological and structural feature tensors of graphs as tensors by using multi-filtrations and graph convolutions respectively. Then, it utilizes TT-CL and TG-CL to learn hidden local and global topological representations of graphs. It further designs a module of Tensor Transformation Layers (TTL) which employs tensor low-rank decomposition to address the model complexity and computation issues.

