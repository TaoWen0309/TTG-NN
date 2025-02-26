# TTG-NN
The PyTorch implementation of TTG-NN (**Tensor-view Topological Graph Neural Network**) published @ AISTATS-24.
[\[arXiv\]](https://arxiv.org/abs/2401.12007) 

## Method
This work proposes two tensor-based graph representation learning schemes, i.e., Tensor-view Topological Convolutional Layers (TT-CL) and Tensor-view Graph Convolutional Layers (TG-CL). It first produces topological and structural feature tensors of graphs as tensors by using multi-filtrations and graph convolutions respectively. Then, it utilizes TT-CL and TG-CL to learn hidden local and global topological representations of graphs. It further designs a module of Tensor Transformation Layers (TTL) which employs tensor low-rank decomposition to address the model complexity and computation issues.

![TTG-NN](TTG-NN.png)

## Requirements
Python 3.10, torch 2.0.0, gudhi 3.7.1, tensorly-torch 0.4.0, networkx 3.0, numpy 1.24.2, scipy 1.10.1, scikit-learn 1.2.2.

Warning: Don't set hidden_dim too large for TRL (4 is recommended).

## Citation
If you find this repository useful in your research, please consider giving a star :star: and a citation
```
@InProceedings{pmlr-v238-wen24a,
  title = 	 {Tensor-view Topological Graph Neural Network},
  author =       {Wen, Tao and Chen, Elynn and Chen, Yuzhou},
  booktitle = 	 {Proceedings of The 27th International Conference on Artificial Intelligence and Statistics},
  pages = 	 {4330--4338},
  year = 	 {2024},
  editor = 	 {Dasgupta, Sanjoy and Mandt, Stephan and Li, Yingzhen},
  volume = 	 {238},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {02--04 May},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v238/wen24a/wen24a.pdf},
  url = 	 {https://proceedings.mlr.press/v238/wen24a.html},
  abstract = 	 {Graph classification is an important learning task for graph-structured data. Graph neural networks (GNNs) have recently gained growing attention in graph learning and shown significant improvements on many important graph problems. Despite their state-of-the-art performances, existing GNNs only use local information from a very limited neighborhood around each node, suffering from loss of multi-modal information and overheads of excessive computation. To address these issues, we propose a novel Tensor-view Topological Graph Neural Network (TTG-NN), a class of simple yet effective topological deep learning built upon persistent homology, graph convolution, and tensor operations. This new method incorporates tensor learning to simultaneously capture {\it Tensor-view Topological} (TT), as well as Tensor-view Graph (TG) structural information on both local and global levels. Computationally, to fully exploit graph topology and structure, we propose two flexible TT and TG representation learning modules which disentangles feature tensor aggregation and transformation, and learns to preserve multi-modal structure with less computation. Theoretically, we derive high probability bounds on both the out-of-sample and in-sample mean squared approximation errors for our proposed Tensor Transformation Layer (TTL). Real data experiments show that the proposed TTG-NN outperforms 20 state-of-the-art methods on various graph benchmarks.}
}
```
