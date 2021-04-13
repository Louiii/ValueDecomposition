# Value-Decomposition Networks For Cooperative Multi-Agent Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Louiii/ValueDecomposition/blob/main/ValueDecomposition.ipynb)

I have implement one of the algorithms discussed in this paper:
https://arxiv.org/pdf/1706.05296.pdf

Specifically, the agent has:
- Value decomposition
- Shared weights (shared critic neural network)
- Role information (one-hot vector indicating which agent it is, concatenated to the observation)
- Centralisation (add each agents Q-values before optimising the weights, during training)
- (No low/high level differentiable communication)

Note: The code supports moving the data and model to a GPU. However, due to the format of the training process it doesn't provide huge speed-ups. This can be improved by training less frequently- but with multiple batches from the episode memory. 
