import torch
import torch.nn as nn
import torch.nn.functional as F

# MLP in GCN block
class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''
    
        super(MLP, self).__init__()
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear_or_not = True
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()
            # Input layer
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            # Hidden layer
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            # Output layer
            self.linears.append(nn.Linear(hidden_dim, output_dim))
            # Batch Norm on Input and Hidden layer
            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear
            return self.linear(x)
        else:
            # If MLP
            h = x
            # Input and Hidden layer
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            # Output layer(no activation)
            return self.linears[self.num_layers - 1](h)

# MLP classifier
class MLP_output(nn.Module):
    def __init__(self, hidden_dim, output_dim, dropout):
        '''
            hidden_dim: dimensionality of hidden features
            output_dim: number of classes for prediction
            device: which device to use
        '''
    
        super(MLP_output, self).__init__()

        self.linear = nn.Linear(hidden_dim,1)
        self.output = nn.Sequential(nn.BatchNorm1d(hidden_dim),
                                    nn.ReLU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(hidden_dim, output_dim))

    def forward(self, h):
        h = self.linear(h).squeeze()
        h = self.output(h)
        return h