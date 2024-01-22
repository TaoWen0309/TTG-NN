import torch
import torch.nn.functional as F
import torch.nn as nn

# for PI block
class CNN(nn.Module):
    def __init__(self, dim_in, hidden_dim):
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(dim_in, hidden_dim, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    def forward(self, PI):
        feature = self.cnn(PI)
        return feature

# compute output dim given the above kernel_size and stride
def cnn_output_dim(dim_in):
    tmp_dim = int((dim_in-2)/2)+1
    output_dim = int((tmp_dim-2)/2)+1
    return output_dim