import torch
import torch.nn as nn
from torch_geometric.nn.pool.topk_pool import topk,filter_adj
from torch_geometric.nn import GCNConv

import sys
sys.path.append("models/")
from mlp import MLP, MLP_output
from cnn import CNN
from diagram import sum_diag_from_point_cloud

from tltorch import TRL, TCL

class TenGCN(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, output_dim, final_dropout, tensor_layer_type, node_pooling, PI_dim, sublevel_filtration_methods, device):
        '''
            num_layers: number of GCN layers (INCLUDING the input layer)
            num_mlp_layers: number of layers in mlps (EXCLUDING the input layer)
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of for all hidden units
            output_dim: number of classes for prediction
            final_dropout: dropout ratio on the final linear layer
            tensor_layer: Tensor layer type, TCL/TRL'
            PI_dim: int size of PI
            sublevel_filtration_methods: methods to generate PD
            device: which device to use
        '''

        super(TenGCN, self).__init__()

        self.device = device
        self.num_layers = num_layers
        self.num_neighbors = 5
        self.hidden_dim = hidden_dim
        # point clouds pooling for nodes
        self.score_node_layer = GCNConv(input_dim, self.num_neighbors * 2)
        self.node_pooling = node_pooling

        # GCN block = gcn + mlp
        self.GCNs = torch.nn.ModuleList()
        self.mlps = torch.nn.ModuleList()
        for layer in range(self.num_layers-1):
            if layer == 0:
                self.GCNs.append(GCNConv(input_dim,hidden_dim))
            else:
                self.GCNs.append(GCNConv(hidden_dim**2,hidden_dim))
            self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim**2))
        # tensor layer
        tensor_input_shape = (self.num_layers-1,hidden_dim,hidden_dim)
        tensor_hidden_shape = [hidden_dim,hidden_dim,hidden_dim]
        if tensor_layer_type == 'TCL':
            self.GCN_tensor_layer = TCL(tensor_input_shape,tensor_hidden_shape)
        elif tensor_layer_type == 'TRL':
            self.GCN_tensor_layer = TRL(tensor_input_shape,tensor_hidden_shape)

        # PI tensor block
        # CNN
        self.cnn = CNN(len(sublevel_filtration_methods),hidden_dim,kernel_size=2,stride=2)
        cnn_output_shape = self.cnn.cnn_output_dim(PI_dim)
        # tensor layer
        tensor_input_shape = (hidden_dim,cnn_output_shape,cnn_output_shape)
        tensor_hidden_shape = [hidden_dim,hidden_dim,hidden_dim]
        if tensor_layer_type == 'TCL':
            self.PI_tensor_layer = TCL(tensor_input_shape,tensor_hidden_shape)
        elif tensor_layer_type == 'TRL':
            self.PI_tensor_layer = TRL(tensor_input_shape,tensor_hidden_shape)

        # output block
        self.attend = nn.Linear(2*hidden_dim, 1)
        tensor_input_shape = (2*hidden_dim,hidden_dim,hidden_dim)
        tensor_hidden_shape = [2*hidden_dim,hidden_dim,hidden_dim]
        if tensor_layer_type == 'TCL':
            self.output_tensor_layer = TCL(tensor_input_shape,tensor_hidden_shape)
        elif tensor_layer_type == 'TRL':
            self.output_tensor_layer = TRL(tensor_input_shape,tensor_hidden_shape)
        self.output = MLP_output(hidden_dim,output_dim,final_dropout)

    def compute_batch_feat(self, batch_graph):
        edge_attr = None
        edge_mat_list = []
        start_idx = [0]
        pooled_x = []
        pooled_graph_sizes = []

        for i, graph in enumerate(batch_graph):
            x = graph.x.to(self.device)
            edge_index = graph.edge_index.to(self.device)
            
            if self.node_pooling:
                # graph pooling based on node topological scores
                node_embeddings = self.score_node_layer(x, edge_index).to(self.device)
                node_point_clouds = node_embeddings.view(-1, self.num_neighbors, 2).to(self.device)
                score_lifespan = torch.FloatTensor([sum_diag_from_point_cloud(node_point_clouds[i,...]) for i in range(node_point_clouds.size(0))]).to(self.device)
                batch = torch.LongTensor([0] * x.size(0)).to(self.device)
                perm = topk(score_lifespan, 0.5, batch)
                x = x[perm]
                edge_index, _ = filter_adj(edge_index, edge_attr, perm, num_nodes=graph.x.size(0))
            
            start_idx.append(start_idx[i] + x.size(0))
            edge_mat_list.append(edge_index + start_idx[i])
            pooled_x.append(x)
            pooled_graph_sizes.append(x.size(0))

        pooled_X_concat = torch.cat(pooled_x, 0).to(self.device)
        Adj_block_idx = torch.cat(edge_mat_list, 1).to(self.device)
        return Adj_block_idx, pooled_X_concat, pooled_graph_sizes

    def GCN_layer(self, h, edge_index, layer):
        h = self.GCNs[layer](h, edge_index)
        h = self.mlps[layer](h)
        return h

    def forward(self, batch_graph, batch_PI):

        Adj_block_idx, pooled_X_concat, pooled_graph_sizes = self.compute_batch_feat(batch_graph)
        
        ## PI block
        # CNN
        PI_emb = self.cnn(batch_PI) # before cnn: [batch_size,5,PI_dim,PI_dim]; after cnn: [batch_size,hidden_dim,cnn_output_shape,cnn_output_shape]
        # tensor layer
        PI_hidden = self.PI_tensor_layer(PI_emb).to(self.device) # [batch_size,hidden_dim,hidden_dim,hidden_dim]

        ## GCN block
        hidden_rep = []
        h = pooled_X_concat
        edge_index = Adj_block_idx
        for layer in range(self.num_layers-1):
            h = self.GCN_layer(h, edge_index, layer) # shape: [start_idx[-1]=N,hidden_dim**2]
            hidden_rep.append(h)
        # batch GCN tensor
        hidden_rep = torch.stack(hidden_rep).transpose(0,1) # shape: [start_idx[-1]=N, self.num_layers-1, hidden_dim**2]

        ## graph tensor concat
        graph_sizes = pooled_graph_sizes
        batch_graph_tensor = torch.zeros(len(graph_sizes), 2 * self.hidden_dim, self.hidden_dim, self.hidden_dim).to(self.device)
        node_embeddings = torch.split(hidden_rep, graph_sizes, dim=0)
        for g_i in range(len(graph_sizes)):
            # current graph GCN tensor
            cur_node_embeddings = node_embeddings[g_i] # (n,self.num_layers-1,hidden_dim**2)
            cur_node_embeddings = cur_node_embeddings.view(-1,self.num_layers-1,self.hidden_dim,self.hidden_dim) # (n,self.num_layers-1,hidden_dim,hidden_dim)
            cur_node_embeddings_hidden = self.GCN_tensor_layer(cur_node_embeddings).to(self.device) # (n,hidden_dim,hidden_dim,hidden_dim)
            cur_graph_tensor_hideen = torch.mean(cur_node_embeddings_hidden,dim=0) # (hidden_dim,hidden_dim,hidden_dim)
            # concat with PI tensor
            cur_PI_tensor_hidden = PI_hidden[g_i] # (hidden_dim,hidden_dim,hidden_dim)
            cur_tensor_hidden = torch.cat([cur_graph_tensor_hideen, cur_PI_tensor_hidden], dim=0) # (2*hidden_dim,hidden_dim,hidden_dim)
            batch_graph_tensor[g_i] = cur_tensor_hidden
        
        ## output block
        # attention on the concat dim
        batch_graph_tensor = self.output_tensor_layer(batch_graph_tensor).transpose(1,3) # (batch_size,hidden_dim,hidden_dim,hidden_dim*2)
        batch_graph_attn = self.attend(batch_graph_tensor).squeeze() # (batch_size,hidden_dim,hidden_dim)
        score = self.output(batch_graph_attn) # (batch_size,output_dim)

        return score
