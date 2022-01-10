import torch
import torch.utils
import torch.utils.data
import torch.nn as nn
import torch
from torch.nn import Sequential, Linear, ReLU, Parameter, Sequential, BatchNorm1d, Dropout
import torch.nn.functional as F
import math
from torch_geometric.nn import MessagePassing
import math

### GraphSN convolution along the graph structure
class GraphSN(MessagePassing):
    def __init__(self, in_dim, emb_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GraphSN, self).__init__(aggr = "add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(in_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), 
                                       torch.nn.ReLU(), torch.nn.Linear(emb_dim, emb_dim))
        
        self.eps = torch.nn.Parameter(torch.FloatTensor(1))

    def reset_parameters(self):
        stdv_eps = 0.1 / math.sqrt(self.eps.size(0))
        nn.init.constant_(self.eps, stdv_eps)
        for layer in self.mlp:
            if hasattr(layer, 'reset_paramters'):
                print(layer)
                layer.reset_parameters()

    def forward(self, x, edge_index, norm_edge_weight, norm_self_loop):
        out = self.mlp(self.eps * norm_self_loop.view(-1, 1) * x + \
            self.propagate(edge_index, x=x, norm=norm_edge_weight))
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * F.relu(x_j)

    def update(self, aggr_out):
        return aggr_out


### GNN to generate node embedding
class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, in_dim, emb_dim, drop_ratio = 0.5, gnn_type = 'graphSN'):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers
        '''

        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        ### add residual connection or not

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'graphSN':
                if layer == 0:
                    self.convs.append(GraphSN(in_dim,emb_dim))
                else:
                    self.convs.append(GraphSN(emb_dim,emb_dim))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.batch_norms:
            bn.reset_parameters()
            
    def forward(self, batched_data):
        x, edge_index, norm_edge_weight, norm_self_loop = batched_data.x, batched_data.edge_index, batched_data.norm_edge_weight, batched_data.norm_self_loop

        ### computing input node embedding

        node_repr = 0
        for layer in range(self.num_layer - 1):

            x = self.convs[layer](x, edge_index, norm_edge_weight, norm_self_loop)
            x = self.batch_norms[layer](x)
            x = F.dropout(F.relu(x), self.drop_ratio, training = self.training)
            node_repr += x

        x = F.dropout(x, self.drop_ratio, training = self.training)
        node_repr += x
        return node_repr