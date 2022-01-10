import torch.nn as nn
import torch.nn.functional as F
from layers import GraphSN
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
import torch.nn.functional as F

from layers import GNN_node

class GNN(torch.nn.Module):

    def __init__(self, num_tasks, in_dim, num_layer = 5, emb_dim = 64, 
                    gnn_type = 'graphSN', drop_ratio = 0.5, graph_pooling = "mean"): 
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(GNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
    
        self.gnn_node = GNN_node(num_layer, in_dim, emb_dim, drop_ratio = drop_ratio, gnn_type = gnn_type)


        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise ValueError("Invalid graph pooling type.")

        self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)

        self.reset_parameters()

    def reset_parameters(self):
        self.gnn_node.reset_parameters()
        self.graph_pred_linear.reset_parameters()

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)

        h_graph = self.pool(h_node, batched_data.batch)

        return self.graph_pred_linear(h_graph)


if __name__ == '__main__':
    GNN(num_tasks = 10)