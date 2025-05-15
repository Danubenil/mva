import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F

# Define model ( in your class_model_gnn.py)
class StudentModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_heads = 4):
        super(StudentModel, self).__init__()

        self.gatconv1 = graphnn.GATConv(input_size, hidden_size, heads = n_heads, concat = True)
        self.gatconv2 = graphnn.GATConv(hidden_size * n_heads, hidden_size, heads = n_heads, concat = True)
        
        self.gatconv3 = graphnn.GATConv(hidden_size* n_heads , output_size, heads = 6, concat = False)
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, edge_index):
        x1 = self.elu(self.gatconv1(x, edge_index))
        x2= self.elu(self.gatconv2(x1, edge_index))
        #x2 = torch.concat((x1, x2),dim = -1)
        x3= self.gatconv3(x2, edge_index)
        #x4= self.gatconv4(x3, edge_index)
        return x3

