import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv

class MultiGCN(nn.Module):
    def __init__(self, n_units=[17, 128, 100], dropout=0.0):
        super(MultiGCN, self).__init__()
        self.num_layers = len(n_units) - 1
        self.dropout = dropout
        layer_stack = []

        # in_channels, out_channels, heads
        for i in range(self.num_layers):
            layer_stack.append(GCNConv(in_channels=n_units[i], out_channels=n_units[i+1], cached=False))
        self.layer_stack = nn.ModuleList(layer_stack)
    
    def forward(self, x, edges):
        edges = edges.long()
        for idx, gcn_layer in enumerate(self.layer_stack):
            x = gcn_layer(x=x, edge_index=edges)
            if idx+1 < self.num_layers:
                x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)
        return x

class MultiGAT(nn.Module):
    def __init__(self, n_units=[17, 128, 100], n_heads=[2, 2], dropout=0.0):
        super(MultiGAT, self).__init__()
        self.num_layers = len(n_units) - 1
        self.dropout = dropout
        layer_stack = []

        # in_channels, out_channels, heads
        for i in range(self.num_layers):
            in_channels = n_units[i] * n_heads[i-1] if i else n_units[i]
            layer_stack.append(GATConv(in_channels=in_channels, out_channels=n_units[i+1], heads=n_heads[i]))
        
        self.layer_stack = nn.ModuleList(layer_stack)
    def forward(self, x, edges):
        
        for idx, gat_layer in enumerate(self.layer_stack):
            x = F.dropout(x, self.dropout, training=self.training)
            x = gat_layer(x, edges)
            if idx+1 < self.num_layers:
                x = F.elu(x)
        
        return x

if __name__ == '__main__':
    
    hiddenUnits=[17, 128, 128]
    heads = [2, 2]
    
    # hid = 8
    # in_head = 8
    # out_head = 1
    numFeatures = 3
    
    x = torch.randn((19, numFeatures))
    edges = torch.Tensor([[1, 2], [2, 3], [1, 3]])
    edges = torch.transpose(edges, 0, 1).to(torch.int64)

    # model = MultiGAT(n_units=hiddenUnits, n_heads=heads)
    # out = model(x, edges)
    # print(out.size())

    print(x.shape, edges.shape)
    model = MultiGCN(n_units=[3, 256, 256])
    out = model(x, edges)
    print(out.size())
    # summary(model, [(3, 10), (10, 10)])