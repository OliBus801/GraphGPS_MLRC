import torch
import torch.nn as nn

from torch_geometric.graphgym.register import register_layer
from torch_geometric.utils import softmax
from torch_scatter import scatter

from yacs.config import CfgNode as CN

# Note: A registered GNN layer should take 'batch' as input
# and 'batch' as output


# Registering the GRIT attention layer
class MultiHeadAttentionLayerGrit(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, use_bias=True,
                 clamp=5., dropout=0., act=None,
                 edge_enhance=True,
                 sqrt_relu=False,
                 signed_sqrt=True,
                 cfg=CN(),
                 **kwargs):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.clamp = clamp
        self.dropout = nn.Dropout(dropout)

        # Learnable projections for multi-head attention

        # Node-pair (edge) level projections for attention score computation
        self.Q = nn.Linear(in_channels, out_channels * num_heads, bias=use_bias)  # Query projection
        self.K = nn.Linear(in_channels, out_channels * num_heads, bias=use_bias)  # Key projection
        self.E_w = nn.Linear(in_channels, out_channels * num_heads, bias=use_bias)  # Edge feature modulation (multiplicative part)
        self.E_b = nn.Linear(in_channels, out_channels * num_heads, bias=use_bias)  # Edge feature bias (additive part)

        # Node-level projection for value aggregation
        self.V = nn.Linear(in_channels, out_channels * num_heads, bias=use_bias)  # Value projection
        self.E_v = nn.Parameter(torch.Tensor(out_channels, num_heads, out_channels)) # edge enhancement parameter

        # Learnable weight to convert attention vectors to scalars per head
        self.A = nn.Parameter(torch.Tensor(out_channels, num_heads, 1))

        # Activation function
        self.activation = nn.ReLU()

        # Initialize all learnable parameters with Xavier normal
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.Q.weight)
        nn.init.xavier_normal_(self.K.weight)
        nn.init.xavier_normal_(self.E_w.weight)
        nn.init.xavier_normal_(self.E_b.weight)
        nn.init.xavier_normal_(self.V.weight)
        nn.init.xavier_normal_(self.E_v)
        nn.init.xavier_normal_(self.A)


    def forward(self, batch):
        x = batch.x
        edge_attr = batch.edge_attr if hasattr(batch, 'edge_attr') else None

        # Compute projections
        Q_x = self.Q(x)  # Query projection
        K_x = self.K(x) # Key projection
        V_x = self.V(x) # Value projection

        if edge_attr is not None:
            E_w = self.E_w(edge_attr) # Edge feature modulation
            E_b = self.E_b(edge_attr) # Edge feature bias

        # Reshape projections for multi-head attention
        batch.Q_x = Q_x.view(-1, self.num_heads, self.out_channels)
        batch.K_x = K_x.view(-1, self.num_heads, self.out_channels)
        batch.V_x = V_x.view(-1, self.num_heads, self.out_channels)
        if edge_attr is not None:
            batch.E_w = E_w.view(-1, self.num_heads, self.out_channels)
            batch.E_b = E_b.view(-1, self.num_heads, self.out_channels)

        # Propagate attention scores and values
        self.propagate(batch)

        # Extract node and edge representations
        node_representations = batch.node_representations
        edge_representations = batch.edge_representations


        return node_representations, edge_representations
    
    def propagate(self, batch):
        W_xi = batch.Q_x[batch.edge_index[0]] # Query projection from node x_i (source node)
        W_xj = batch.K_x[batch.edge_index[1]] # Key projection from node x_j (target node)

        attn_score = W_xi + W_xj # Element-wise addition of query and key projections

        # Check if edge feature exists
        if hasattr(batch, 'E_w') and hasattr(batch, 'E_b'):   
            attn_score = attn_score * batch.E_w # Element-wise multiplication with edge feature modulation
            attn_score = torch.sqrt(torch.relu(attn_score)) - torch.sqrt(torch.relu(-attn_score)) # Signed Squared Root Stabilization
            attn_score = attn_score + batch.E_b # Element-wise addition with edge feature bias 
        
        attn_score = self.activation(attn_score)
        edge_representations = attn_score

        # Output Edge Representations
        if hasattr(batch, 'E_w') and hasattr(batch, 'E_b'):
            batch.edge_representations = edge_representations.flatten(1)

        # Compute Final Attention Score
        attn_scalars = torch.einsum('ehd,dhc->ehc', attn_score, self.A) # Project attention scores to scalars
        attn_score = softmax(attn_scalars, batch.edge_index[1]) # Normalize attention scores with softmax
        attn_score = self.dropout(attn_score)
        batch.attn = attn_score

        # Compute Node Reprensentations
        batch.node_representations = torch.zeros_like(batch.V_x)
        W_xj = attn_score * batch.V_x[batch.edge_index[0]] # alpha_ij * W_V(x_j)
        scatter(W_xj, batch.edge_index[1], dim=0, out=batch.node_representations, reduce='add') # Aggregate over neighbors

        if hasattr(batch, 'E_w') and hasattr(batch, 'E_b'):
            alpha_eij = scatter(edge_representations * attn_score, batch.edge_index[1], dim=0, reduce="add")
            W_eij = torch.einsum('nhd, dhc -> nhc', alpha_eij, self.E_v)

            batch.node_representations = batch.node_representations + W_eij

        batch.edge_representations = edge_representations.flatten(1)

@register_layer("GritTransformer")
class GritTransformerLayer(nn.Module):
    """
        Transformer layer for GRIT taken from the paper definition.
    """

    def __init__(self, in_channels, out_channels, num_heads,
                 dropout=0.0,
                 attn_dropout=0.0,
                 layer_norm=False, batch_norm=True,
                 residual=True,
                 act='relu',
                 norm_e=True,
                 O_e=True,
                 cfg=dict(),
                 **kwargs):
        super().__init__()

        # Multi-head attention layer
        self.attention = MultiHeadAttentionLayerGrit(
            in_channels=in_channels,
            out_channels=out_channels // num_heads,
            num_heads=num_heads,
            dropout=attn_dropout,
            clamp=cfg.attn.get("clamp", 5.),
            act=cfg.attn.get("act", "relu")
        )
        
        # Degree Scaler learnable parameter
        self.theta1 = nn.Parameter(torch.Tensor(1, out_channels))
        self.theta2 = nn.Parameter(torch.Tensor(1, out_channels))
        nn.init.xavier_normal_(self.theta1)
        nn.init.xavier_normal_(self.theta2)

        # Node and edge feature normalization
        self.norm_node_1 = nn.BatchNorm1d(in_channels)
        self.norm_node_2 = nn.BatchNorm1d(in_channels)
        self.norm_edge = nn.BatchNorm1d(in_channels)


        self.ffn = nn.Sequential(
            nn.Linear(in_channels, out_channels * 2),
            nn.ReLU(),
            nn.Linear(in_channels * 2, out_channels),
        )

    def forward(self, batch):
        x_in = batch.x
        e_in = batch.edge_attr if hasattr(batch, 'edge_attr') else None


        # *** Node Level ***
        # Multi-Scale Attention with Residual Connection
        x_attn, e_attn = self.attention(batch)
        x = x_in + x_attn.view(batch.num_nodes, -1) # Residual connection

        # Degree Scaling from definition (5) in the paper
        if hasattr(batch, 'log_deg'):
            x = (x * self.theta1) + ((batch.log_deg.unsqueeze(-1) * x) * self.theta2)

        # First BatchNorm
        x = self.norm_node_1(x)

        # Feed Forward Network with Residual Connection
        x_ffn = self.ffn(x)
        x = x + x_ffn

        # Second BatchNorm
        x_out = self.norm_node_2(x)

        # *** Edge Level ***
        e = e_in + e_attn
        e_out = self.norm_edge(e)

        # Update batch with new node and edge features
        batch.x = x_out
        batch.edge_attr = e_out

        return batch

