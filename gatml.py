import torch
import torch.nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor




#GAT Network Setup
class GraphAttentionLayer(nn.Module):
    #2 Head Graph Attention Layer for debris attention
    def __init__(self, in_features: int = 32, out_features: int = 32, heads: int = 2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads

        #Weight Projections per head: [head, in_features, out_features -> (2,32,32)]
        self.W = nn.Parameter(torch.empty(size=(heads, in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414) #According to the original GAT paper

        #Bias per head -> (2,32)
        self.bias = nn.Parameter(torch.empty(heads, out_features))
        nn.init.zeros_(self.bias)


        #Attention Parameter Vector
        self.a = nn.Parameter(torch.empty(size=(heads, 2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(0.2) #Industry standard for LeakuReLU for GATs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [Batch, Num_Debris (10), In_Features (32)][cite: 6, 7]
        B, N, _ = x.shape
        x_expanded = x.unsqueeze(1).repeat(1, self.heads, 1, 1)

        h = torch.matmul(x_expanded, self.W) + self.bias.unsqueeze(0).unsqueeze(2)
        h_i = h.unsqueeze(3).repeat(1,1,1,N,1)
        h_j = h.unsqueeze(2).repeat(1,1,N,1,1)


        #Expand inputs per headx
        a_input = torch.cat([h_i, h_j], dim=-1)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))

        attention = F.sotmax(e, dim=-1)
        h_prime = torch.matmul(attention, h)

        h_prime = torch.mean(h_prime, dim=1)
        return F.elu(h_prime)


class SatelliteGATNetwork(nn.Module):
    def __init__(
        self,
        num_debris: int = 10,
        debris_dim: int = 6,
        host_dim: int = 4,
        embed_dim: int = 32,
        ffn_hidden_dim: int = 64,
        policy_hidden_dim: int = 72,
        num_discrete_actions: int = 4,
        num_continuous_outputs: int = 2
    ):
        super(SatelliteGATNetwork, self).__init__()

        #Debris Input Projection (6 -> 32)
        self.debris_embed = nn.Linear(debris_dim, embed_dim)

        #Graph Attention Layer (2 heads, 32 inputs/outputs)
        self.gat_layer = GraphAttentionLayer(in_features=embed_dim, out_features=embed_dim, heads=2)

        #Feed-Forward Section (32->64->32)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden_dim),
            nn.ReLU(),
            nn.Linear(ffn_hidden_dim, embed_dim)
        )

        #Graph Normalization
        self.layer_norm = nn.LayerNorm(embed_dim)

        #Shared Policy Hidden Layer
        self.policy_hidden = nn.Sequential(
            nn.Linear(embed_dim + host_dim, policy_hidden_dim),
            nn.ReLU()
        )

        #Hybrid Output Branches
        self.discreet_head = nn.Linear(policy_hidden_dim, num_discrete_actions) #(72->4)
        self.continous_head = nn.Linear(policy_hidden_dim, num_continuous_outputs) #(72->2)

    def forward(self, debris_tensor: torch.Tensor, host_tensor: torch.Tensor):
        #Feature Projection
        x = F.relu(self.debris_embed(debris_tensor))

        #Graph Attention
        x_gat = self.gat_layer(x)

        #FFN + LayerNorm
        x_ffn = self.ffn(x_gat)
        x_norm = self.layer_norm(x_gat + x_ffn)

        debris_pooled = torch.mean(x_norm, dim=1)

        #Combined State
        combined_state = torch.cat([debris_pooled, host_tensor], dim=1)
        hidden_features = self.policy_hidden(combined_state)

        #Compute hybrid outputs
        discreet_output = self.discreet_head(hidden_features)
        continious_output = self.continous_head(hidden_features)

        return discreet_output, continious_output

    