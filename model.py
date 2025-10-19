import torch
from torch.nn import (
    LayerNorm,
    Linear,
    Module,
    ModuleList,
    ReLU,
    GELU,
    Dropout,
    Sequential,
    Identity,
)
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau
from torch_geometric.nn import GINEConv, GPSConv
from loss import CombinedLoss
from sam import SAM
from timm.layers import DropPath

INPUT_DIM = 34 + 7 + 1 + 26 # one-hot
EDGE_DIM = 2
ESM_DIM = 1280


class GPS(Module):
    def __init__(
        self, 
        channels: int,
        heads: int,
        dropout: float,
        attn_dropout: float,
        act: str, 
        pe_dim: int,
        pe_ratio: float,
        esm_out: int,
        num_layers: int,
        lr: float,
        weight_decay: float,
        weight: float,
        alpha: float,
        beta: float,
        use_esm: bool,
        esm_dim: int,
        ):
        super().__init__()
        pe_out = max(int(pe_ratio * channels), 2)
        self.node_emb = Linear(INPUT_DIM, channels - pe_out)
        self.pe_lin = Linear(pe_dim, pe_out)
        self.pe_norm = LayerNorm(pe_dim)
        self.lr = lr
        self.weight_decay = weight_decay
        
        # esm
        self.use_esm = use_esm
        if self.use_esm:
            self.esm_emb = Sequential(
                LayerNorm(esm_dim),
                Linear(esm_dim, esm_out),
                GELU(),
                Dropout(dropout)
            )
            self.channels = channels + esm_out
        else:
            self.channels = channels
        assert self.channels % heads == 0
        
        self.edge_emb = Linear(EDGE_DIM, self.channels)
        self.conv_gps = ModuleList()
        for _ in range(num_layers):
            nn = Sequential(
                Linear(self.channels, self.channels),
                ReLU(),
                Dropout(dropout),
                Linear(self.channels, self.channels),
            )
            conv1 = GPSConv(
                self.channels, GINEConv(nn, edge_dim=self.channels), 
                heads=heads, dropout=dropout, act=act,
                attn_kwargs={'dropout': attn_dropout}
            )
            gatedgps = GatedGNNBlock(self.channels, conv1)
            
            self.conv_gps.append(gatedgps)

        self.mlp = Sequential(
            Linear(self.channels, self.channels // 2),
            ReLU(),
            Linear(self.channels // 2, self.channels // 4),
            ReLU(),
            Linear(self.channels // 4, 2),
        )
        
        self.criterion = CombinedLoss(weight, alpha, beta)

    def forward(self, x, edge_index, edge_attr, pe, batch, esm=None):
        x = x.float()
        x_pe = self.pe_norm(pe)
        if esm is not None and self.use_esm:
            x = torch.cat((self.node_emb(x), self.pe_lin(x_pe), self.esm_emb(esm)), 1) # h0
        else:
            x = torch.cat((self.node_emb(x), self.pe_lin(x_pe)), 1)
        edge_attr = self.edge_emb(edge_attr)

        for conv in self.conv_gps:
            x = conv(x, edge_index, batch, edge_attr=edge_attr)
        
        return self.mlp(x)

    def get_optimizer_scheduler(self):
        optimizer = torch.optim.Adam
        self.optimizer = SAM(self.parameters(), optimizer, lr=self.lr, weight_decay=self.weight_decay)
        
        self.warmup_scheduler = LinearLR(self.optimizer, start_factor=0.1, total_iters=50)
        self.main_scheduler = ReduceLROnPlateau(self.optimizer.base_optimizer, mode='max', factor=0.6, patience=10, min_lr=1e-6)

        self.warmup_epochs = 10
        self.current_epoch = 0

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Total trainable parameters: {total_params}')

class GatedGNNBlock(Module):
    def __init__(self, hidden_channels, conv, drop_path_prob=0.):
        super(GatedGNNBlock, self).__init__()
        self.norm_node = LayerNorm(hidden_channels, eps=1e-6)
        self.norm_edge = LayerNorm(hidden_channels, eps=1e-6)
        self.fc_node_1 = Linear(hidden_channels, hidden_channels * 4)
        self.fc_node_2 = Linear(hidden_channels * 2, hidden_channels)
        
        self.fc_edge_1 = Linear(hidden_channels, hidden_channels * 4) 
        
        self.split_indices = [2 * hidden_channels, hidden_channels, hidden_channels]
        
        self.act = GELU()
        self.drop_path = DropPath(drop_path_prob) if drop_path_prob > 0. else Identity()
        
        self.conv = conv

    def forward(self, x, edge_index, batch, edge_attr):
        shortcut = x
        x = self.norm_node(x)
        edge_attr = self.norm_edge(edge_attr)

        g_node, i_node, x = torch.split(self.fc_node_1(x), self.split_indices, dim=-1)
        _, _, edge_attr = torch.split(self.fc_edge_1(edge_attr), self.split_indices, dim=-1)

        x = self.conv(x, edge_index, batch, edge_attr=edge_attr)

        filter = self.act(g_node) * torch.cat((i_node, x), dim=-1)
        x = self.fc_node_2(filter)
        
        x = self.drop_path(x)
        return x + shortcut
