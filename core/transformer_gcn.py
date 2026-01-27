import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg

from .axisnet_model import AxisNetFusion


class AxisNetTransformer(AxisNetFusion):
    def __init__(self, input_dim, num_classes, dropout, edgenet_input_dim, edge_dropout, hgc, lg,
                 microbiome_dim=2503, contrastive_weight=0.5, n_heads=4):
        super().__init__(input_dim, num_classes, dropout, edgenet_input_dim, edge_dropout, hgc, lg,
                         microbiome_dim, contrastive_weight)

        self.gconv = nn.ModuleList()
        hidden = [hgc for _ in range(lg)]
        for i in range(lg):
            in_channels = input_dim if i == 0 else hidden[i - 1]
            self.gconv.append(
                tg.nn.TransformerConv(
                    in_channels,
                    hgc // n_heads,
                    heads=n_heads,
                    edge_dim=1,
                    dropout=dropout,
                )
            )
        self.model_init()

    def forward(self, features, edge_index, edgenet_input, microbiome_data=None, enforce_edropout=False):
        edge_index, edgenet_input = self._apply_edge_dropout(edge_index, edgenet_input, enforce_edropout)

        edge_weight = torch.squeeze(self.edge_net(edgenet_input))
        self.edge_index_used = edge_index
        edge_attr = edge_weight.unsqueeze(-1)

        features = F.dropout(features, self.dropout, self.training)
        h = self.relu(self.gconv[0](features, edge_index, edge_attr))
        h0 = h

        for i in range(1, self.lg):
            h = F.dropout(h, self.dropout, self.training)
            h = self.relu(self.gconv[i](h, edge_index, edge_attr))
            jk = torch.cat((h0, h), axis=1)
            h0 = jk

        self.jk_features = jk
        logit = self.cls(jk)

        if microbiome_data is not None:
            microbiome_embed = self.microbiome_encoder(microbiome_data)
            brain_embed = self.contrastive_projector(self.jk_features)
            return logit, edge_weight, microbiome_embed, brain_embed

        return logit, edge_weight


class AxisNetGcnTransformer(AxisNetFusion):
    def __init__(self, input_dim, num_classes, dropout, edgenet_input_dim, edge_dropout, hgc, lg,
                 microbiome_dim=2503, contrastive_weight=0.5, n_heads=4, n_layers=2):
        super().__init__(input_dim, num_classes, dropout, edgenet_input_dim, edge_dropout, hgc, lg,
                         microbiome_dim, contrastive_weight)

        cls_input_dim = sum([hgc for _ in range(lg)])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cls_input_dim,
            nhead=n_heads,
            dim_feedforward=cls_input_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(cls_input_dim)

    def forward(self, features, edge_index, edgenet_input, microbiome_data=None, enforce_edropout=False):
        edge_index, edgenet_input = self._apply_edge_dropout(edge_index, edgenet_input, enforce_edropout)

        edge_weight = torch.squeeze(self.edge_net(edgenet_input))
        self.edge_index_used = edge_index

        features = F.dropout(features, self.dropout, self.training)
        h = self.relu(self.gconv[0](features, edge_index, edge_weight))
        h0 = h

        for i in range(1, self.lg):
            h = F.dropout(h, self.dropout, self.training)
            h = self.relu(self.gconv[i](h, edge_index, edge_weight))
            jk = torch.cat((h0, h), axis=1)
            h0 = jk

        jk = self.transformer_encoder(jk.unsqueeze(0)).squeeze(0)
        jk = self.norm(jk)
        self.jk_features = jk

        logit = self.cls(jk)
        if microbiome_data is not None:
            microbiome_embed = self.microbiome_encoder(microbiome_data)
            brain_embed = self.contrastive_projector(self.jk_features)
            return logit, edge_weight, microbiome_embed, brain_embed

        return logit, edge_weight
