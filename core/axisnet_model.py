import torch
import torch_geometric as tg
import torch.nn.functional as F
from torch import nn

from .edge_encoder import EdgeEncoder


class MicrobiomeEncoder(nn.Module):
    def __init__(self, input_dim=2503, embed_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        return self.encoder(x)


class AxisNetFusion(nn.Module):
    def __init__(self, input_dim, num_classes, dropout, edgenet_input_dim, edge_dropout, hidden_dim, num_layers,
                 microbiome_dim=2503, contrastive_weight=0.5):
        super().__init__()
        k_hops = 3
        hidden = [hidden_dim for _ in range(num_layers)]
        self.dropout = dropout
        self.edge_dropout = edge_dropout
        self.relu = nn.ReLU(inplace=True)
        self.num_layers = num_layers

        self.gconv = nn.ModuleList()
        for i in range(num_layers):
            in_channels = input_dim if i == 0 else hidden[i - 1]
            self.gconv.append(tg.nn.ChebConv(in_channels, hidden[i], k_hops, normalization="sym", bias=False))

        cls_input_dim = sum(hidden)
        self.cls = nn.Sequential(
            nn.Linear(cls_input_dim, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, num_classes),
        )

        self.edge_net = EdgeEncoder(input_dim=edgenet_input_dim // 2, dropout=dropout)
        self.model_init()

        self.microbiome_encoder = MicrobiomeEncoder(microbiome_dim, embed_dim=128)
        self.contrastive_projector = nn.Sequential(
            nn.Linear(sum([hidden_dim for _ in range(num_layers)]), 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )

        self.domain_discriminator = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.contrastive_weight = contrastive_weight

    def model_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def _apply_edge_dropout(self, edge_index, edgenet_input, enforce_edropout):
        if self.edge_dropout > 0:
            if enforce_edropout or self.training:
                one_mask = torch.ones([edgenet_input.shape[0], 1], device=edgenet_input.device)
                drop_mask = F.dropout(one_mask, self.edge_dropout, True)
                bool_mask = torch.squeeze(drop_mask.type(torch.bool))
                edge_index = edge_index[:, bool_mask]
                edgenet_input = edgenet_input[bool_mask]
        return edge_index, edgenet_input

    def forward(self, features, edge_index, edgenet_input, microbiome_data=None, enforce_edropout=False):
        edge_index, edgenet_input = self._apply_edge_dropout(edge_index, edgenet_input, enforce_edropout)

        edge_weight = torch.squeeze(self.edge_net(edgenet_input))
        self.edge_index_used = edge_index

        features = F.dropout(features, self.dropout, self.training)
        h = self.relu(self.gconv[0](features, edge_index, edge_weight))
        h0 = h

        for i in range(1, self.num_layers):
            h = F.dropout(h, self.dropout, self.training)
            h = self.relu(self.gconv[i](h, edge_index, edge_weight))
            jk = torch.cat((h0, h), axis=1)
            h0 = jk

        self.jk_features = jk
        logit = self.cls(jk)

        if microbiome_data is not None:
            microbiome_embed = self.microbiome_encoder(microbiome_data)
            brain_embed = self.contrastive_projector(self.jk_features)
            return logit, edge_weight, microbiome_embed, brain_embed

        return logit, edge_weight

    def contrastive_loss(self, microbiome_embed, brain_embed, labels, temperature=0.5):
        if microbiome_embed is None or brain_embed is None:
            return 0.0

        microbiome_embed = F.normalize(microbiome_embed, dim=1)
        brain_embed = F.normalize(brain_embed, dim=1)

        labels = labels.float()
        labels_matrix = labels.unsqueeze(0)

        batch_size = microbiome_embed.shape[0]
        loss = 0.0
        for i in range(batch_size):
            pos_microbiome = microbiome_embed[i:i + 1]
            pos_brain = brain_embed[i:i + 1]

            neg_microbiome = torch.cat([microbiome_embed[:i], microbiome_embed[i + 1:]], dim=0)
            neg_brain = torch.cat([brain_embed[:i], brain_embed[i + 1:]], dim=0)

            pos_sim = torch.sum(pos_microbiome * pos_brain) / temperature
            neg_sim_micro_brain = torch.sum(pos_microbiome * neg_brain, dim=1) / temperature
            neg_sim_brain_micro = torch.sum(pos_brain * neg_microbiome, dim=1) / temperature
            neg_sims = torch.cat([neg_sim_micro_brain, neg_sim_brain_micro])

            exp_pos = torch.exp(pos_sim)
            exp_neg = torch.exp(neg_sims)
            loss += -torch.log(exp_pos / (exp_pos + exp_neg.sum()))

        return loss / batch_size

    def adversarial_loss(self, microbiome_embed, brain_embed, batch_domains):
        if batch_domains is None:
            return 0.0

        combined_embed = torch.cat([microbiome_embed, brain_embed], dim=0)
        combined_domains = torch.cat([batch_domains, batch_domains], dim=0)
        domain_preds = self.domain_discriminator(combined_embed)
        return F.binary_cross_entropy(domain_preds.squeeze(), combined_domains.float())

    def graph_consistency_loss(self, edge_index, edge_weight, microbiome_features):
        if microbiome_features is None:
            return 0.0

        if edge_index is None:
            edge_index = getattr(self, "edge_index_used", None)
        if edge_index is None:
            return 0.0

        micro_norm = F.normalize(microbiome_features, p=2, dim=1)
        edge_count = min(edge_weight.shape[0], edge_index.shape[1])
        src = edge_index[0][:edge_count]
        dst = edge_index[1][:edge_count]
        target = (micro_norm[src] * micro_norm[dst]).sum(dim=1)
        target = (target + 1.0) * 0.5
        return F.mse_loss(edge_weight[:edge_count], target)
