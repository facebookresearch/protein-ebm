# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import torch
import torch.nn.functional as F
from fairseq.models.transformer import Linear
from fairseq.modules import LayerNorm, MultiheadAttention, SinusoidalPositionalEmbedding
from torch import nn
from torch_geometric_utils import GATConv


class AbstractRotomerModel(nn.Module):
    def __init__(self, scale_factor=1):
        super(AbstractRotomerModel, self).__init__()
        self.element_embed = torch.nn.Embedding(5, 28 * scale_factor)
        self.amino_embed = torch.nn.Embedding(20, 28 * scale_factor)
        self.position_embed = torch.nn.Embedding(21, 28 * scale_factor)
        self.xyz_map = nn.Linear(3, 172 * scale_factor)

    def embed_atoms(self, x):
        """
        x is a collection of atoms. See definition in mmcif_utils.py.

        First dimension is __, second dimension is ___, and third dimension is input features:
            0:   residue index. e.g. Ala or Leu
            1:   atom index. e,g. N, C or O
            2:   count index. index of the atom in the carbon chain
            3-5: normalized x,y,z coordinates of atom
        """
        res_idx, atom_idx, count_idx = x[:, :, 0].long(), x[:, :, 1].long(), x[:, :, 2].long()
        res_encode, atom_encode, amino_ordinal_encode = (
            self.amino_embed(res_idx),
            self.element_embed(atom_idx),
            self.position_embed(count_idx),
        )

        xyz_encode = F.relu(self.xyz_map(x[:, :, 3:]))

        embedding = torch.cat([res_encode, atom_encode, amino_ordinal_encode, xyz_encode], dim=2)
        return embedding


class RotomerFCModel(AbstractRotomerModel):
    """
    Fully connected baseline.
    """

    def __init__(self, args):
        super(RotomerFCModel, self).__init__(scale_factor=1)
        self.args = args

        self.embed = nn.Linear(args.max_size * 256, 1024)
        self.layers = nn.ModuleList([nn.Linear(1024, 1024) for i in range(3)])
        self.energy = nn.Linear(1024, 1)

    def forward(self, x):
        x = self.embed_atoms(x)
        x = x.view(-1, self.args.max_size * 256)
        x = F.relu(self.embed(x))

        for layer in self.layers:
            x = F.relu(layer(x))

        x = self.energy(x)

        return x


class RotomerSet2SetModel(AbstractRotomerModel):
    """
    Set 2 set baseline, following Zaheer et al., 2017.
    """

    def __init__(self, args):
        super(RotomerSet2SetModel, self).__init__(scale_factor=1)

        self.embed_up = nn.Linear(256, 1024)
        self.lstm = nn.LSTM(2048, 1024, batch_first=True)
        self.attention_1 = nn.Linear(2048, 128)
        self.attention_2 = nn.Linear(128, 1)
        self.energy = nn.Linear(1024, 1)

        self.processing_steps = 6
        # Initial state to read for Set2Set Model
        self.default_read = torch.zeros(1, 1024)

        if args.cuda:
            self.default_read = self.default_read.cuda()

    def forward(self, x):
        x = self.embed_atoms(x)
        x = self.embed_up(x)

        state = None
        input_embed = x.mean(dim=1)
        input_embed = torch.cat(
            [input_embed, self.default_read.repeat(input_embed.size(0), 1)], dim=1
        )
        for i in range(self.processing_steps):
            input_embed = input_embed.unsqueeze(1)
            read, state = self.lstm(input_embed, state)
            attention_sub_1 = F.relu(
                self.attention_1(torch.cat([read.repeat(1, args.max_size, 1), x], dim=2))
            )
            output = self.attention_2(attention_sub_1)
            attention_weights = F.softmax(output, dim=1)

            input_embed = (attention_weights * x).sum(dim=1)
            input_embed = torch.cat([input_embed, read[:, 0, :]], dim=1)

        cell = state[1][0]
        x = self.energy(cell)

        return x


class RotomerTransformerModel(AbstractRotomerModel):
    def __init__(self, args):
        """
        Transformer model, following Vaswani et al., 2017.
        """
        scale_factor = args.encoder_embed_dim // 256
        super(RotomerTransformerModel, self).__init__(scale_factor=scale_factor)

        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(args) for i in range(args.encoder_layers)]
        )

        self.fc1 = nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim)
        self.fc2 = nn.Linear(args.encoder_embed_dim, 1)

        self.embed_positions = SinusoidalPositionalEmbedding(
            args.encoder_embed_dim, -1, left_pad=False,
        )

    def forward(self, x, return_hidden=False):
        # If node is empty, pad 0 for the entire node
        encoder_padding_mask = (x.eq(0)).all(dim=2)
        x = self.embed_atoms(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        for layer in self.layers:
            x = layer(x, encoder_padding_mask)

        x_mean, _ = x.max(dim=0)
        hidden = F.relu(self.fc1(x_mean))
        energy = self.fc2(hidden)

        if return_hidden:
            return energy, hidden

        return energy


class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.self_attn = MultiheadAttention(
            self.embed_dim, args.encoder_attention_heads, dropout=args.attention_dropout,
        )
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = Linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = Linear(args.encoder_ffn_embed_dim, self.embed_dim)

        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for i in range(2)])

    def forward(self, x, encoder_padding_mask):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        x, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)
        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x


class RotomerGraphModel(AbstractRotomerModel):
    def __init__(self, args):
        scale_factor = args.encoder_embed_dim // 256 * 2
        super(RotomerGraphModel, self).__init__(scale_factor=scale_factor)
        self.args = args

        self.conv1 = GATConv(512, 512)
        self.conv2 = GATConv(512, 512)
        self.conv3 = GATConv(512, 512)
        self.conv4 = GATConv(512, 512)
        self.conv5 = GATConv(512, 512)
        self.conv6 = GATConv(512, 512)
        self.conv7 = GATConv(512, 512)
        self.conv8 = GATConv(512, 512)
        self.conv9 = GATConv(512, 512)
        self.energy = nn.Linear(512, 1)

        edge_idx = 1 - np.tri(args.max_size)
        grid_x, grid_y = np.arange(args.max_size), np.arange(args.max_size)
        grid_x, grid_y = (
            np.tile(grid_x[:, None], (1, args.max_size)),
            np.tile(grid_y[None, :], (args.max_size, 1)),
        )
        self.pos = np.stack([grid_x, grid_y], axis=2).reshape((-1, 2))
        self.default_mask = edge_idx.astype(np.bool)

    def forward(self, x):
        x = self.embed_atoms(x)

        pos = x[:, :, 3:]
        diff = torch.norm(pos[:, :, None] - pos[:, None, :], p=2, dim=3)
        tresh = diff < 0.3
        mask = tresh.detach().cpu().numpy()
        default_mask = self.default_mask
        mask_new = mask.astype(np.bool) * self.default_mask[None, :, :]
        mask_new = mask_new.reshape((-1, self.args.max_size ** 2))
        pos = self.pos

        edge_idxs = [self.pos[mask_new[i]] + self.args.max_size * i for i in range(x.size(0))]
        edge_idxs = np.concatenate(edge_idxs, axis=0)
        edge = torch.LongTensor(edge_idxs).transpose(0, 1)

        if self.args.cuda:
            edge = edge.cuda()

        # Reshape batch to be a form in which we can apply the graph network on
        batch, atom_num, node_embed = x.size()
        x = x.view(-1, node_embed)

        x = F.relu(self.conv1(x, edge) + x)
        x = F.relu(self.conv2(x, edge) + x)
        x = F.relu(self.conv3(x, edge) + x)
        x = F.relu(self.conv4(x, edge) + x)
        x = F.relu(self.conv5(x, edge) + x)
        x = F.relu(self.conv6(x, edge) + x)
        x = F.relu(self.conv7(x, edge) + x)
        x = F.relu(self.conv8(x, edge) + x)
        x = F.relu(self.conv9(x, edge) + x)

        x = self.energy(x)
        x = x.view(-1, atom_num, 1).mean(dim=1)

        return x
