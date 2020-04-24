# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import collections
import pathlib
import pickle
import numpy as np
import models
import torch
from config import MMCIF_PATH


def _construct_residue_atom_indices(node_embed, pos, pos_exist):
    """Dirty nonsense to get a (residue_index, atom_index) pair for each element
    in node_embed."""

    def _hash(vec3):
        return (vec3[0], vec3[1], vec3[2])

    vec_to_indices = {}
    for residue_index in range(pos.shape[0]):
        for atom_index in range(pos.shape[1]):
            if not pos_exist[residue_index, atom_index]:
                continue
            pos_chosen = pos[residue_index, atom_index]
            vec_to_indices[_hash(pos_chosen)] = (residue_index, atom_index)

    residue_and_atom_indices = -np.ones((node_embed.shape[0], 2), dtype=np.int64)
    for idx in range(0, node_embed.shape[0]):
        node_embed_pos = node_embed[idx, -3:]  # 3-dim
        residue_and_atom_indices[idx] = np.array(
            list(vec_to_indices[_hash(node_embed_pos)]), dtype=np.int64
        )

    return residue_and_atom_indices


def attention_vis(args, model, node_embed, pos, pos_exist):
    """Main function to visualize attention maps around each residue."""
    residue_and_atom_indices = _construct_residue_atom_indices(node_embed, pos, pos_exist)

    results = collections.OrderedDict()

    for idx in range(pos.shape[0]):
        for center_atom in list(range(5)):
            # sort the atoms by how far away they are
            # sort key is the first atom on the sidechain
            pos_chosen = pos[idx, center_atom]
            close_idx = np.argsort(np.square(node_embed[:, -3:] - pos_chosen).sum(axis=1))
            close_idx = close_idx[: args.max_size]

            # Grab the 64 closest atoms
            node_embed_short = node_embed[close_idx].copy()
            close_residue_and_atom_indices = residue_and_atom_indices[close_idx].copy()

            # Normalize each coordinate of node_embed to have x, y, z coordinate to be equal 0
            node_embed_short[:, -3:] = node_embed_short[:, -3:] - np.mean(
                node_embed_short[:, -3:], axis=0
            )

            # Torch-ify & enable grads
            node_embed_short = node_embed_short[np.newaxis, :, :]
            node_embed_short = torch.from_numpy(node_embed_short).float().cuda()
            node_embed_short.requires_grad = True

            with torch.autograd.enable_grad():
                energy = model(node_embed_short)
                energy_sum = energy.sum()
                node_embed_short_grad = torch.autograd.grad([energy_sum], [node_embed_short])[0]

            node_embed_short_grad = node_embed_short_grad[:, :, -3:].detach().cpu().numpy()

            assert node_embed_short_grad.shape[0] == 1
            grad_norms = np.linalg.norm(node_embed_short_grad[0], axis=1)

            result = {
                "center_position": np.array([idx, center_atom], dtype=np.int64),
                "viewport_grad_norms": grad_norms,
                "viewport_positions": close_residue_and_atom_indices,
            }
            results[(idx, center_atom)] = result

    return results


def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pdb_name", type=str, required=True)
    parser.add_argument("--results_file", type=pathlib.Path, required=True)

    parser.add_argument("--model_file", type=pathlib.Path, default=None)
    parser.add_argument(
        "--cache_dir",
        type=pathlib.Path,
        default=pathlib.Path(MMCIF_PATH + "/mmCIF/"),
    )

    # Transformer arguments
    parser.add_argument(
        "--encoder_layers",
        type=int,
        default=6,
        help="number of layers to apply the transformer on",
    )
    parser.add_argument("--dropout", type=float, default=0.0, help="chance of dropping out a unit")
    parser.add_argument(
        "--relu_dropout", type=float, default=0.0, help="chance of dropping out a relu unit"
    )
    parser.add_argument(
        "--encoder_normalize_after",
        action="store_false",
        dest="encoder_normalize_before",
        help="whether to normalize outputs before",
    )
    parser.add_argument(
        "--encoder_attention_heads",
        type=int,
        default=8,
        help="number of heads of attention to use",
    )
    parser.add_argument(
        "--attention_dropout", type=float, default=0.0, help="dropout rate of attention"
    )
    parser.add_argument(
        "--encoder_ffn_embed_dim",
        type=int,
        default=1024,
        help="hidden dimension to use in transformer",
    )
    parser.add_argument(
        "--encoder_embed_dim", type=int, default=256, help="original embed dimension of element"
    )
    parser.add_argument("--max_size", type=int, default=64, help="maximum size of time series")

    return parser


def main(args):
    model = models.RotomerTransformerModel(args)
    model = model.cuda()
    model = model.eval()

    if args.model_file is not None:
        checkpoint = torch.load(args.model_file)
        model_state_dict = {
            k.replace("module.", ""): v for k, v in checkpoint["model_state_dict"].items()
        }
        model.load_state_dict(model_state_dict)

    cache_file = args.cache_dir / args.pdb_name[1:3] / f"{args.pdb_name}.cache"
    with open(cache_file, "rb") as infile:
        node_embed, par, child, pos, pos_exist, res, chis_valid, angles = pickle.load(infile)

    viewport_results = attention_vis(args, model, node_embed, pos, pos_exist)
    with open(args.results_file, "wb") as outfile:
        pickle.dump({
            "viewports": viewport_results,
            "residues": res,
        }, outfile)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
