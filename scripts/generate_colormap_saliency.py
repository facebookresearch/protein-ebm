# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import pathlib
import pickle

import numpy as np

import amino_acid_config


def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--results_file", type=pathlib.Path, required=True)
    parser.add_argument("--color_file", type=pathlib.Path, required=True)
    parser.add_argument("--viewport_center_res", type=int, required=True)
    parser.add_argument("--viewport_center_atom", type=int, required=True)

    return parser


def to_atom_name(res, atom_idx):
    return amino_acid_config.res_atoms[res.upper()][atom_idx]


def intensity_to_color(intensity):
    assert 0 <= intensity <= 1

    return 1.0, (1.0 - intensity) * 0.8, (1.0 - intensity) * 0.8


def main(args):
    with open(args.results_file, "rb") as infile:
        results = pickle.load(infile)

    viewports = results["viewports"]
    residues = results["residues"]

    viewport = viewports[args.viewport_center_res, args.viewport_center_atom]

    norms = viewport["viewport_grad_norms"]
    maxnorm = norms.max()
    residue_and_atom_indices = viewport["viewport_positions"]

    assert norms.shape[0] == residue_and_atom_indices.shape[0]

    center_residue_index, center_atom_index = viewport["center_position"]

    ordered_residues = np.argsort(norms)
    rankings = np.zeros(len(norms))
    rankings[ordered_residues] = np.arange(len(norms)) + 1
    intensities = rankings / len(norms)
    intensities = intensities ** 5.0

    with open(args.color_file, "w") as outfile:
        for i, intensity in enumerate(intensities):
            residue_index, atom_index = residue_and_atom_indices[i]
            atom_name = to_atom_name(residues[residue_index], atom_index)

            r, g, b = intensity_to_color(intensity)

            if residue_index == center_residue_index and atom_index >= 4:
                g, r, b = r, g, b

            print(residue_index, atom_name, r, g, b, sep=",", file=outfile)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
