# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Quantiative and qualtitative metrics of performance on rotamer recovery.
Additional scripts for visualization are in scripts/

Several of the visualizations are helper functions that can easily be
imported into a jupyter notebook for further analysis.
"""

import argparse
import itertools
import os
import os.path as osp
import pickle
import random
import numpy as np

import matplotlib.pyplot as plt
from scipy.stats import special_ortho_group

import torch
from amino_acid_config import kvs
from config import MMCIF_PATH
from constants import atom_names, residue_names
from data import MMCIFTransformer
from easydict import EasyDict
from mmcif_utils import (
    compute_dihedral,
    compute_rotamer_score_planar,
    exhaustive_sample,
    interpolated_sample_normal,
    load_rotamor_library,
    mixture_sample_normal,
    parse_dense_format,
    reencode_dense_format,
    rotate_dihedral_fast,
)
from models import RotomerFCModel, RotomerGraphModel, RotomerSet2SetModel, RotomerTransformerModel
from torch import nn
from tqdm import tqdm


def add_args(parser):
    parser.add_argument(
        "--logdir",
        default="cachedir",
        type=str,
        help="location where log of experiments will be stored",
    )
    parser.add_argument(
        "--no-cuda", default=False, action="store_true", help="do not use GPUs for computations"
    )
    parser.add_argument(
        "--exp",
        default="transformer_gmm_uniform",
        type=str,
        help="name of experiment to run" "for pretrained model in the code, exp can be pretrained",
    )
    parser.add_argument(
        "--resume-iter",
        default=130000,
        type=int,
        help="list the iteration from which to continue training",
    )
    parser.add_argument(
        "--task",
        default="pair_atom",
        type=str,
        help="use a series of different tasks"
        "pair_atom for measuring differences between atoms"
        "sequence_recover for sequence recovery "
        "pack_rotamer for sequence recovery "
        "rotamer_trials for rotamer trials ",
    )
    parser.add_argument(
        "--pdb-name", default="6mdw.0", type=str, help="pdb on which to run analysis"
    )
    parser.add_argument(
        "--outdir", default="rotation_energies", type=str, help="output for experiments"
    )

    #############################
    ##### Analysis hyperparameters
    #############################

    parser.add_argument(
        "--rotations", default=1, type=int, help="number of rotations to use when evaluating"
    )
    parser.add_argument(
        "--sample-mode", default="rosetta", type=str, help="gmm or weighted_gauss or rosetta"
    )
    parser.add_argument(
        "--ensemble", default=1, type=int, help="number of ensembles to use for evaluation"
    )
    parser.add_argument(
        "--neg-sample",
        default=500,
        type=int,
        help="number of negative rotamer samples for rotamer trials (1-1 ratio)",
    )

    return parser


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


def rotamer_trials(model, FLAGS, test_dataset):
    test_files = test_dataset.files
    random.shuffle(test_files)
    db = load_rotamor_library()
    so3 = special_ortho_group(3)

    node_embed_evals = []
    nminibatch = 4

    if FLAGS.ensemble > 1:
        models = model

    # The three different sampling methods are weighted_gauss, gmm, rosetta
    rotamer_scores_total = []
    surface_scores_total = []
    buried_scores_total = []
    amino_recovery_total = {}
    for k, v in kvs.items():
        amino_recovery_total[k.lower()] = []

    counter = 0
    rotations = FLAGS.rotations

    for test_file in tqdm(test_files):
        (node_embed,) = pickle.load(open(test_file, "rb"))
        node_embed_original = node_embed
        par, child, pos, pos_exist, res, chis_valid = parse_dense_format(node_embed)
        angles = compute_dihedral(par, child, pos, pos_exist)

        amino_recovery = {}
        for k, v in kvs.items():
            amino_recovery[k.lower()] = []

        if node_embed is None:
            continue

        rotamer_scores = []
        surface_scores = []
        buried_scores = []
        types = []

        gt_chis = []
        node_embed_evals = []
        neg_chis = []
        valid_chi_idxs = []
        res_names = []

        neg_sample = FLAGS.neg_sample

        n_amino = pos.shape[0]
        amino_recovery_curr = {}
        for idx in range(1, n_amino - 1):
            res_name = res[idx]
            if res_name == "gly" or res_name == "ala":
                continue

            res_names.append(res_name)

            gt_chis.append(angles[idx, 4:8])
            valid_chi_idxs.append(chis_valid[idx, :4])

            hacked_pos = np.copy(pos)
            swap_hacked_pos = np.swapaxes(hacked_pos, 0, 1)  # (20, 59, 3)
            idxs_to_change = swap_hacked_pos[4] == [0, 0, 0]  # (59, 3)
            swap_hacked_pos[4][idxs_to_change] = swap_hacked_pos[3][idxs_to_change]
            hacked_pos_final = np.swapaxes(swap_hacked_pos, 0, 1)

            neighbors = np.linalg.norm(pos[idx : idx + 1, 4] - hacked_pos_final[:, 4], axis=1) < 10
            neighbors = neighbors.astype(np.int32).sum()

            if neighbors >= 24:
                types.append("buried")
            elif neighbors < 16:
                types.append("surface")
            else:
                types.append("neutral")

            if neighbors >= 24:
                tresh = 0.98
            else:
                tresh = 0.95

            if FLAGS.sample_mode == "weighted_gauss":
                chis_list = interpolated_sample_normal(
                    db, angles[idx, 1], angles[idx, 2], res[idx], neg_sample, uniform=False
                )
            elif FLAGS.sample_mode == "gmm":
                chis_list = mixture_sample_normal(
                    db, angles[idx, 1], angles[idx, 2], res[idx], neg_sample, uniform=False
                )
            elif FLAGS.sample_mode == "rosetta":
                chis_list = exhaustive_sample(
                    db, angles[idx, 1], angles[idx, 2], res[idx], tresh=tresh
                )

            neg_chis.append(chis_list)

            node_neg_embeds = []
            length_chis = len(chis_list)
            for i in range(neg_sample):
                chis_target = angles[:, 4:8].copy()

                if i >= len(chis_list):
                    node_neg_embed_copy = node_neg_embed.copy()
                    node_neg_embeds.append(node_neg_embeds[i % length_chis])
                    neg_chis[-1].append(chis_list[i % length_chis])
                    continue

                chis = chis_list[i]

                chis_target[idx] = (
                    chis * chis_valid[idx, :4] + (1 - chis_valid[idx, :4]) * chis_target[idx]
                )
                pos_new = rotate_dihedral_fast(
                    angles, par, child, pos, pos_exist, chis_target, chis_valid, idx
                )

                node_neg_embed = reencode_dense_format(node_embed, pos_new, pos_exist)
                node_neg_embeds.append(node_neg_embed)

            node_neg_embeds = np.array(node_neg_embeds)
            dist = np.square(node_neg_embeds[:, :, -3:] - pos[idx : idx + 1, 4:5, :]).sum(axis=2)
            close_idx = np.argsort(dist)
            node_neg_embeds = np.take_along_axis(node_neg_embeds, close_idx[:, :64, None], axis=1)
            node_neg_embeds[:, :, -3:] = node_neg_embeds[:, :, -3:] / 10.0
            node_neg_embeds[:, :, -3:] = node_neg_embeds[:, :, -3:] - np.mean(
                node_neg_embeds[:, :, -3:], axis=1, keepdims=True
            )

            node_embed_evals.append(node_neg_embeds)

            if len(node_embed_evals) == nminibatch or idx == (n_amino - 2):
                n_entries = len(node_embed_evals)
                node_embed_evals = np.concatenate(node_embed_evals)
                s = node_embed_evals.shape

                # For sample rotations per batch
                node_embed_evals = np.tile(node_embed_evals[:, None, :, :], (1, rotations, 1, 1))
                rot_matrix = so3.rvs(rotations)

                if rotations == 1:
                    rot_matrix = rot_matrix[None, :, :]

                node_embed_evals[:, :, :, -3:] = np.matmul(
                    node_embed_evals[:, :, :, -3:], rot_matrix[None, :, :, :]
                )
                node_embed_evals = node_embed_evals.reshape((-1, *s[1:]))

                node_embed_feed = torch.from_numpy(node_embed_evals).float().cuda()

                with torch.no_grad():
                    energy = 0
                    if FLAGS.ensemble > 1:
                        for model in models:
                            energy_tmp = model.forward(node_embed_feed)
                            energy = energy + energy_tmp
                    else:
                        energy = model.forward(node_embed_feed)

                energy = energy.view(n_entries, -1, rotations).mean(dim=2)
                select_idx = torch.argmin(energy, dim=1).cpu().numpy()

                for i in range(n_entries):
                    select_idx_i = select_idx[i]
                    valid_chi_idx = valid_chi_idxs[i]
                    rotamer_score, _ = compute_rotamer_score_planar(
                        gt_chis[i], neg_chis[i][select_idx_i], valid_chi_idx[:4], res_names[i]
                    )
                    rotamer_scores.append(rotamer_score)

                    amino_recovery[str(res_names[i])] = amino_recovery[str(res_names[i])] + [
                        rotamer_score
                    ]

                    if types[i] == "buried":
                        buried_scores.append(rotamer_score)
                    elif types[i] == "surface":
                        surface_scores.append(rotamer_score)

                gt_chis = []
                node_embed_evals = []
                neg_chis = []
                valid_chi_idxs = []
                res_names = []
                types = []

            counter += 1

        rotamer_scores_total.append(np.mean(rotamer_scores))

        if len(buried_scores) > 0:
            buried_scores_total.append(np.mean(buried_scores))
        surface_scores_total.append(np.mean(surface_scores))

        for k, v in amino_recovery.items():
            if len(v) > 0:
                amino_recovery_total[k] = amino_recovery_total[k] + [np.mean(v)]

        print(
            "Obtained a rotamer recovery score of ",
            np.mean(rotamer_scores_total),
            np.std(rotamer_scores_total) / len(rotamer_scores_total) ** 0.5,
        )
        print(
            "Obtained a buried recovery score of ",
            np.mean(buried_scores_total),
            np.std(buried_scores_total) / len(buried_scores_total) ** 0.5,
        )
        print(
            "Obtained a surface recovery score of ",
            np.mean(surface_scores_total),
            np.std(surface_scores_total) / len(surface_scores_total) ** 0.5,
        )
        for k, v in amino_recovery_total.items():
            print(
                "per amino acid recovery of {} score of ".format(k),
                np.mean(v),
                np.std(v) / len(v) ** 0.5,
            )


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


def make_tsne(model, FLAGS, node_embed):
    """
    grab representations for each of the residues in a pdb
    """
    pdb_name = FLAGS.pdb_name
    pickle_file = MMCIF_PATH + f"/mmCIF/{pdb_name[1:3]}/{pdb_name}.p"
    (node_embed,) = pickle.load(open(pickle_file, "rb"))
    par, child, pos, pos_exist, res, chis_valid = parse_dense_format(node_embed)
    angles = compute_dihedral(par, child, pos, pos_exist)

    all_hiddens = []
    all_energies = []

    n_rotations = 2
    so3 = special_ortho_group(3)
    rot_matrix = so3.rvs(n_rotations)
    rot_matrix = torch.from_numpy(rot_matrix).float().cuda()

    for idx in range(len(res)):
        # sort the atoms by how far away they are
        # sort key is the first atom on the sidechain
        pos_chosen = pos[idx, 4]
        close_idx = np.argsort(np.square(node_embed[:, -3:] - pos_chosen).sum(axis=1))

        # Grab the 64 closest atoms
        node_embed_short = node_embed[close_idx[: FLAGS.max_size]].copy()

        # Normalize each coordinate of node_embed to have x, y, z coordinate to be equal 0
        node_embed_short[:, -3:] = node_embed_short[:, -3:] - np.mean(
            node_embed_short[:, -3:], axis=0
        )
        node_embed_short = torch.from_numpy(node_embed_short).float().cuda()

        node_embed_short = node_embed_short[None, :, :].repeat(n_rotations, 1, 1)
        node_embed_short[:, :, -3:] = torch.matmul(node_embed_short[:, :, -3:], rot_matrix)

        # Compute the energies for the n_rotations * batch_size for this window of 64 atoms.
        # Batch the first two dimensions, then pull them apart aftewrads.
        # node_embed_short = node_embed_short.reshape(node_embed_short.shape[0] * node_embed_short.shape[1], *node_embed_short.shape[2:])

        energies, hidden = model.forward(node_embed_short, return_hidden=True)  # (12000, 1)

        # all_hiddens.append(hidden.mean(0)) # mean over the rotations
        all_hiddens.append(hidden[0])  # take first rotation
        all_energies.append(energies[0])

    surface_core_type = []
    for idx in range(len(res)):
        # >16 c-beta neighbors within 10A of its own c-beta (or c-alpha for gly).
        hacked_pos = np.copy(pos)
        swap_hacked_pos = np.swapaxes(hacked_pos, 0, 1)  # (20, 59, 3)
        idxs_to_change = swap_hacked_pos[4] == [0, 0, 0]  # (59, 3)
        swap_hacked_pos[4][idxs_to_change] = swap_hacked_pos[3][idxs_to_change]
        hacked_pos_final = np.swapaxes(swap_hacked_pos, 0, 1)

        dist = np.sqrt(
            np.square(hacked_pos_final[idx : idx + 1, 4] - hacked_pos_final[:, 4]).sum(axis=1)
        )
        neighbors = (dist < 10).sum()

        if neighbors >= 24:
            surface_core_type.append("core")
        elif neighbors <= 16:
            surface_core_type.append("surface")
        else:
            surface_core_type.append("unlabeled")

    output = {
        "res": res,
        "surface_core_type": surface_core_type,
        "all_hiddens": torch.stack(all_hiddens).cpu().numpy(),
        "all_energies": torch.stack(all_energies).cpu().numpy(),
    }
    # Dump the output
    output_path = osp.join(FLAGS.outdir, f"{pdb_name}_representations.p")
    if not osp.exists(FLAGS.outdir):
        os.makedirs(FLAGS.outdir)
    pickle.dump(output, open(output_path, "wb"))


def new_model(model, FLAGS, node_embed):
    BATCH_SIZE = 120
    pdb_name = FLAGS.pdb_name  #'6mdw.0'
    pickle_file = f"/private/home/yilundu/dataset/mmcif/mmCIF/{pdb_name[1:3]}/{pdb_name}.p"
    (node_embed,) = pickle.load(open(pickle_file, "rb"))
    par, child, pos, pos_exist, res, chis_valid = parse_dense_format(node_embed)
    angles = compute_dihedral(par, child, pos, pos_exist)

    chis_target_initial = angles[
        :, 4:8
    ].copy()  # dihedral for backbone (:4); dihedral for sidechain (4:8)

    NUM_RES = len(res)
    all_energies = np.empty((NUM_RES, 4, 360))  # 4 is number of possible chi angles

    surface_core_type = []
    for idx in range(NUM_RES):
        dist = np.sqrt(np.square(pos[idx : idx + 1, 2] - pos[:, 2]).sum(axis=1))
        neighbors = (dist < 10).sum()
        if neighbors >= 24:
            surface_core_type.append("core")
        elif neighbors <= 16:
            surface_core_type.append("surface")
        else:
            surface_core_type.append("unlabeled")

    for idx in tqdm(range(NUM_RES)):
        for chi_num in range(4):
            if not chis_valid[idx, chi_num]:
                continue

            # init_angle = chis_target[idx, chi_num]
            for angle_deltas in batch(range(-180, 180, 3), BATCH_SIZE):
                pre_rot_node_embed_short = []
                for angle_delta in angle_deltas:
                    chis_target = chis_target_initial.copy()  # make a local copy

                    # modify the angle by angle_delta amount. rotate to chis_target
                    chis_target[
                        idx, chi_num
                    ] += angle_delta  # Set the specific chi angle to be the sampled value

                    # pos_new is n residues x 20 atoms x 3 (xyz)
                    pos_new = rotate_dihedral_fast(
                        angles, par, child, pos, pos_exist, chis_target, chis_valid, idx
                    )
                    node_neg_embed = reencode_dense_format(node_embed, pos_new, pos_exist)

                    # sort the atoms by how far away they are
                    # sort key is the first atom on the sidechain
                    pos_chosen = pos_new[idx, 4]
                    close_idx = np.argsort(
                        np.square(node_neg_embed[:, -3:] - pos_chosen).sum(axis=1)
                    )

                    # Grab the 64 closest atoms
                    node_embed_short = node_neg_embed[close_idx[: FLAGS.max_size]].copy()

                    # Normalize each coordinate of node_embed to have x, y, z coordinate to be equal 0
                    node_embed_short[:, -3:] = node_embed_short[:, -3:] - np.mean(
                        node_embed_short[:, -3:], axis=0
                    )
                    node_embed_short = torch.from_numpy(node_embed_short).float().cuda()
                    pre_rot_node_embed_short.append(node_embed_short.unsqueeze(0))
                pre_rot_node_embed_short = torch.stack(pre_rot_node_embed_short)

                # Now rotate all elements
                n_rotations = 100
                so3 = special_ortho_group(3)
                rot_matrix = so3.rvs(n_rotations)  # n x 3 x 3
                node_embed_short = pre_rot_node_embed_short.repeat(1, n_rotations, 1, 1)
                rot_matrix = torch.from_numpy(rot_matrix).float().cuda()
                node_embed_short[:, :, :, -3:] = torch.matmul(
                    node_embed_short[:, :, :, -3:], rot_matrix
                )  # (batch_size, n_rotations, 64, 20)

                # Compute the energies for the n_rotations * batch_size for this window of 64 atoms.
                # Batch the first two dimensions, then pull them apart aftewrads.
                node_embed_short = node_embed_short.reshape(
                    node_embed_short.shape[0] * node_embed_short.shape[1],
                    *node_embed_short.shape[2:],
                )
                energies = model.forward(node_embed_short)  # (12000, 1)

                # divide the batch dimension by the 10 things we just did
                energies = energies.reshape(BATCH_SIZE, -1)  # (10, 200)

                # Average the energy across the n_rotations, but keeping batch-wise seperate
                energies = energies.mean(1)  # (10, 1)

                # Save the result
                all_energies[idx, chi_num, angle_deltas] = energies.cpu().numpy()

    # Can use these for processing later.
    avg_chi_angle_energy = (all_energies * chis_valid[:NUM_RES, :4, None]).sum(0) / np.expand_dims(
        chis_valid[:NUM_RES, :4].sum(0), 1
    )  # normalize by how many times each chi angle occurs
    output = {
        "all_energies": all_energies,
        "chis_valid": chis_valid,
        "chis_target_initial": chis_target_initial,
        "avg_chi_angle_energy": avg_chi_angle_energy,  # make four plots from this (4, 360),
        "res": res,
        "surface_core_type": surface_core_type,
    }
    # Dump the output
    output_path = osp.join(FLAGS.outdir, f"{pdb_name}_rot_energies.p")
    if not osp.exists(FLAGS.outdir):
        os.makedirs(FLAGS.outdir)
    pickle.dump(output, open(output_path, "wb"))


def pair_model(model, FLAGS, node_embed):
    # Generate a dataset where two atoms are very close to each other and everything else is very far
    # Indices for atoms
    atom_names = ["X", "C", "N", "O", "S"]
    residue_names = [
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLU",
        "GLN",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
    ]

    energies_output_dict = {}

    def make_key(n_rotations, residue_name1, residue_name2, atom_name1, atom_name2):
        return f"{n_rotations}_{residue_name1}_{residue_name2}_{atom_name1}_{atom_name2}"

    # Save a copy of the node embed
    node_embed = node_embed[0]
    node_embed_orig = node_embed.clone()

    # Try different combinations
    for n_rotations in [5]:
        # Rotations
        so3 = special_ortho_group(3)
        rot_matrix_neg = so3.rvs(n_rotations)  # number of random rotations to average

        residue_names_proc = ["ALA", "TYR", "LEU"]
        atom_names_proc = ["C", "N", "O"]
        for residue_name1, residue_name2 in itertools.product(residue_names_proc, repeat=2):
            for atom_name1, atom_name2 in itertools.product(atom_names_proc, repeat=2):
                eps = []
                energies = []

                residue_index1 = residue_names.index(residue_name1)
                residue_index2 = residue_names.index(residue_name2)
                atom_index1 = atom_names.index(atom_name1)
                atom_index2 = atom_names.index(atom_name2)

                for i in np.linspace(0.1, 1.0, 100):
                    node_embed = node_embed_orig.clone()
                    node_embed[-2, -3:] = torch.Tensor([1.0, 0.5, 0.5])
                    node_embed[-1, -3:] = torch.Tensor([1.0 + i, 0.5, 0.5])
                    node_embed[-1, 0] = residue_index1
                    node_embed[-2, 0] = residue_index2
                    node_embed[-1, 1] = atom_index1
                    node_embed[-2, 1] = atom_index2
                    node_embed[-1, 2] = 6  # res_counter
                    node_embed[-2, 2] = 6  # res_counter

                    node_embed = np.tile(node_embed[None, :, :], (n_rotations, 1, 1))
                    node_embed[:, :, -3:] = np.matmul(node_embed[:, :, -3:], rot_matrix_neg)
                    node_embed_feed = torch.Tensor(node_embed).cuda()
                    node_embed_feed[:, :, -3:] = node_embed_feed[:, :, -3:] - node_embed_feed[
                        :, :, -3:
                    ].mean(dim=1, keepdim=True)
                    energy = model.forward(node_embed_feed)  #
                    energy = energy.mean()

                    eps.append(i * 10)
                    energies.append(energy.item())

                key = make_key(n_rotations, residue_name1, residue_name2, atom_name1, atom_name2)
                energies_output_dict[key] = (eps, energies)

                # Optionally make plots here -- potentially add conditions to avoid making too many
                plt.plot(eps, energies)
                plt.xlabel("Atom Distance")
                plt.ylabel("Energy")
                plt.title(
                    f"{n_rotations} rots: {atom_name1}, {atom_name2} distance in {residue_name1}/{residue_name2}"
                )
                plt.savefig(
                    f"distance_plots/{n_rotations}_{atom_name1}_{atom_name2}_in_{residue_name1}_{residue_name2}_distance.png"
                )
                plt.clf()

    # Back to outside
    output_path = osp.join(FLAGS.outdir, "atom_distances.p")
    pickle.dump(energies_output_dict, open(output_path, "wb"))


def main_single(FLAGS):
    FLAGS_OLD = FLAGS

    logdir = osp.join(FLAGS.logdir, FLAGS.exp)

    if FLAGS.resume_iter != 0:
        model_path = osp.join(logdir, "model_{}".format(FLAGS.resume_iter))
        checkpoint = torch.load(model_path)
        try:
            FLAGS = checkpoint["FLAGS"]

            FLAGS.resume_iter = FLAGS_OLD.resume_iter
            FLAGS.neg_sample = FLAGS_OLD.neg_sample

            for key in FLAGS.keys():
                if "__" not in key:
                    FLAGS_OLD[key] = getattr(FLAGS, key)

            FLAGS = FLAGS_OLD
        except Exception as e:
            print(e)
            print("Didn't find keys in checkpoint'")

    models = []
    if FLAGS.ensemble > 1:
        for i in range(FLAGS.ensemble):
            if FLAGS.model == "transformer":
                model = RotomerTransformerModel(FLAGS).eval()
            elif FLAGS.model == "fc":
                model = RotomerFCModel(FLAGS).eval()
            elif FLAGS.model == "s2s":
                model = RotomerSet2SetModel(FLAGS).eval()
            elif FLAGS.model == "graph":
                model = RotomerGraphModel(FLAGS).eval()
            elif FLAGS.model == "s2s":
                model = RotomerSet2SetModel(FLAGS).eval()
            models.append(model)
    else:
        if FLAGS.model == "transformer":
            model = RotomerTransformerModel(FLAGS).eval()
        elif FLAGS.model == "fc":
            model = RotomerFCModel(FLAGS).eval()
        elif FLAGS.model == "s2s":
            model = RotomerSet2SetModel(FLAGS).eval()

    gpu = 0
    world_size = 0

    it = FLAGS.resume_iter

    if not osp.exists(logdir):
        os.makedirs(logdir)

    checkpoint = None

    if FLAGS.ensemble > 1:
        for i, model in enumerate(models):
            if FLAGS.resume_iter != 0:
                model_path = osp.join(logdir, "model_{}".format(FLAGS.resume_iter - i * 1000))
                checkpoint = torch.load(model_path)
                try:
                    model.load_state_dict(checkpoint["model_state_dict"])
                except Exception as e:
                    print("Transfer between distributed to non-distributed")

                    if world_size > 1:
                        model_state_dict = {
                            k.replace("module.", ""): v
                            for k, v in checkpoint["model_state_dict"].items()
                        }
                    else:
                        model_state_dict = {
                            k.replace("module.", ""): v
                            for k, v in checkpoint["model_state_dict"].items()
                        }
                    model.load_state_dict(model_state_dict)

            models[i] = nn.DataParallel(model)
        model = models
    else:
        if FLAGS.resume_iter != 0:
            model_path = osp.join(logdir, "model_{}".format(FLAGS.resume_iter))
            checkpoint = torch.load(model_path)
            try:
                model.load_state_dict(checkpoint["model_state_dict"])
            except Exception as e:
                print("Transfer between distributed to non-distributed")

                if world_size > 1:
                    model_state_dict = {
                        k.replace("module.", ""): v
                        for k, v in checkpoint["model_state_dict"].items()
                    }
                else:
                    model_state_dict = {
                        k.replace("module.", ""): v
                        for k, v in checkpoint["model_state_dict"].items()
                    }
                model.load_state_dict(model_state_dict)
            model = nn.DataParallel(model)

    if FLAGS.cuda:
        if FLAGS.ensemble > 1:
            for i, model in enumerate(models):
                models[i] = model.cuda(gpu)

            model = models
        else:
            torch.cuda.set_device(gpu)
            model = model.cuda(gpu)

    FLAGS.multisample = 1
    print("New Values of args: ", FLAGS)

    with torch.no_grad():
        if FLAGS.task == "pair_atom":
            train_dataset = MMCIFTransformer(FLAGS, mmcif_path=MMCIF_PATH, split="train")
            (
                node_embeds,
                node_embeds_negatives,
                select_atom_idxs,
                select_atom_masks,
                select_chis_valids,
                select_ancestors,
            ) = train_dataset[256]
            pair_model(model, FLAGS, node_embeds)
        if FLAGS.task == "pack_rotamer":
            train_dataset = MMCIFTransformer(FLAGS, mmcif_path=MMCIF_PATH, split="train")
            node_embed, _, _, _, _, _, = train_dataset[0]
            pack_rotamer(model, FLAGS, node_embed)
        if FLAGS.task == "rotamer_trial":
            test_dataset = MMCIFTransformer(FLAGS, mmcif_path=MMCIF_PATH, split="test")
            rotamer_trials(model, FLAGS, test_dataset)
        if FLAGS.task == "new_model":
            train_dataset = MMCIFTransformer(FLAGS, mmcif_path=MMCIF_PATH, split="train")
            node_embed, _, _, _, _, _, = train_dataset[7]
            new_model(model, FLAGS, node_embed)
        if FLAGS.task == "tsne":
            train_dataset = MMCIFTransformer(FLAGS, mmcif_path=MMCIF_PATH, split="train")
            node_embed, _, _, _, _, _, = train_dataset[3]
            make_tsne(model, FLAGS, node_embed)
        else:
            assert False


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    FLAGS = parser.parse_args()

    # convert to easy_dict; this is what is saved with model checkpoints and used in logic above
    keys = dir(FLAGS)
    flags_dict = EasyDict()
    for key in keys:
        if "__" not in key:
            flags_dict[key] = getattr(FLAGS, key)

    # postprocess arguments
    FLAGS.cuda = not FLAGS.no_cuda

    # set seeds
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)

    main_single(flags_dict)
