# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import os.path as osp
import pickle
import random

import numpy as np
from scipy.stats import special_ortho_group

import gemmi
import torch
from constants import test_rotamers
from math_utils import rotate_v1_v2
from mmcif_utils import (
    compute_dihedral,
    exhaustive_sample,
    interpolated_sample_normal,
    load_rotamor_library,
    mixture_sample_normal,
    parse_dense_format,
    reencode_dense_format,
    rotate_dihedral_fast,
)
from torch.utils.data import Dataset
from tqdm import tqdm


class MMCIFTransformer(Dataset):
    def __init__(
        self,
        FLAGS,
        mmcif_path="./mmcif",
        split="train",
        rank_idx=0,
        world_size=1,
        uniform=True,
        weighted_gauss=False,
        gmm=False,
        chi_mean=False,
        valid=False,
    ):
        files = []
        dirs = os.listdir(osp.join(mmcif_path, "mmCIF"))

        self.split = split
        self.so3 = special_ortho_group(3)
        self.chi_mean = chi_mean
        self.weighted_gauss = weighted_gauss
        self.gmm = gmm
        self.uniform = uniform

        # Filter out proteins in test dataset
        for d in tqdm(dirs):
            directory = osp.join(mmcif_path, "mmCIF", d)
            d_files = os.listdir(directory)
            files_tmp = [osp.join(directory, d_file) for d_file in d_files if ".p" in d_file]

            for f in files_tmp:
                name = f.split("/")[-1]
                name = name.split(".")[0]
                if name in test_rotamers and self.split == "test":
                    files.append(f)
                elif name not in test_rotamers and self.split in ["train", "val"]:
                    files.append(f)

        self.files = files

        if split in ["train", "val"]:
            duplicate_seqs = set()

            # Remove proteins in the train dataset that are too similar to the test dataset
            with open(osp.join(mmcif_path, "duplicate_sequences.txt"), "r") as f:
                for line in f:
                    duplicate_seqs.add(line.strip())

            fids = set()

            # Remove low resolution proteins
            with open(
                osp.join(mmcif_path, "cullpdb_pc90_res1.8_R0.25_d190807_chains14857"), "r"
            ) as f:
                i = 0
                for line in f:
                    if i is not 0:
                        fid = line.split()[0]
                        if fid not in duplicate_seqs:
                            fids.add(fid)

                    i += 1

            files_new = []

            alphabet = []
            for letter in range(65, 91):
                alphabet.append(chr(letter))

            for f in files:
                tup = (f.split("/")[-1]).split(".")

                if int(tup[1]) >= len(alphabet):
                    continue

                seq_id = tup[0].upper() + alphabet[int(tup[1])]

                if seq_id in fids:
                    files_new.append(f)

            self.files = files_new
        elif split == "test":
            fids = set()

            # Remove low resolution proteins
            with open(
                osp.join(mmcif_path, "cullpdb_pc90_res1.8_R0.25_d190807_chains14857"), "r"
            ) as f:
                i = 0
                for line in f:
                    if i is not 0:
                        fid = line.split()[0]
                        fids.add(fid)

                    i += 1

            files_new = []

            alphabet = []
            for letter in range(65, 91):
                alphabet.append(chr(letter))

            for f in files:
                tup = (f.split("/")[-1]).split(".")

                if int(tup[1]) >= len(alphabet):
                    continue

                seq_id = tup[0].upper() + alphabet[int(tup[1])]

                if seq_id in fids:
                    files_new.append(f)

            self.files = files_new

        chunksize = len(self.files) // world_size

        n = len(self.files)

        # Set up a validation dataset
        if split == "train":
            n = self.files[int(0.95 * n) :]
        elif split == "val":
            n = self.files[: int(0.95 * n)]

        self.FLAGS = FLAGS
        self.db = load_rotamor_library()
        print(f"Loaded {len(self.files)} files for {split} dataset split")

        self.split = split

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index, forward=False):
        FLAGS = self.FLAGS

        if FLAGS.single and not forward:
            index = 0

        FLAGS = self.FLAGS
        pickle_file = self.files[index]

        # node_embed: D x 6
        (node_embed,) = pickle.load(open(pickle_file, "rb"))
        node_embed_original = node_embed

        # Remove proteins with small numbers of atoms
        if node_embed.shape[0] < 20:
            return self.__getitem__((index + 1) % len(self.files), forward=True)

        # Remove invalid proteins
        if (
            node_embed.max(axis=0)[2] >= 21
            or node_embed.max(axis=0)[0] >= 20
            or node_embed.max(axis=0)[1] >= 5
        ):
            return self.__getitem__((index + 1) % len(self.files), forward=True)

        par, child, pos, pos_exist, res, chis_valid = parse_dense_format(node_embed)

        if par is None:
            return self.__getitem__((index + 1) % len(self.files), forward=True)

        if len(res) < 5:
            return self.__getitem__((index + 1) % len(self.files), forward=True)

        angles = compute_dihedral(par, child, pos, pos_exist)

        tries = 0
        perm = np.random.permutation(np.arange(1, len(res) - 1))
        select_idxs = []

        while True:
            # Randomly sample an amino acid that are not the first and last amino acid
            idx = perm[tries]
            if res[idx] == "gly" or res[idx] == "ala":
                idx = random.randint(1, len(res) - 2)
            else:
                select_idxs.append(idx)

                if len(select_idxs) == FLAGS.multisample:
                    break

            tries += 1

            if tries > 1000 or tries == perm.shape[0]:
                return self.__getitem__((index + 1) % len(self.files), forward=True)

        node_embeds = []
        node_embeds_negatives = []
        select_atom_idxs = []
        select_atom_masks = []
        select_chis_valids = []
        select_ancestors = []

        for idx in select_idxs:
            neg_samples = []
            gt_chis = [(angles[idx, 4:8], chis_valid[idx, :4])]
            neg_chis = []

            # Choose number of negative samples
            if FLAGS.train and self.split in ["val", "test"]:
                neg_sample = 150
            else:
                neg_sample = FLAGS.neg_sample

            atom_idxs = []
            atoms_mask = []
            chis_valids = []
            ancestors = []

            if self.split == "test":
                dist = np.sqrt(np.square(pos[idx : idx + 1, 2] - pos[:, 2]).sum(axis=1))
                neighbors = (dist < 10).sum()

                # Choose different tresholds of sampling dependent on whether an atom is dense
                # or not
                if neighbors < 24:
                    tresh = 0.95
                else:
                    tresh = 0.98

                if self.weighted_gauss:
                    chis_list = interpolated_sample_normal(
                        self.db,
                        angles[idx, 1],
                        angles[idx, 2],
                        res[idx],
                        neg_sample,
                        uniform=self.uniform,
                    )
                elif self.gmm:
                    chis_list = mixture_sample_normal(
                        self.db,
                        angles[idx, 1],
                        angles[idx, 2],
                        res[idx],
                        neg_sample,
                        uniform=self.uniform,
                    )
                else:
                    chis_list = exhaustive_sample(
                        self.db,
                        angles[idx, 1],
                        angles[idx, 2],
                        res[idx],
                        tresh=tresh,
                        chi_mean=self.chi_mean,
                    )

                    if len(chis_list) < neg_sample:
                        repeat = neg_sample // len(chis_list) + 1
                        chis_list = chis_list * repeat

                    random.shuffle(chis_list)

            else:
                dist = np.sqrt(np.square(pos[idx : idx + 1, 2] - pos[:, 2]).sum(axis=1))
                neighbors = (dist < 10).sum()

                if neighbors < 24:
                    tresh = 1.0
                else:
                    tresh = 1.0

                if self.weighted_gauss:
                    chis_list = interpolated_sample_normal(
                        self.db,
                        angles[idx, 1],
                        angles[idx, 2],
                        res[idx],
                        neg_sample,
                        uniform=self.uniform,
                    )
                elif self.gmm:
                    chis_list = mixture_sample_normal(
                        self.db,
                        angles[idx, 1],
                        angles[idx, 2],
                        res[idx],
                        neg_sample,
                        uniform=self.uniform,
                    )
                else:
                    chis_list = exhaustive_sample(
                        self.db,
                        angles[idx, 1],
                        angles[idx, 2],
                        res[idx],
                        tresh=tresh,
                        chi_mean=self.chi_mean,
                    )

                    if len(chis_list) < neg_sample:
                        repeat = neg_sample // len(chis_list) + 1
                        chis_list = chis_list * repeat

                    random.shuffle(chis_list)

            for i in range(neg_sample):
                chis_target = angles[:, 4:8].copy()
                chis = chis_list[i]

                chis_target[idx] = (
                    chis * chis_valid[idx, :4] + (1 - chis_valid[idx, :4]) * chis_target[idx]
                )
                pos_new = rotate_dihedral_fast(
                    angles, par, child, pos, pos_exist, chis_target, chis_valid, idx
                )

                node_neg_embed = reencode_dense_format(node_embed, pos_new, pos_exist)
                neg_samples.append(node_neg_embed)
                neg_chis.append((chis_target[idx], chis_valid[idx, :4]))
                nelem = pos_exist[:idx].sum()
                offset = pos_exist[idx].sum()
                mask = np.zeros(20)
                mask[:offset] = 1

                atom_idxs.append(
                    np.concatenate(
                        [np.arange(nelem, nelem + offset), np.ones(20 - offset) * (nelem)]
                    )
                )
                atoms_mask.append(mask)
                chis_valids.append(chis_valid[idx, :4].copy())
                ancestors.append(np.stack([par[idx], child[idx]], axis=0))

            node_embed_negative = np.array(neg_samples)

            pos_chosen = pos[idx, 4]

            atoms_mask = np.array(atoms_mask)
            atom_idxs = np.array(atom_idxs)
            chis_valids = np.array(chis_valids)
            ancestors = np.array(ancestors)

            # Choose the closest atoms to the chosen locaiton:
            close_idx = np.argsort(np.square(node_embed[:, -3:] - pos_chosen).sum(axis=1))
            node_embed_short = node_embed[close_idx[: FLAGS.max_size]].copy()

            pos_chosen = pos_new[idx, 4]
            close_idx_neg = np.argsort(
                np.square(node_embed_negative[:, :, -3:] - pos_chosen).sum(axis=2), axis=1
            )

            # Compute the corresponding indices for atom_idxs
            # Get the position of each index ik
            pos_code = np.argsort(close_idx_neg, axis=1)
            choose_idx = np.take_along_axis(pos_code, atom_idxs.astype(np.int32), axis=1)

            if choose_idx.max() >= FLAGS.max_size:
                return self.__getitem__((index + 1) % len(self.files), forward=True)

            node_embed_negative = np.take_along_axis(
                node_embed_negative, close_idx_neg[:, : FLAGS.max_size, None], axis=1
            )

            # Normalize each coordinate of node_embed to have x, y, z coordinate to be equal 0
            node_embed_short[:, -3:] = node_embed_short[:, -3:] - np.mean(
                node_embed_short[:, -3:], axis=0
            )
            node_embed_negative[:, :, -3:] = node_embed_negative[:, :, -3:] - np.mean(
                node_embed_negative[:, :, -3:], axis=1, keepdims=True
            )

            if FLAGS.augment:
                # Now rotate all elements
                rot_matrix = self.so3.rvs(1)
                node_embed_short[:, -3:] = np.matmul(node_embed_short[:, -3:], rot_matrix)

                rot_matrix_neg = self.so3.rvs(node_embed_negative.shape[0])
                node_embed_negative[:, :, -3:] = np.matmul(
                    node_embed_negative[:, :, -3:], rot_matrix_neg
                )

            # # Additionally scale values to be in the same scale
            node_embed_short[:, -3:] = node_embed_short[:, -3:] / 10.0
            node_embed_negative[:, :, -3:] = node_embed_negative[:, :, -3:] / 10.0

            # Augment the data with random rotations
            node_embed_short = torch.from_numpy(node_embed_short).float()
            node_embed_negative = torch.from_numpy(node_embed_negative).float()

            if self.split == "train":
                node_embeds.append(node_embed_short)
                node_embeds_negatives.append(node_embed_negative)
            elif self.split in ["val", "test"]:
                return node_embed_short, node_embed_negative, gt_chis, neg_chis, res[idx]

        return node_embeds, node_embeds_negatives


def collate_fn_transformer(inp):
    node_embed, node_embed_neg = zip(*inp)
    node_embed, node_embed_neg = sum(node_embed, []), sum(node_embed_neg, [])
    max_size = max([ne.size(0) for ne in node_embed])

    neg_sample_size = node_embed_neg[0].size(0)
    sizes = list(node_embed_neg[0].size())
    node_embed_batch = torch.zeros(*(len(node_embed), max_size, node_embed[0].size(1)))
    node_embed_neg_batch = (node_embed_batch.clone()[:, None, :, :]).repeat(1, sizes[0], 1, 1)

    for i, (ne, neg) in enumerate(zip(node_embed, node_embed_neg)):
        node_embed_batch[i, : ne.size(0), :] = ne
        node_embed_neg_batch[i, :, : neg.size(1), :] = neg

    sizes = list(node_embed_neg_batch.size())
    node_embed_neg_batch = node_embed_neg_batch.view(-1, *sizes[2:])

    return node_embed_batch, node_embed_neg_batch


def collate_fn_transformer_test(inp):
    node_embed, node_embed_neg, gt_chis, neg_chis, res = zip(*inp)
    max_size = max([ne.size(0) for ne in node_embed])

    neg_sample_size = node_embed_neg[0].size(0)
    sizes = list(node_embed_neg[0].size())
    node_embed_batch = torch.zeros(*(len(node_embed), max_size, node_embed[0].size(1)))
    node_embed_neg_batch = (node_embed_batch.clone()[:, None, :, :]).repeat(1, sizes[0], 1, 1)

    for i, (ne, neg) in enumerate(zip(node_embed, node_embed_neg)):
        node_embed_batch[i, : ne.size(0), :] = ne
        node_embed_neg_batch[i, :, : neg.size(1), :] = neg

    sizes = list(node_embed_neg_batch.size())
    node_embed_neg_batch = node_embed_neg_batch.view(-1, *sizes[2:])

    return node_embed_batch, node_embed_neg_batch, gt_chis, neg_chis, res
