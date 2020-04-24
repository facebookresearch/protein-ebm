# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Utility code for converting between protein representations.

Note that several methods here are no longer used in any of the training routines.
However, they were quite useful to us during the course of research,
so we are releasing them here in case they help others.
"""
import collections
import os
import os.path as osp
import pickle
import random
from itertools import product
from multiprocessing import Pool

import numpy as np
import pandas as pd

import gemmi
from amino_acid_config import kvs, res_atoms, res_children, res_chis, res_parents
from config import MMCIF_PATH, ROTAMER_LIBRARY_PATH
from constants import atom_names, residue_names
from math_utils import rotate_v1_v2, rotate_v1_v2_vec


def parse_dense_format(node_embed):
    """
    In protein-ebm, we represent amino acids in two different formats.
    This method converts from the dense format to a sparse format.
    
    ===============
    ==== Dense ====
    ===============
    The dense format represents a protein using a is a D x 6 dimensional represention.
    Each 6 dimensional vector represents an atom, following this scheme:
        [1]: amino acid identity of the amino acid the atom is part of (residue_idx)
        [2]: element identity of the amino acid the atom is part of (atom_idx)
        [3]: positional location of atom in the amino acid (atom_num)
        [4..6]: x,y,z coordinates
    The dense format is useful for feeding data into a neural network.

    ===============
    ==== Sparse ===
    ===============
    The sparse format represents a data based on its topology (parent/child/etc).
    It follows this scheme:
        amino_name: amino acid to substitue
        par: A N x 20 encoding of the relative offset of the parent of each atom. For example,
                the amino acid glycine would be represented as [-18 -1 -1 -1 0, ...]
        child: A N x 20 encoding of the child of each atom. For example, the amino acid glycine
                would be represented as [1 1 18 0 0 0 ..]
        pos_exist: A N x 20 mask encoding of which atoms are valid for each amino acid so for
        example the amino acid glycine would be represented as [1 1 1 1 0 0 ...]
        chi_valid: A N x 5 mask encoding which chi angles are valid, so for example glycine would
        be represented as [0 0 0 0 0]
        pos: A N x 20 x 3 encoding the (x, y, z) coordinates of each atom per amino acid in a protein
        i: amino acid position to substitute
        sequence_map: map from amino acid to structure
        rotate_matrix: matrix of rotation to amino acid position
    This format is easier for manipulating the proteins, e.g changing the rotamers
    during negative sampling. 

    See comments in the implementation below for more details.
    """

    # The input is a list of atoms. We keep track of how many we have processed.
    start = 0

    # Construct amino acid-level information from the atomic inputs
    # Each amino acid is described on the atomic-level by 20-dim lists
    pars = []  # ordinal distance of parent atoms
    childs = []  # ordinal distance of cildren atoms
    pos = []  # 3d translations of each atom
    pos_exists = []  # whether a position exists or not
    residues = []  # the name of the amino acid
    chis_valid = []  # a 20-dim list describing which atoms are part of the chi angle

    # consume all of the atoms in the input
    while start < node_embed.shape[0]:
        idx = int(node_embed[start, 0])
        residue = residue_names[idx]

        # Get the parent and child representation (see amino_acid_config.py)
        par = res_parents[residue].copy()
        child = res_children[residue].copy()
        n = len(par)

        # 20-dim mask of which positions encode meaningful values
        pos_exist = [1] * n + [0] * (20 - n)  # this is the mask

        # pad up to 20-dim with 0s
        par = par + [0] * (20 - n)
        child = child + [0] * (20 - len(child))

        # x,y,z coordinates for each of the atoms in the amino acid, padded to 20-dim
        pos_temp = np.concatenate(
            [node_embed[start : start + n, -3:], np.zeros((20 - n, 3))], axis=0
        )

        # If we can fit these n atom in, then record the information
        if start + n <= node_embed.shape[0]:
            pars.append(par)
            childs.append(child)
            pos.append(pos_temp)
            pos_exists.append(pos_exist)
            chis = res_chis[residue]
            chis_valid.append([1] * len(chis) + [0] * (20 - len(chis)))
            residues.append(residue.lower())

        # All atoms from start <-> start+n should belong to the same amino acid
        if not (node_embed[start : start + n, 0] == idx).all():
            return None, None, None, None, None, None

        # keep track of number of atoms consumeed
        start = start + n

    # Don't proceess single amino acid prorteins
    if len(pos) < 2:
        return None, None, None, None, None, None

    # Wrap the results in numpy arrays
    pars, childs, pos, pos_exists, chis_valid = (
        np.array(pars),
        np.array(childs),
        np.stack(pos, axis=0),
        np.array(pos_exists),
        np.array(chis_valid),
    )

    # The code above assumes that each nitrogen is connected to previous carbon
    # and each carbon is connected to the next nitrogen. This is not the case
    # for the N-terminus and C-terminus, so we need to override those cases.
    pars[0, 0] = 0
    childs[-1, 2] = 0

    # return the new encoding in amino acid form
    return pars, childs, pos, pos_exists, residues, chis_valid


def reencode_dense_format(node_embed, pos_new, pos_exist):
    """Updates x,y,z positions in dense encoding with new positions"""
    node_embed_new = node_embed.copy()
    pos_mask = pos_exist.astype(np.bool)
    elem_num = pos_mask.sum()
    node_embed_new[:elem_num, -3:] = pos_new[pos_mask]

    return node_embed_new


def cif_to_embed(cif_file, ix=None, parse_skip=False):
    """
    Parses a CIF file into a more convenient representation.

    # Embedding format for nodes:
    # 'one hot amino acid' amino type of molecule
    # 'x, y, z' positional encoding
    # 'one hot representation of atom type', either C, CA, N, O,

    """
    st = gemmi.read_structure(cif_file)

    results = []
    skips = []
    for model in st:
        for i, chain in enumerate(model):

            if (ix is not None) and (ix != i):
                continue

            atoms = []
            node_embeddings = []
            for j, residue in enumerate(chain):
                translation = []

                if residue.name not in residue_names:
                    # Skip over any structure that contains nucleotides
                    if residue.name in ["DA", "DC", "DG", "DT"]:
                        return None, None
                    else:
                        continue

                residue_counter = 0
                namino_elements = len(res_parents[residue.name])
                amino_atoms = res_atoms[residue.name]

                residue_atoms = []
                residue_embed = []

                # reisdue object contains information about the residue, including identity
                # and spatial coordiantes for atoms in the residue. We parse this into a
                # dense encoding, for feeding into a neural network.
                node_embed = parse_residue_embed(residue)

                if len(node_embed) == 0:
                    skips.append(j)

                node_embeddings.extend(node_embed)

            node_embeddings = np.array(node_embeddings)

            result = (node_embeddings,)
            results.append(result)

    if parse_skip:
        return st, results, skips
    else:
        return st, results


def vis_cif(cif_path, im_path):
    import pymol
    from pymol import cmd

    cmd.load(cif_path, "mov")
    cmd.zoom()
    cmd.png(im_path, 300, 200)


def compute_chi_angle_st(st, ix):
    angles = []

    num = int(ix)
    chain_counter = 0
    for model in st:
        for chain in model:
            if num != chain_counter:
                chain_counter += 1
                continue
            else:
                for residue in chain:
                    if residue.name in residue_names:
                        chi_angles = compute_chi_angle_residue(residue)
                        if chi_angles is not None:
                            angles.append(chi_angles)
                return angles


def compute_chi_angle_residue(residue):
    # look up the atoms that are used for computing the chi angles.
    chi_angles_atoms = kvs[residue.name]

    angles = []

    try:
        for chi_angles_atom in chi_angles_atoms:
            atoms = chi_angles_atom.split("-")
            pos = []

            for atom in atoms:
                # In some cases, amino acid side chains are listed with CD1 instead of CD
                if atom == "CD":
                    if "CD" not in residue:
                        atom = residue["CD1"]
                    else:
                        atom = residue[atom]
                else:
                    atom = residue[atom]

                pos.append((atom.pos.x, atom.pos.y, atom.pos.z))

            pos = np.array(pos)
            diff_vec = pos[2] - pos[1]
            # Compute the axis in which we are computing the dihedral angle
            diff_vec_normalize = diff_vec / np.linalg.norm(diff_vec)

            diff_bot = pos[0] - pos[1]
            diff_top = pos[3] - pos[2]

            # Now project diff_bot and diff_top to be on the plane
            diff_bot = diff_bot - diff_bot.dot(diff_vec_normalize) * diff_vec_normalize
            diff_top = diff_top - diff_top.dot(diff_vec_normalize) * diff_vec_normalize

            diff_bot_normalize = diff_bot / np.linalg.norm(diff_bot)
            diff_top_normalize = diff_top / np.linalg.norm(diff_top)

            # Compute the dot product for cos and cross product for sin
            sin = (np.cross(diff_bot_normalize, diff_top_normalize) * diff_vec_normalize).sum(
                axis=1
            )
            cos = diff_bot_normalize.dot(diff_top_normalize)

            # print("trig value ", sin, cos, np.linalg.norm([sin, cos]))
            angle = np.arctan2(sin, cos)
            # print("angle ", angle)

            angles.append(angle)
    except Exception as e:
        return None

    return angles


def parse_cif(path):
    base_folder, f = osp.split(path)
    base_name, *junk = f.split(".")

    st, infos = cif_to_embed(path)

    if infos is not None:
        for i, info in enumerate(infos):
            pickle_file = osp.join(base_folder, "{}.{}.p".format(base_name, i))
            pickle.dump(info, open(pickle_file, "wb"))

    return None


def script_parse_cif():
    mmcif_path = osp.join(MMCIF_PATH, "mmCIF")
    files = []
    dirs = os.listdir(mmcif_path)
    pool = Pool()

    for d in dirs:
        directory = osp.join(mmcif_path, d)
        d_files = os.listdir(directory)
        files_tmp = [osp.join(directory, d_file) for d_file in d_files if ".cif" in d_file]
        files.extend(files_tmp)

    pool.map(parse_cif, files)


def clean_cif():
    mmcif_path = osp.join(MMCIF_PATH, mmCIF)
    dirs = os.listdir(mmcif_path)

    for d in dirs:
        directory = osp.join(mmcif_path, d)
        d_files = os.listdir(directory)
        files_tmp = [osp.join(directory, d_file) for d_file in d_files if ".p" in d_file]

        for f in files_tmp:
            os.remove(f)


def recorrect_name(name):
    if (name[-1]).isdigit() and name[-1] == "1":
        return name[:-1]
    elif not (name[-1].isdigit()):
        return name + "1"
    else:
        return name


def _parse_residue(residue):
    """Obtains a sparse representation of residue from gemmi"""

    # list of atoms in the residue (e.g. N-CA-C-O)
    atoms = res_atoms[residue.name]

    # ordinal encoding of how far away the parents are
    parents = res_parents[residue.name]

    # ordinal encoding of how far away the children are
    children = res_children[residue.name]

    # atoms belonging to chi anglse
    chis = res_chis[residue.name]

    # accumulate the xyz postions of the atoms, and node_embed encodings
    pos, node_embeds = [], []

    residue_counter = 0
    for atom in atoms:
        if atom in residue:
            atom = residue[atom]
        elif recorrect_name(atom) in residue:
            atom = residue[recorrect_name(atom)]
        else:
            return None

        pos.append((atom.pos.x, atom.pos.y, atom.pos.z))
        node_embeds.append(
            (
                residue_names.index(residue.name),
                atom_names.index(atom.element.name),
                residue_counter,
                atom.pos.x,
                atom.pos.y,
                atom.pos.z,
            )
        )
        residue_counter = residue_counter + 1

    # 20-dim mask for each residue for atom existence
    exist = [1] * len(parents) + [0] * (20 - len(parents))

    # pad the parents and children to 20-dim
    parents = parents + [0] * (20 - len(parents))
    children = children + [0] * (20 - len(children))

    # place the x,y,z coordinates into a numpy array
    pos_fill = np.zeros((20, 3))
    pos_fill[: len(pos)] = pos

    # pad the chi angles
    chis = [1] * len(chis) + [0] * (5 - len(chis))

    # return the new representation
    return parents, children, pos_fill, exist, chis, node_embeds


# shorthand methods for the above, since logic is the same
def parse_residue(residue):
    ret = _parse_residue(residue, 0)

    if ret:
        parents, children, pos_fill, exist, chis, _, _ = ret
        return parents, children, pos_fill, exist, chis
    else:
        return None, None, None, None, None


def parse_residue_embed(residue):
    ret = _parse_residue(residue)

    if ret:
        _, _, _, _, _, node_embeds = ret
        return node_embeds
    else:
        return []


def flatten(arr):
    return arr.reshape((-1, *arr.shape[2:]))


def rotate_dihedral_fast(a, p, c, pos, pos_e, ch, chv, idx):
    """
    Where as rotate_dihedral(...) rotates all amino acids in the batch by some angle,
    this function just rotates a single amino acid in a protein.
    """
    pos = pos.copy()
    ai, pi, ci, pos_i, pos_ei, chi, chvi = (
        a[idx - 1 : idx + 1],
        p[idx - 1 : idx + 1],
        c[idx - 1 : idx + 1],
        pos[idx - 1 : idx + 1],
        pos_e[idx - 1 : idx + 1],
        ch[idx - 1 : idx + 1],
        chv[idx - 1 : idx + 1],
    )
    pnew = rotate_dihedral(ai, pi, ci, pos_i, pos_ei, chi, chvi)
    pos[idx - 1 : idx + 1] = pnew

    return pos


def rotate_dihedral(angles, par, child, pos, pos_exist, chis, chi_valid):
    """Rotate a protein representation by a set of dihedral angles:
        N represents the number of amino acids in the batch, 20 is the number of atoms.
        angles: N x 20 set of angles to rotate each atom by
        par: A N x 20 encoding of the relative offset of the parent of each atom. For example,
            the amino acid glycine would be represented at [-18 -1 -1 -1 0, ...]
        child: A N x 20 encoding of the child of each atom. For example, the amino acid glycine
            would be represented as [1 1 18 0 0 0 ..]
        pos_exist: A N x 20 mask encoding of which atoms are valid for each amino acid so for
            example the amino acid glycine would be represented as [1 1 1 1 0 0 ...]
        chis: A N x 20 representation of the existing chi angles
        chi_valid: A N x 5 mask encoding which chi angles are valid, so for example glycine would
        be represented as [0 0 0 0 0]
        """

    angles = angles / 180 * np.pi
    chis = chis / 180 * np.pi
    pos_orig = pos
    pos = pos.copy()

    for i in range(4):
        # There are a maximum of 5 chi angles
        p2 = pos[:, 4 + i]
        index = np.tile(4 + i, (pos.shape[0], 1)) + par[:, 4 + i : 5 + i]
        # print("index, pos shape ", index.shape, pos.shape)
        p1 = np.take_along_axis(pos, index[:, :, None], axis=1)[:, 0, :]

        rot_angle = chis[:, i] - angles[:, 4 + i]

        diff_vec = p2 - p1
        diff_vec_normalize = diff_vec / (np.linalg.norm(diff_vec, axis=1, keepdims=True) + 1e-10)

        # Rotate all subsequent points by the rotamor angle with the defined line where normalize on the origin
        rot_points = pos[:, 5 + i :].copy() - p1[:, None, :]

        par_points = (rot_points * diff_vec_normalize[:, None, :]).sum(
            axis=2, keepdims=True
        ) * diff_vec_normalize[:, None, :]
        perp_points = rot_points - par_points

        perp_points_norm = np.linalg.norm(perp_points, axis=2, keepdims=True) + 1e-10
        perp_points_normalize = perp_points / perp_points_norm

        a3 = np.cross(diff_vec_normalize[:, None, :], perp_points_normalize)

        rot_points = (
            perp_points * np.cos(rot_angle)[:, None, None]
            + np.sin(rot_angle)[:, None, None] * a3 * perp_points_norm
            + par_points
            + p1[:, None, :]
        )

        rot_points[np.isnan(rot_points)] = 10000

        # Only set the points that vald chi angles
        first_term = rot_points * chi_valid[:, i : i + 1, None]
        second_term = pos[:, 5 + i :] * (1 - chi_valid[:, i : i + 1, None])

        pos[:, 5 + i :] = first_term + second_term

    return pos


def compute_dihedral(par, child, pos, pos_exist, reshape=True):
    """Compute the dihedral angles of all atoms in a structure
        par: A N x 20 encoding of the relative offset of the parent of each atom. For example,
             the amino acid glycine would be represented at [-18 -1 -1 -1 0, ...]
        child: A N x 20 encoding of the child of each atom. For example, the amino acid glycine
                would be represented as [1 1 18 0 0 0 ..]
        pos_exist: A N x 20 mask encoding of which atoms are valid for each amino acid so for
        pos: A N x 20 x 3 encoding the (x, y, z) coordinates of each atom per amino acid in a protein
    """
    par, child, pos, pos_exist = flatten(par), flatten(child), flatten(pos), flatten(pos_exist)
    # pos[~pos_exist] = 0.1
    idx = np.arange(par.shape[0])

    child_idx = idx + child
    child_pos = pos[child_idx, :].copy()
    up_edge_idx = idx + par
    up_edge_pos = pos[up_edge_idx, :].copy()
    parent_idx = up_edge_idx + par[up_edge_idx]
    parent_pos = pos[parent_idx, :].copy()

    # The dihedral angle is given by parent_pos -> up_edge_pos -> pos -> child_pos
    p0, p1, p2, p3 = parent_pos, up_edge_pos, pos, child_pos

    p23 = p3 - p2
    p12 = p2 - p1
    p01 = p1 - p0

    n1 = np.cross(p01, p12)
    n2 = np.cross(p12, p23)

    n1 = n1 / (np.linalg.norm(n1, axis=1, keepdims=True) + 1e-10)
    n2 = n2 / (np.linalg.norm(n2, axis=1, keepdims=True) + 1e-10)

    sin = (np.cross(n1, n2) * p12 / (np.linalg.norm(p12, axis=1, keepdims=True) + 1e-10)).sum(
        axis=1
    )
    cos = (n1 * n2).sum(axis=1)

    angle = np.arctan2(sin, cos)

    # Convert the angles to -180 / 180
    angle = angle / np.pi * 180

    if reshape:
        angle = angle.reshape((-1, 20))

    return angle


# The code below does sampling from the dunbrack library
def sample_df(df, uniform=False, sample=1):
    """Sample from rotamer library based off gaussian on nearby slots"""
    cum_prob = df["Probabil"].cumsum()
    cutoff = np.random.uniform(0, cum_prob.max(), (sample,))
    ixs = cum_prob.searchsorted(cutoff)

    if uniform:
        ix = cum_prob.searchsorted(0.99)
        if ix == 0:
            ixs = [0] * sample
        else:
            ixs = np.random.randint(ix, size=(sample,))

    chis_list = []
    for ix in ixs:
        means = df[["chi{}Val".format(i) for i in range(1, 5)]][ix : ix + 1].to_numpy()
        std = df[["chi{}Sig".format(i) for i in range(1, 5)]][ix : ix + 1].to_numpy()
        chis = std[0] * np.random.normal(0, 1, (4,)) + means[0]
        chis[chis > 180] = chis[chis > 180] - 360
        chis[chis < -180] = chis[chis < -180] + 360
        chis_list.append(chis)

    if sample == 1:
        chis_list = chis_list[0]

    return chis_list


def sample_weighted_df(dfs, weights_array, uniform=False):
    """sample from rotamer library based off a weighted gaussian from nearby slots"""
    n = min(df["Probabil"].to_numpy().shape[0] for df in dfs)
    dfs = [df[:n].sort_values("chi1Val") for df in dfs]

    probs = []
    for weight, df in zip(weights_array, dfs):
        probs.append(df["Probabil"].to_numpy()[:n] * weight)

    probs = np.sum(np.array(probs), axis=0) / 100
    cum_prob = np.cumsum(probs)
    cutoff = np.random.uniform(0, cum_prob.max())
    ix = np.searchsorted(cum_prob, cutoff)

    if uniform:
        # ix = np.searchsorted(cum_prob, 0.99)
        if ix == 0:
            ix = 0
        else:
            ix = np.random.randint(ix)

    means = [
        weight * df[["chi{}Val".format(i) for i in range(1, 5)]].to_numpy()[:n]
        for weight, df in zip(weights_array, dfs)
    ]
    std = [
        weight * df[["chi{}Sig".format(i) for i in range(1, 5)]].to_numpy()[:n]
        for weight, df in zip(weights_array, dfs)
    ]

    means = np.sum(means, axis=0) / 100.0
    std = np.sum(std, axis=0) / 100

    chis = std[ix] * np.random.normal(0, 1, (4,)) + means[ix]

    # chis = (360 - chis) % 360

    chis[chis > 180] = chis[chis > 180] - 360
    chis[chis < -180] = chis[chis < -180] + 360
    return chis


def discrete_angle_to_bucket(ang):
    assert isinstance(ang, int)
    assert ang % 10 == 0
    assert -180 <= ang < 180
    return (ang + 180) // 10


def get_rotind(r1, r2, r3, r4):
    return 1000000 * r1 + 10000 * r2 + 100 * r3 + r4


QuadrantData = collections.namedtuple(
    "QuadrantData",
    ["chimeans", "chisigmas", "probs", "meanprobs", "cumprobs", "exists", "rotinds"],
)


def _preprocess_db(db, name):
    df = db[name]

    bucketed_data = [[{} for _1 in range(36)] for _2 in range(36)]

    df_rows = df.to_dict("records")
    for row in df_rows:
        phi, psi = row["Phi"], row["Psi"]

        wraparound = False
        if phi == 180:
            wraparound = True
            phi = -180
        if psi == 180:
            wraparound = True
            psi = -180

        phi_bucket, psi_bucket = discrete_angle_to_bucket(phi), discrete_angle_to_bucket(psi)

        rotind = get_rotind(row["r1"], row["r2"], row["r3"], row["r4"])
        chimeans = np.array([row[f"chi{i}Val"] for i in range(1, 5)])
        chisigmas = np.array([row[f"chi{i}Sig"] for i in range(1, 5)])
        prob = row["Probabil"]

        bucket = bucketed_data[phi_bucket][psi_bucket]
        bucket_data = (chimeans, chisigmas, prob)

        if wraparound:
            assert (
                (bucket[rotind][0] == bucket_data[0]).all()
                and (bucket[rotind][1] == bucket_data[1]).all()
                and (bucket[rotind][2] == bucket_data[2])
            )

        else:
            bucket[rotind] = bucket_data

    quadrant_data = [[None for _1 in range(36)] for _2 in range(36)]

    for lower_phi_bucket in range(36):
        for lower_psi_bucket in range(36):
            upper_phi_bucket = (lower_phi_bucket + 1) % 36
            upper_psi_bucket = (lower_psi_bucket + 1) % 36

            quadrants = [
                bucketed_data[lower_phi_bucket][lower_psi_bucket],
                bucketed_data[upper_phi_bucket][lower_psi_bucket],
                bucketed_data[lower_phi_bucket][upper_psi_bucket],
                bucketed_data[upper_phi_bucket][upper_psi_bucket],
            ]

            rotinds = np.array(
                sorted(set().union(*[set(quadrant.keys()) for quadrant in quadrants])),
                dtype=np.int,
            )

            assert len(rotinds) > 0

            exists = np.zeros((len(rotinds), 4), dtype=np.bool)
            probs = np.zeros((len(rotinds), 4), dtype=np.float64)
            chimeans = np.zeros((len(rotinds), 4, 4), dtype=np.float64)
            chisigmas = np.zeros((len(rotinds), 4, 4), dtype=np.float64)

            for i, rotind in enumerate(rotinds):
                for qid, quadrant in enumerate(quadrants):
                    if rotind not in quadrant:
                        continue

                    quadrant_chimeans, quadrant_chisigmas, quadrant_prob = quadrant[rotind]

                    exists[i, qid] = True
                    probs[i, qid] = quadrant_prob
                    chimeans[i, qid] = quadrant_chimeans
                    chisigmas[i, qid] = quadrant_chisigmas

            meanprobs = probs.mean(1)
            order = np.argsort(-meanprobs, kind="stable")
            meanprobs = meanprobs[order]
            cumprobs = np.cumsum(meanprobs)

            assert np.abs(cumprobs[-1] - 1) < 1e-5

            quadrant_data[lower_phi_bucket][lower_psi_bucket] = QuadrantData(
                chimeans=chimeans[order],
                chisigmas=chisigmas[order],
                probs=probs[order],
                exists=exists[order],
                rotinds=rotinds[order],
                meanprobs=meanprobs,
                cumprobs=cumprobs,
            )

    return quadrant_data


_PREPROCESS_DB_CACHE = {}


def preprocess_db(db, name):
    key = (id(db), name)
    val = _PREPROCESS_DB_CACHE.get(key)

    if val is None:
        val = _preprocess_db(db, name)
        _PREPROCESS_DB_CACHE[key] = val

    return val


def get_quadrant_data_with_interpolated_weights(db, name, phi, psi):
    lower_phi, lower_psi = int(phi // 10) * 10, int(psi // 10) * 10
    upper_phi, upper_psi = lower_phi + 10, lower_psi + 10
    lower_phi_bucket, lower_psi_bucket = (
        discrete_angle_to_bucket(lower_phi),
        discrete_angle_to_bucket(lower_psi),
    )

    quadrant_data = preprocess_db(db, name)[lower_phi_bucket][lower_psi_bucket]

    weights = np.array(
        [
            (10 - (phi - lower_phi)) * (10 - (psi - lower_psi)),
            (10 - (upper_phi - phi)) * (10 - (psi - lower_psi)),
            (10 - (phi - lower_phi)) * (10 - (upper_psi - psi)),
            (10 - (upper_phi - phi)) * (10 - (upper_psi - psi)),
        ]
    )

    sum_existing_weights = (weights[np.newaxis, :] * quadrant_data.exists).sum(1)
    effective_weights = weights[np.newaxis, :] / sum_existing_weights[:, np.newaxis]

    return quadrant_data, effective_weights


def exhaustive_sample(db, phi, psi, name, tresh=0.99, chi_mean=False):
    """sample  a set of discrete possibilitys for rotamers following protocol used in Rosetta"""
    quadrant_data, weights = get_quadrant_data_with_interpolated_weights(db, name, phi, psi)

    chimeans = (quadrant_data.chimeans * weights[:, :, np.newaxis]).sum(1)
    chisigmas = (quadrant_data.chisigmas * weights[:, :, np.newaxis]).sum(1)

    cumprobs = quadrant_data.cumprobs
    search_limit = (np.searchsorted(cumprobs, tresh) + 1) if tresh < (1 - 1e-6) else len(cumprobs)
    assert search_limit <= len(cumprobs)

    chimeans = chimeans[:search_limit]
    chisigmas = chisigmas[:search_limit]

    sigma_masks = np.array(list(product([-1, 0, 1], [-1, 0, 1], [0], [0])), dtype=np.float64)

    if chi_mean:
        return list(chimeans)

    angles = chimeans[:, np.newaxis, :] + (
        chisigmas[:, np.newaxis, :] * sigma_masks[np.newaxis, :, :]
    )
    angles = angles.reshape(-1, 4)

    for _ in range(2):
        angles[angles >= 180] = angles[angles >= 180] - 360
        angles[angles < -180] = angles[angles < -180] + 360

    return list(angles)


def _sample_from_cumprobs(cumprobs, n, uniform):
    if uniform:
        return np.random.randint(len(cumprobs), size=n)
    else:
        searchvals = np.random.uniform(low=0.0, high=cumprobs[-1], size=n)
        indices = np.searchsorted(cumprobs, searchvals)
        assert (indices < len(cumprobs)).all()
        return indices


def interpolated_sample_normal(db, phi, psi, name, n, uniform=False):
    quadrant_data, weights = get_quadrant_data_with_interpolated_weights(db, name, phi, psi)

    chimeans = (quadrant_data.chimeans * weights[:, :, np.newaxis]).sum(1)
    chisigmas = (quadrant_data.chisigmas * weights[:, :, np.newaxis]).sum(1)

    cumprobs = quadrant_data.cumprobs
    sample_indices = _sample_from_cumprobs(cumprobs=cumprobs, n=n, uniform=uniform)
    assert sample_indices.shape == (n,)

    chimeans = chimeans[sample_indices]
    chisigmas = chisigmas[sample_indices]

    angles = chimeans + np.random.randn(n, 4) * chisigmas

    for _ in range(2):
        angles[angles >= 180] = angles[angles >= 180] - 360
        angles[angles < -180] = angles[angles < -180] + 360

    return list(angles)


def mixture_sample_normal(db, phi, psi, name, n, uniform=False):
    quadrant_data, weights = get_quadrant_data_with_interpolated_weights(db, name, phi, psi)

    chimeans = quadrant_data.chimeans
    chisigmas = quadrant_data.chisigmas

    cumprobs = quadrant_data.cumprobs
    sample_indices = _sample_from_cumprobs(cumprobs=cumprobs, n=n, uniform=uniform)
    assert sample_indices.shape == (n,)

    angles = np.zeros((n, 4))

    for aidx in range(n):
        i = sample_indices[aidx]
        quadrant = np.random.choice(4, p=weights[i])
        chimean = chimeans[i, quadrant]
        chisigma = chisigmas[i, quadrant]
        angles[aidx] = chimean + np.random.randn(4) * chisigma

    for _ in range(2):
        angles[angles >= 180] = angles[angles >= 180] - 360
        angles[angles < -180] = angles[angles < -180] + 360

    return list(angles)


def sample_rotomor_angle_db(db, phi, psi, name, uniform=False, n=1):
    df = db[name]
    lower_phi = (phi // 10) * 10
    upper_phi = lower_phi + 10

    lower_psi = (psi // 10) * 10
    upper_psi = lower_psi + 10

    weights = [
        (10 - (phi - lower_phi)) * (10 - (psi - lower_psi)),
        (10 - (upper_phi - phi)) * (10 - (psi - lower_psi)),
        (10 - (phi - lower_phi)) * (10 - (upper_psi - psi)),
        (10 - (upper_phi - phi)) * (10 - (upper_psi - psi)),
    ]
    weights_array = weights

    weights = np.cumsum(weights)

    dfs = [
        df[(df.Phi == lower_phi) & (df.Psi == lower_psi)],
        df[(df.Phi == upper_phi) & (df.Psi == lower_psi)],
        df[(df.Phi == lower_phi) & (df.Psi == upper_psi)],
        df[(df.Phi == upper_phi) & (df.Psi == upper_psi)],
    ]

    calc = np.random.uniform(0, 100, (n,))

    if n == 1:
        idxs = np.searchsorted(weights, calc)
        chis = sample_df(dfs[idxs[0]], uniform=uniform)
        return chis
    else:
        idxs = np.searchsorted(weights, calc)
        chis = []

        for i in range(4):
            count = (idxs == i).sum()
            if count > 0:
                chi = sample_df(dfs[i], uniform=uniform, sample=count)

                if count > 1:
                    chis.extend(chi)
                else:
                    chis.append(chi)

        return chis


def load_rotamor_library():
    # Loads the rotamor library
    amino_acids = [
        "arg",
        "asp",
        "asn",
        "cys",
        "glu",
        "gln",
        "his",
        "ile",
        "leu",
        "lys",
        "met",
        "phe",
        "pro",
        "ser",
        "thr",
        "trp",
        "tyr",
        "val",
    ]
    db = {}

    columns = collections.OrderedDict()
    columns["T"] = np.str
    columns["Phi"] = np.int64
    columns["Psi"] = np.int64
    columns["Count"] = np.int64
    columns["r1"] = np.int64
    columns["r2"] = np.int64
    columns["r3"] = np.int64
    columns["r4"] = np.int64
    columns["Probabil"] = np.float64
    columns["chi1Val"] = np.float64
    columns["chi2Val"] = np.float64
    columns["chi3Val"] = np.float64
    columns["chi4Val"] = np.float64
    columns["chi1Sig"] = np.float64
    columns["chi2Sig"] = np.float64
    columns["chi3Sig"] = np.float64
    columns["chi4Sig"] = np.float64

    for amino_acid in amino_acids:
        db[amino_acid] = pd.read_csv(
            osp.join(ROTAMER_LIBRARY_PATH, f"ExtendedOpt1-5/{amino_acid}.bbdep.rotamers.lib"),
            names=list(columns.keys()),
            dtype=columns,
            comment="#",
            delim_whitespace=True,
            engine="c",
        )

    return db


def compute_rotamer_score_planar(gt_chi, neg_chi, chi_valid, res_name):
    select_res = {"phe": 1, "tyr": 1, "asp": 1, "glu": 2}

    if res_name in select_res.keys():
        n = select_res[res_name]
        chi_val = (
            np.minimum(
                np.minimum(
                    np.abs(neg_chi[:n] - gt_chi[:n]), np.abs(neg_chi[:n] - gt_chi[:n] - 360)
                ),
                np.abs(neg_chi[:n] - gt_chi[:n] + 360),
            )
        ) * chi_valid[:n]
        chi_bool_i = chi_val < 20

        c1, c2 = neg_chi[n], gt_chi[n]
        c1, c2 = c1 % 180, c2 % 180
        min_dist = min(min(abs(c1 - c2), abs(c1 - c2 - 180)), abs(c1 - c2 + 180)) * chi_valid[n]
        chi_bool_last = min_dist < 20

        max_dist = chi_val + [min_dist]
        chi_bool = chi_bool_i.all() & chi_bool_last
    else:
        angle_diff = np.minimum(
            np.minimum(np.abs(neg_chi - gt_chi), np.abs(neg_chi - gt_chi - 360)),
            np.abs(neg_chi - gt_chi + 360),
        )

        angle_diff[np.isnan(angle_diff)] = 0
        chi_bool = (angle_diff * chi_valid) < 20
        max_dist = angle_diff * chi_valid

    return chi_bool.all(), max_dist


def compute_dihedral_test():
    pos = np.array([[-1, 0, 0], [0, 0, 0], [0, 0, 1], [0, 1, 1]])[:, None, :]
    par = np.array([0, -1, -1, -1])
    child = np.array([1, 1, 1, 0])
    pos_exist = np.zeros(3)
    angle = compute_dihedral(par, child, pos, pos_exist, reshape=False)
    assert angle[2] == -90


def compute_dihedral_test_2():
    # Load a random CIF file
    _, dense_embed = cif_to_embed("tests/101m.cif.gz")
    dense_embed = dense_embed[0][0]
    par, child, pos, pos_exist, res, chis_valid = parse_dense_format(dense_embed)

    angles = compute_dihedral(par, child, pos, pos_exist, reshape=True)
    perturb_angles = np.random.uniform(-180, 180, (angles.shape[0], 5))

    pos_new = rotate_dihedral(angles, par, child, pos, pos_exist, perturb_angles, chis_valid)
    new_angles = compute_dihedral(par, child, pos_new, pos_exist, reshape=True)

    chi_vals = new_angles[:, 4:9]
    error = (chi_vals - perturb_angles) * chis_valid[:, 4:9]
    zeros = np.zeros_like(error)
    assert np.isclose(error, zeros, rtol=1e-6).all()


if __name__ == "__main__":
    print("Conducting unit tests on an example pdb file...")

    # Unit test for computing dihedral angles for simple perpendicular angles
    compute_dihedral_test()

    # Test compute dihedral angles via rotations from rotate_dihedral_angle_fast
    compute_dihedral_test_2()

    print("Unit tests passed.")

    # Preprocess dataset
    print("Preprocessing MMCIF files to construct the dataset...")
    script_parse_cif()
    print("Operation completed.")
