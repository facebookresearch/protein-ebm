# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict

# List of how to compute dihedral angles for amino acids

kvs = defaultdict(list)

kvs["ARG"].append("N-CA-CB-CG")
kvs["ASN"].append("N-CA-CB-CG")
kvs["ASP"].append("N-CA-CB-CG")
kvs["CYS"].append("N-CA-CB-SG")
kvs["GLN"].append("N-CA-CB-CG")
kvs["GLU"].append("N-CA-CB-CG")
kvs["HIS"].append("N-CA-CB-CG")
kvs["ILE"].append("N-CA-CB-CG1")
kvs["LEU"].append("N-CA-CB-CG")
kvs["LYS"].append("N-CA-CB-CG")
kvs["MET"].append("N-CA-CB-CG")
kvs["PHE"].append("N-CA-CB-CG")
kvs["PRO"].append("N-CA-CB-CG")
kvs["SER"].append("N-CA-CB-OG")
kvs["THR"].append("N-CA-CB-OG1")
kvs["TRP"].append("N-CA-CB-CG")
kvs["TYR"].append("N-CA-CB-CG")
kvs["VAL"].append("N-CA-CB-CG1")
kvs["ARG"].append("CA-CB-CG-CD")
kvs["ASN"].append("CA-CB-CG-OD1")
kvs["ASP"].append("CA-CB-CG-OD1")
kvs["GLN"].append("CA-CB-CG-CD")
kvs["GLU"].append("CA-CB-CG-CD")
kvs["HIS"].append("CA-CB-CG-ND1")
kvs["ILE"].append("CA-CB-CG1-CD")
kvs["LEU"].append("CA-CB-CG-CD1")
kvs["LYS"].append("CA-CB-CG-CD")
kvs["MET"].append("CA-CB-CG-SD")
kvs["PHE"].append("CA-CB-CG-CD1")
kvs["PRO"].append("CA-CB-CG-CD")
kvs["TRP"].append("CA-CB-CG-CD1")
kvs["TYR"].append("CA-CB-CG-CD1")
kvs["ARG"].append("CB-CG-CD-NE")
kvs["GLN"].append("CB-CG-CD-OE1")
kvs["GLU"].append("CB-CG-CD-OE1")
kvs["LYS"].append("CB-CG-CD-CE")
kvs["MET"].append("CB-CG-SD-CE")
kvs["ARG"].append("CG-CD-NE-CZ")
kvs["LYS"].append("CG-CD-CE-NZ")
kvs["ARG"].append("CD-NE-CZ-NH1")

# List Atomic Order to Amino Acid for Forward Kinematics
res_atoms = {}
res_parents = {}
res_children = {}
res_chis = {}

base_parents = [-18, -1, -1, -1]
base_children = [1, 1, 18, 0]
base_atoms = ["N", "CA", "C", "O"]

# List the configs per amino acid

# Valine
res_atoms["VAL"] = base_atoms + ["CB", "CG1", "CG2"]
res_parents["VAL"] = base_parents + [-3, -1, -2]
res_children["VAL"] = base_children + [1, 0, 0]
res_chis["VAL"] = [4]

# Alanine
res_atoms["ALA"] = base_atoms + ["CB"]
res_parents["ALA"] = base_parents + [-3]
res_children["ALA"] = base_children + [0]
res_chis["ALA"] = []

# Leucine
res_atoms["LEU"] = base_atoms + ["CB", "CG", "CD1", "CD2"]
res_parents["LEU"] = base_parents + [-3, -1, -1, -2]
res_children["LEU"] = base_children + [1, 1, 0, 0]
res_chis["LEU"] = [4, 5]

# Isoleucine
res_atoms["ILE"] = base_atoms + ["CB", "CG1", "CG2", "CD1"]
res_parents["ILE"] = base_parents + [-3, -1, -2, -1]
res_children["ILE"] = base_children + [1, 2, 0, 0]
res_chis["ILE"] = [4, 5]

# Proline
res_atoms["PRO"] = base_atoms + ["CB", "CG", "CD"]
res_parents["PRO"] = base_parents + [-3, -1, -1]
res_children["PRO"] = base_children + [1, 1, 0]
res_chis["PRO"] = [4, 5]

# Methionine
res_atoms["MET"] = base_atoms + ["CB", "CG", "SD", "CE"]
res_parents["MET"] = base_parents + [-3, -1, -1, -1]
res_children["MET"] = base_children + [1, 1, 1, 0]
res_chis["MET"] = [4, 5, 6]

# Phenylalanine
res_atoms["PHE"] = base_atoms + ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"]
res_parents["PHE"] = base_parents + [-3, -1, -1, -2, -2, -2, -2]
res_children["PHE"] = base_children + [1, 1, 2, 2, 2, 1, 0]
res_chis["PHE"] = [4, 5]

# Tryptophan
res_atoms["TRP"] = base_atoms + ["CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"]
res_parents["TRP"] = base_parents + [-3, -1, -1, -2, -2, -2, -3, -2, -2, -2]
res_children["TRP"] = base_children + [1, 1, 2, 2, 0, 2, 2, 2, 0, 0]
res_chis["TRP"] = [4, 5]

# Glycine
res_atoms["GLY"] = base_atoms
res_parents["GLY"] = base_parents
res_children["GLY"] = base_children
res_chis["GLY"] = []

# Serine
res_atoms["SER"] = base_atoms + ["CB", "OG"]
res_parents["SER"] = base_parents + [-3, -1]
res_children["SER"] = base_children + [1, 0]
res_chis["SER"] = [4]

# Threonine
res_atoms["THR"] = base_atoms + ["CB", "OG1", "CG2"]
res_parents["THR"] = base_parents + [-3, -1, -2]
res_children["THR"] = base_children + [1, 0, 0]
res_chis["THR"] = [4]

# Cystine
res_atoms["CYS"] = base_atoms + ["CB", "SG"]
res_parents["CYS"] = base_parents + [-3, -1]
res_children["CYS"] = base_children + [1, 0]
res_chis["CYS"] = [4]

# Tyrosine
res_atoms["TYR"] = base_atoms + ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"]
res_parents["TYR"] = base_parents + [-3, -1, -1, -2, -2, -2, -2, -1]
res_children["TYR"] = base_children + [1, 1, 2, 2, 2, 1, 1, 0]
res_chis["TYR"] = [4, 5]

# Asparagine
res_atoms["ASN"] = base_atoms + ["CB", "CG", "OD1", "ND2"]
res_parents["ASN"] = base_parents + [-3, -1, -1, -2]
res_children["ASN"] = base_children + [1, 1, 0, 0]
res_chis["ASN"] = [4, 5]

# Aspartic acid
res_atoms["ASP"] = base_atoms + ["CB", "CG", "OD1", "OD2"]
res_parents["ASP"] = base_parents + [-3, -1, -1, -2]
res_children["ASP"] = base_children + [1, 1, 0, 0]
res_chis["ASP"] = [4, 5]

# Glutamine
res_atoms["GLN"] = base_atoms + ["CB", "CG", "CD", "OE1", "NE2"]
res_parents["GLN"] = base_parents + [-3, -1, -1, -1, -2]
res_children["GLN"] = base_children + [1, 1, 1, 0, 0]
res_chis["GLN"] = [4, 5, 6]

# Glutamic Acid
res_atoms["GLU"] = base_atoms + ["CB", "CG", "CD", "OE1", "OE2"]
res_parents["GLU"] = base_parents + [-3, -1, -1, -1, -2]
res_children["GLU"] = base_children + [1, 1, 1, 0, 0]
res_chis["GLU"] = [4, 5, 6]

# Lysine
res_atoms["LYS"] = base_atoms + ["CB", "CG", "CD", "CE", "NZ"]
res_parents["LYS"] = base_parents + [-3, -1, -1, -1, -1]
res_children["LYS"] = base_children + [1, 1, 1, 1, 0]
res_chis["LYS"] = [4, 5, 6, 7]

# Arginine
res_atoms["ARG"] = base_atoms + ["CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"]
res_parents["ARG"] = base_parents + [-3, -1, -1, -1, -1, -1, -2]
res_children["ARG"] = base_children + [1, 1, 1, 1, 1, 1, 0]
res_chis["ARG"] = [4, 5, 6, 7, 8]

# Histidine
res_atoms["HIS"] = base_atoms + ["CB", "CG", "ND1", "CD2", "CE1", "NE2"]
res_parents["HIS"] = base_parents + [-3, -1, -1, -2, -2, -2]
res_children["HIS"] = base_children + [1, 1, 2, 2, 1, 0]
res_chis["HIS"] = [4, 5]
