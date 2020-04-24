# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np


def rotate_v1_v2(v1, v2):
    # Compute rotation matrix to rotate v1 to v2

    # First compute unit vectors in the two directions
    v1 = v1 / (np.linalg.norm(v1) + 1e-10)
    v2 = v2 / (np.linalg.norm(v2) + 1e-10)

    v = np.cross(v1, v2)
    skew_matrix = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    s = np.linalg.norm(v)
    cos = (v1 * v2).sum()

    rot_matrix = (
        np.eye(3) + skew_matrix + np.matmul(skew_matrix, skew_matrix) * (1 - cos) / (s ** 2)
    )
    return rot_matrix


def rotate_v1_v2_vec(v1, v2):
    v1 = v1 / (np.linalg.norm(v1, axis=1, keepdims=True) + 1e-10)
    v2 = v2 / (np.linalg.norm(v2, axis=1, keepdims=True) + 1e-10)
    v = np.cross(v1, v2)
    n = v1.shape[0]
    z = np.zeros(n)

    skew_matrix = np.array(
        [[z, -v[:, 2], v[:, 1]], [v[:, 2], z, -v[:, 0]], [-v[:, 1], v[:, 0], z]]
    )
    skew_matrix = skew_matrix.transpose((2, 0, 1))
    s = np.linalg.norm(v, keepdims=True, axis=1)
    c = (v1 * v2).sum(axis=1, keepdims=True)

    rot_matrix = (
        np.eye(3)[None, :, :]
        + skew_matrix
        + np.matmul(skew_matrix, skew_matrix) * (1 - c[:, :, None]) / (s[:, :, None] ** 2)
    )

    return rot_matrix


if __name__ == "__main__":
    # Quick tests for functionality of math_utils
    v1 = np.array([0.5, 0.5, 0])
    v2 = np.array([0, 1, 0])
    rotate_matrix = rotate_v1_v2(v1, v2)

    print(rotate_matrix)
    print(np.matmul(rotate_matrix, v1[:, None]))

    v1 = np.tile(np.array([0.5, 0.5, 0])[None, :], (3, 1))
    v2 = np.tile(np.array([0, 1, 0]), (3, 1))
    rotate_matrix = rotate_v1_v2_vec(v1, v2)

    print(rotate_matrix.shape)
    print(rotate_matrix[0])
    print(np.matmul(rotate_matrix, v1[:, :, None]))
