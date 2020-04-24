# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import subprocess

import numpy as np


def init_distributed_mode(params):
    """
    Handle single and multi-GPU / multi-node / SLURM jobs.
    Initialize the following variables:
        - n_nodes
        - node_id
        - local_rank
        - global_rank
        - world_size
    """
    SLURM_VARIABLES = [
        "SLURM_JOB_ID",
        "SLURM_JOB_NODELIST",
        "SLURM_JOB_NUM_NODES",
        "SLURM_NTASKS",
        "SLURM_TASKS_PER_NODE",
        "SLURM_MEM_PER_NODE",
        "SLURM_MEM_PER_CPU",
        "SLURM_NODEID",
        "SLURM_PROCID",
        "SLURM_LOCALID",
        "SLURM_TASK_PID",
    ]

    PREFIX = "%i - " % int(os.environ["SLURM_PROCID"])
    for name in SLURM_VARIABLES:
        value = os.environ.get(name, None)
        print(PREFIX + "%s: %s" % (name, str(value)))

    # number of nodes / node ID
    params.nodes = int(os.environ["SLURM_JOB_NUM_NODES"])
    params.node_rank = int(os.environ["SLURM_NODEID"])

    # define master address and master port
    hostnames = subprocess.check_output(
        ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]]
    )
    params.master_addr = hostnames.split()[0].decode("utf-8")
    print("master address ", params.master_addr)


if __name__ == "__main__":
    # Quick unittest for collate_graph / decollate_graph
    node_embed = np.random.randn(10, 5)
    node_embed_2 = np.random.randn(10, 5)

    edges = np.array([[0, 1], [1, 2], [5, 6], [7, 8]]).transpose()
    edges_2 = np.array([[0, 5], [2, 6], [3, 7], [4, 8]]).transpose()

    nodes = [node_embed, node_embed_2]
    edges = [edges, edges_2]

    batch, edges_batch, nodes_batch = collate_graph(nodes, edges)
    new_nodes, new_edges = decollate_graph(batch, nodes_batch, edges_batch)

    print("Nodes old", nodes, "Nodes new", new_nodes)
    print("Edges old", edges, "Edges new", new_edges)
