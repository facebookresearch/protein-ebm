# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import os.path as osp
import pickle
import random

import numpy as np
from scipy.stats import special_ortho_group

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from config import MMCIF_PATH
from data import MMCIFTransformer, collate_fn_transformer, collate_fn_transformer_test
from easydict import EasyDict
from mmcif_utils import compute_rotamer_score_planar
from models import RotomerFCModel, RotomerGraphModel, RotomerSet2SetModel, RotomerTransformerModel
from tensorboard import TensorBoardOutputFormat
from tensorflow.python.platform import flags
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from utils import init_distributed_mode


def add_args(parser):
    #############################
    ##### Training hyperparameters
    #############################

    parser.add_argument(
        "--logdir",
        default="cachedir",
        type=str,
        help="location where log of experiments will be stored",
    )
    parser.add_argument("--exp", default="default", type=str, help="name of experiment")
    parser.add_argument("--resume-iter", default=0, type=int, help="resume value")
    parser.add_argument("--n-epochs", default=10000, type=int, help="number of epochs of training")
    parser.add_argument(
        "--batch-size", default=128, type=int, help="batch size to use during training"
    )
    parser.add_argument("--log-interval", default=50, type=int, help="interval to log results")
    parser.add_argument("--save-interval", default=1000, type=int, help="interval to log results")
    parser.add_argument(
        "--no-train",
        default=False,
        action="store_true",
        help="Instead of training, only test the model",
    )
    parser.add_argument(
        "--no-cuda", default=False, action="store_true", help="do not use GPUs for computations"
    )
    parser.add_argument(
        "--model",
        default="transformer",
        type=str,
        help="model to use during training. options: transformer, fc, s2s (set 2 set) "
        "transformer: transformer model"
        "fc: MLP baseline used in paper"
        "s2s: Set2Set baseline used in paper"
        "graph: GNN baseline used in paper",
    )

    #############################
    ##### Dataset hyperparameters
    #############################

    parser.add_argument("--data-workers", default=4, type=int, help="number of dataloader workers")
    parser.add_argument(
        "--multisample",
        default=16,
        type=int,
        help="number of different rotamers to select" "from an individual protein",
    )

    #############################
    ##### Distributed Training
    #############################

    parser.add_argument("--nodes", default=1, type=int, help="number of nodes for training")

    parser.add_argument("--gpus", default=1, type=int, help="number of gpus per node")
    parser.add_argument("--node-rank", default=0, type=int, help="rank of node")
    parser.add_argument(
        "--master-addr", default="8.8.8.8", type=str, help="address of communicating server"
    )
    parser.add_argument("--port", default=10002, type=int, help="port for communicating server")
    parser.add_argument(
        "--slurm", default=False, action="store_true", help="run experiments on SLURM?"
    )

    #############################
    ##### Transformer hyperparameters
    #############################

    parser.add_argument(
        "--encoder-layers", default=6, type=int, help="number of transformer layers"
    )
    parser.add_argument(
        "--dropout", default=0.0, type=float, help="dropout of attention weights in transformer"
    )
    parser.add_argument(
        "--relu-dropout", default=0.0, type=float, help="chance of dropping out a relu unit"
    )
    parser.add_argument(
        "--no-encoder-normalize-before",
        action="store_true",
        default=False,
        help="do not normalize outputs before the encoder (transformer only)",
    )
    parser.add_argument(
        "--encoder-attention-heads",
        default=8,
        type=int,
        help="number of attention heads (transformer only)",
    )
    parser.add_argument(
        "--attention-dropout", default=0.0, type=float, help="dropout probability for attention"
    )
    parser.add_argument(
        "--encoder-ffn-embed-dim",
        default=1024,
        type=int,
        help="hidden dimension to use in transformer",
    )
    parser.add_argument(
        "--encoder-embed-dim", default=256, type=int, help="original embed dim of element"
    )
    parser.add_argument(
        "--max-size",
        default=64,
        type=str,
        help="number of nearby atoms to attend" "when predicting energy of rotamer",
    )

    #############################
    ##### EBM hyperparameters
    #############################

    parser.add_argument(
        "--neg-sample",
        default=1,
        type=int,
        help="number of negative rotamer samples" " per real data sample (1-1 ratio)",
    )
    parser.add_argument("--l2-norm", default=False, action="store_true", help="norm the energies")
    parser.add_argument(
        "--no-augment",
        default=False,
        action="store_true",
        help="do not augment training data with so3 rotations",
    )

    #############################
    ##### generic model params
    #############################

    parser.add_argument(
        "--start-lr", default=1e-10, type=float, help="initial warmup learning rate"
    )
    parser.add_argument("--end-lr", default=2e-4, type=float, help="end lr of training")
    parser.add_argument(
        "--lr-schedule",
        default="constant",
        type=str,
        help="schedule to anneal the learning rate of transformer."
        " options: constant, inverse_sqrt",
    )
    parser.add_argument("--warmup-itr", default=500, type=int, help="number of warm up iterations")
    parser.add_argument(
        "--single",
        default=False,
        action="store_true",
        help="overfit to a single protein in dataset" "(sanity check on architecture)",
    )
    parser.add_argument(
        "--grad_accumulation", default=1, type=int, help="number of gradient accumulation steps"
    )

    #############################
    ##### Negative sampling
    #############################

    parser.add_argument(
        "--uniform",
        default=False,
        action="store_true",
        help="uniform over all candidate bins in Dunbrak library"
        "as oposed to weighted based off empericial frequency",
    )
    parser.add_argument(
        "--weighted-gauss",
        default=False,
        action="store_true",
        help="given chi and psi angles, iterpolate between nearby bins"
        "based off Gaussian with weighted sum of means/var with weights computed by distance",
    )
    parser.add_argument(
        "--gmm",
        default=False,
        action="store_true",
        help="given chi and psi angles, interpolate between nearby bins"
        "by sampling each nearby bin based off Gaussian with weights computed by distance",
    )
    parser.add_argument(
        "--chi-mean",
        default=False,
        action="store_true",
        help="instead of sampling from Gaussians from bins in the Dunbrak library"
        "just sample the mean of the bins",
    )

    return parser


def average_gradients(model):
    """Averages gradients across workers"""
    size = float(dist.get_world_size())

    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size


def sync_model(model):
    """Sync parameters across models"""
    size = float(dist.get_world_size())

    for param in model.parameters():
        dist.broadcast(param.data, 0)


def compute_lr(it, FLAGS):
    lr = FLAGS.start_lr + min(it / FLAGS.warmup_itr, 1) * (FLAGS.end_lr - FLAGS.start_lr)

    if FLAGS.lr_schedule == "inverse_sqrt":
        if it - FLAGS.warmup_itr > 10:
            lr = lr * ((it - FLAGS.warmup_itr) / 10) ** -0.5
    return lr


def train(
    train_dataloader,
    test_dataloader,
    logger,
    model,
    optimizer,
    FLAGS,
    logdir,
    rank_idx,
    train_structures,
    checkpoint=None,
):
    it = FLAGS.resume_iter
    count_it = 0
    for i in range(FLAGS.n_epochs):

        for node_pos, node_neg in train_dataloader:
            if it % 500 == 499 and ((rank_idx == 0 and FLAGS.gpus > 1) or FLAGS.gpus == 1):
                test_accs, test_losses, test_rotamer = test(
                    test_dataloader, model, FLAGS, logdir, test=False
                )
                kvs = {}
                kvs["test_losses"] = test_losses
                kvs["test_rotamer"] = test_rotamer
                kvs["test_accs"] = test_accs

                string = "Test epoch of  {}".format(i)
                for k, v in kvs.items():
                    string += "{}: {}, ".format(k, v)
                    logger.writekvs(kvs)

                print(string)

            if FLAGS.cuda:
                node_pos = node_pos.cuda()
                node_neg = node_neg.cuda()

            lr = compute_lr(it, FLAGS)

            # update the optimizer learning rate
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr / FLAGS.grad_accumulation

            energy_pos = model.forward(node_pos)
            energy_neg = model.forward(node_neg)
            energy_neg = energy_neg.view(energy_pos.size(0), -1)

            partition_function = -torch.cat([energy_pos, energy_neg], dim=1)

            log_prob = (-energy_pos) - torch.logsumexp(partition_function, dim=1, keepdim=True)
            loss = (-log_prob).mean()
            loss.backward()

            if it % FLAGS.grad_accumulation == 0:
                if FLAGS.gpus > 1:
                    average_gradients(model)

                optimizer.step()
                optimizer.zero_grad()

            it += 1
            count_it += 1

            if it % FLAGS.log_interval == 0 and rank_idx == 0:
                loss = loss.item()
                energy_pos = energy_pos.mean().item()
                energy_neg = energy_neg.mean().item()

                kvs = {}
                kvs["loss"] = loss
                kvs["energy_pos"] = energy_pos
                kvs["energy_neg"] = energy_neg

                string = "Iteration {} with values of ".format(it)

                for k, v in kvs.items():
                    string += "{}: {}, ".format(k, v)
                    logger.writekvs(kvs)

                print(string)

            if it % FLAGS.save_interval == 0 and rank_idx == 0:
                model_path = osp.join(logdir, "model_{}".format(it))
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "FLAGS": FLAGS,
                    },
                    model_path,
                )
                print("Saving model in directory....")


def test(test_dataloader, model, FLAGS, logdir, test=False):
    # Load data from a dataloader and then evaluate both the reconstruction MSE and visualize proteins
    cum_corrupt = []
    cum_decorrupt = []
    accs = []
    losses = []
    rotamer_recovery = []
    best_rotamer_recovery = []

    angle_width = np.pi / 180 * 1

    if FLAGS.train:
        neg_sample = 150
    else:
        neg_sample = FLAGS.neg_sample

    if test:
        itr = 1
    else:
        itr = 20

    counter = 0
    for _ in range(itr):
        for node_pos, node_neg, gt_chis, neg_chis, res in tqdm(test_dataloader):
            counter += 1

            if FLAGS.cuda:
                node_pos = node_pos.cuda()
                node_neg = node_neg.cuda()

            with torch.no_grad():
                energy_pos = model.forward(node_pos)
                energy_neg = model.forward(node_neg)
                energy_neg = energy_neg.view(energy_pos.size(0), neg_sample)
                partition_function = -torch.cat([energy_pos, energy_neg], dim=1)
                idx = torch.argmax(partition_function, dim=1)
                sort_idx = torch.argsort(partition_function, dim=1, descending=True)

                log_prob = (-energy_pos) - torch.logsumexp(partition_function, dim=1, keepdim=True)
                loss = (-log_prob).mean()

            # If the minimum idx is 0 then the ground truth configuration has the lowest energy
            acc = idx.eq(0).float().mean()
            accs.append(acc.item())
            losses.append(loss.item())

            node_pos, node_neg = node_pos.cpu().numpy(), node_neg.cpu().numpy()

            idx = idx.cpu().detach().numpy()
            sort_idx = sort_idx.cpu().detach().numpy()
            node_neg = np.reshape(node_neg, (-1, neg_sample, *node_neg.shape[1:]))

            for i in range(node_pos.shape[0]):
                gt_chi, chi_valid = gt_chis[i][0]

                if sort_idx[i, 0] == 0:
                    neg_chi, chi_valid = neg_chis[i][sort_idx[i, 1] - 1]

                else:
                    neg_chi, chi_valid = neg_chis[i][sort_idx[i, 0] - 1]

                neg_chi[neg_chi > 180] = neg_chi[neg_chi > 180] - 360
                score, max_dist = compute_rotamer_score_planar(gt_chi, neg_chi, chi_valid, res[i])
                rotamer_recovery.append(score.all())

                score = 0
                min_distance = float("inf")
                print_dist = None
                if not test:
                    for j, (neg_chi, chi_valid) in enumerate(neg_chis[i]):
                        temp_score, max_dist = compute_rotamer_score_planar(
                            gt_chi, neg_chi, chi_valid, res[i]
                        )
                        score = max(score, temp_score)

                        if max(max_dist) < min_distance:
                            min_distance = max(max_dist)
                            print_dist = max_dist
                else:
                    score = 1

                best_rotamer_recovery.append(score)


            if counter > 20 and (not test):
                # Return preliminary scores of rotamer recovery
                print(
                    "Mean cumulative accuracy of: ",
                    np.mean(accs),
                    np.std(accs) / np.sqrt(len(accs)),
                )
                print("Mean losses of: ", np.mean(losses), np.std(losses) / np.sqrt(len(losses)))
                print(
                    "Rotamer recovery ",
                    np.mean(rotamer_recovery),
                    np.std(rotamer_recovery) / np.sqrt(len(rotamer_recovery)),
                )
                print(
                    "Best Rotamer recovery ",
                    np.mean(best_rotamer_recovery),
                    np.std(best_rotamer_recovery) / np.sqrt(len(best_rotamer_recovery)),
                )
                break

    return np.mean(accs), np.mean(losses), np.mean(rotamer_recovery)


def main_single(gpu, FLAGS):
    if FLAGS.slurm:
        init_distributed_mode(FLAGS)

    os.environ["MASTER_ADDR"] = str(FLAGS.master_addr)
    os.environ["MASTER_PORT"] = str(FLAGS.port)

    rank_idx = FLAGS.node_rank * FLAGS.gpus + gpu
    world_size = FLAGS.nodes * FLAGS.gpus

    if rank_idx == 0:
        print("Values of args: ", FLAGS)

    if world_size > 1:
        if FLAGS.slurm:
            dist.init_process_group(
                backend="nccl", init_method="env://", world_size=world_size, rank=rank_idx
            )
        else:
            dist.init_process_group(
                backend="nccl",
                init_method="tcp://localhost:1492",
                world_size=world_size,
                rank=rank_idx,
            )

    train_dataset = MMCIFTransformer(
        FLAGS,
        split="train",
        rank_idx=rank_idx,
        world_size=world_size,
        uniform=FLAGS.uniform,
        weighted_gauss=FLAGS.weighted_gauss,
        gmm=FLAGS.gmm,
        chi_mean=FLAGS.chi_mean,
        mmcif_path=MMCIF_PATH,
    )
    valid_dataset = MMCIFTransformer(
        FLAGS,
        split="val",
        rank_idx=rank_idx,
        world_size=world_size,
        uniform=FLAGS.uniform,
        weighted_gauss=FLAGS.weighted_gauss,
        gmm=FLAGS.gmm,
        chi_mean=FLAGS.chi_mean,
        mmcif_path=MMCIF_PATH,
    )
    test_dataset = MMCIFTransformer(
        FLAGS,
        split="test",
        rank_idx=0,
        world_size=1,
        uniform=FLAGS.uniform,
        weighted_gauss=FLAGS.weighted_gauss,
        gmm=FLAGS.gmm,
        chi_mean=FLAGS.chi_mean,
        mmcif_path=MMCIF_PATH,
    )
    train_dataloader = DataLoader(
        train_dataset,
        num_workers=FLAGS.data_workers,
        collate_fn=collate_fn_transformer,
        batch_size=FLAGS.batch_size // FLAGS.multisample,
        shuffle=True,
        pin_memory=False,
        drop_last=True,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        num_workers=0,
        collate_fn=collate_fn_transformer_test,
        batch_size=FLAGS.batch_size // FLAGS.multisample,
        shuffle=True,
        pin_memory=False,
        drop_last=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        num_workers=0,
        collate_fn=collate_fn_transformer_test,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        pin_memory=False,
        drop_last=True,
    )

    train_structures = train_dataset.files

    FLAGS_OLD = FLAGS

    logdir = osp.join(FLAGS.logdir, FLAGS.exp)

    if FLAGS.resume_iter != 0:
        model_path = osp.join(logdir, "model_{}".format(FLAGS.resume_iter))
        checkpoint = torch.load(model_path)
        try:
            FLAGS = checkpoint["FLAGS"]

            # Restore arguments to saved checkpoint values except for a select few
            FLAGS.resume_iter = FLAGS_OLD.resume_iter
            FLAGS.nodes = FLAGS_OLD.nodes
            FLAGS.gpus = FLAGS_OLD.gpus
            FLAGS.node_rank = FLAGS_OLD.node_rank
            FLAGS.master_addr = FLAGS_OLD.master_addr
            FLAGS.neg_sample = FLAGS_OLD.neg_sample
            FLAGS.train = FLAGS_OLD.train
            FLAGS.multisample = FLAGS_OLD.multisample
            FLAGS.steps = FLAGS_OLD.steps
            FLAGS.step_lr = FLAGS_OLD.step_lr
            FLAGS.batch_size = FLAGS_OLD.batch_size

            for key in dir(FLAGS):
                if "__" not in key:
                    FLAGS_OLD[key] = getattr(FLAGS, key)

            FLAGS = FLAGS_OLD
        except Exception as e:
            print(e)
            print("Didn't find keys in checkpoint'")

    if FLAGS.model == "transformer":
        import pdb
        pdb.set_trace()
        model = RotomerTransformerModel(FLAGS).train()
    elif FLAGS.model == "fc":
        model = RotomerFCModel(FLAGS).train()
    elif FLAGS.model == "s2s":
        model = RotomerSet2SetModel(FLAGS).train()
    elif FLAGS.model == "graph":
        model = RotomerGraphModel(FLAGS).train()

    if FLAGS.cuda:
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)

    optimizer = optim.Adam(model.parameters(), lr=FLAGS.start_lr, betas=(0.99, 0.999))

    if FLAGS.gpus > 1:
        sync_model(model)

    logger = TensorBoardOutputFormat(logdir)

    it = FLAGS.resume_iter

    if not osp.exists(logdir):
        os.makedirs(logdir)

    checkpoint = None
    if FLAGS.resume_iter != 0:
        model_path = osp.join(logdir, "model_{}".format(FLAGS.resume_iter))
        checkpoint = torch.load(model_path)
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            model.load_state_dict(checkpoint["model_state_dict"])
        except Exception as e:
            print("Transfer between distributed to non-distributed")

            model_state_dict = {
                k.replace("module.", ""): v for k, v in checkpoint["model_state_dict"].items()
            }
            model.load_state_dict(model_state_dict)

    pytorch_total_params = sum([p.numel() for p in model.parameters() if p.requires_grad])

    if rank_idx == 0:
        print("New Values of args: ", FLAGS)
        print("Number of parameters for models", pytorch_total_params)

    if FLAGS.train:
        train(
            train_dataloader,
            valid_dataloader,
            logger,
            model,
            optimizer,
            FLAGS,
            logdir,
            rank_idx,
            train_structures,
            checkpoint=checkpoint,
        )
    else:
        test(test_dataloader, model, FLAGS, logdir)


def main():
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
    flags_dict.train = not flags_dict.no_train
    flags_dict.cuda = not flags_dict.no_cuda
    flags_dict.encoder_normalize_before = not flags_dict.no_encoder_normalize_before
    flags_dict.augment = not flags_dict.no_augment
    if not flags_dict.train:
        flags_dict.multisample = 1

    # Launch the job (optionally in a distributed manner)
    if flags_dict.gpus > 1:
        mp.spawn(main_single, nprocs=flags_dict.gpus, args=(flags_dict,))
    else:
        main_single(0, flags_dict)


if __name__ == "__main__":
    main()
