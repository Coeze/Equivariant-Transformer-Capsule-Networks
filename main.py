# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import argparse
import os
import signal
import subprocess
import torch
from torch import nn
from src.datasets import get_train_valid_dataset
import src.models as m
from src.train import train

parser = argparse.ArgumentParser(description="Train ETCaps Model")
#Dataset Settings
parser.add_argument(
    "--data_dir", type=str, required=True, help="Path where downloaded datasets are stored"
)
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    choices=["cifar10", "svhn", "smallnorb"],
    help="Dataset name",
)

# Training Settings
parser.add_argument(
    "--batch_size", type=int, default=64, help="Batch size for training"
)
parser.add_argument(
    "--random_seed", type=int, default=42, help="Random seed for reproducibility"
)
parser.add_argument(
    "--epochs", type=int, default=200, help="Number of training epochs"
)
parser.add_argument(
    "--lr", type=float, default=2e-3, help="Learning rate"
)
parser.add_argument(
    "--wd", type=float, default=1e-6, help="Weight decay"
)
parser.add_argument(
    "--save_dir", type=str, default="models/", help="Directory to save models"
)

# Model Architecture Settings
parser.add_argument(
    "--encoder", type=str, default="resnet20", help="The resnet encoder to use"
)
parser.add_argument(
    "--num_caps", type=int, default=8, help="Number of capsules"
)
parser.add_argument(
    "--caps_size", type=int, default=4, help="Size of capsules"
)
parser.add_argument(
    "--depth", type=int, default=1, help="Depth of the model"
)
# Training Environment Settings
parser.add_argument("--num_workers", type=int, default=4, help="Number of worker threads for data loading")
parser.add_argument(
    "--world_size", type=int, default=None, 
    help="Number of GPUs to use for distributed training. If not specified, all available GPUs will be used."
)
parser.add_argument("--port", type=int, default=52472)
parser.add_argument(
    "--exp_dir", type=Path, default=Path("experiments/default"), help="Experiment directory"
)
parser.add_argument(
    "--model_name", type=str, default='etcaps', help="name of model"
)


def main():
    args = parser.parse_args()
    args.ngpus_per_node = torch.cuda.device_count()
    if "SLURM_JOB_ID" in os.environ:
        # single-node and multi-node distributed training on SLURM cluster
        # requeue job on SLURM preemption
        signal.signal(signal.SIGUSR1, handle_sigusr1)
        signal.signal(signal.SIGTERM, handle_sigterm)
        # find a common host name on all nodes
        # assume scontrol returns hosts in the same order on all nodes
        cmd = "scontrol show hostnames " + os.getenv("SLURM_JOB_NODELIST")
        stdout = subprocess.check_output(cmd.split())
        host_name = stdout.decode().splitlines()[0]
        args.rank = int(os.getenv("SLURM_NODEID")) * args.ngpus_per_node
        args.world_size = int(os.getenv("SLURM_NNODES")) * args.ngpus_per_node
        print(args.world_size)
        args.dist_url = f"tcp://{host_name}:{args.port}"
    else:
        # single-node distributed training
        args.rank = 0
        args.dist_url = f"tcp://localhost:{args.port}"
        args.world_size = args.ngpus_per_node
    torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)


def main_worker(gpu, args):
    args.rank += gpu
    torch.distributed.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    torch.cuda.set_device(gpu)
    args.device = gpu
    torch.backends.cudnn.benchmark = True

    save_path = f"{args.save_dir}/{args.model_name}"
    os.makedirs(save_path, exist_ok=True)

    os.makedirs(args.exp_dir, exist_ok=True)
    train_dataset, val_dataset = get_train_valid_dataset(data_dir=args.data_dir, dataset=args.dataset, batch_size=args.batch_size, random_seed=args.random_seed, exp='elevation')
    
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True) # Distributing the dataset across GPUs
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    print("per_device_batch_size",per_device_batch_size)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=per_device_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=sampler,
    )
    
    sampler_val = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=per_device_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=sampler_val,
    )
    
    if args.model_name == 'etcaps':
        model = m.ETCAPS(args).cuda(gpu)
    elif args.model_name == 'srcaps':
        model = m.SRCAPS(args).cuda(gpu)
    elif args.model_name == 'et':
        model = m.Transformer(args).cuda(gpu)
    elif args.model_name == 'resnet20':
        model = m.ResNet(args).cuda(gpu)
    else:
        raise ValueError('Model type not recognised, choose either: etcaps or srcaps or et')

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model) # Synchronise batch norm statistics across GPUs
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu],find_unused_parameters=True) # Synchronize gradients across GPUs
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd) 

    if args.dataset == "cifar10":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)
    elif args.dataset == "svhn":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    elif args.dataset == "smallnorb":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

    model = train(model, train_loader, val_loader, scheduler, optimizer, args)
  

def exclude_bias_and_norm(p):
    return p.ndim == 1

def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.environ["SLURM_JOB_ID"]}')
    exit()


def handle_sigterm(signum, frame):
    pass


if __name__ == "__main__":
    main()
