import os
import gc
from src.datasets import DATASET_CONFIGS, get_test_loader
import wandb
import numpy as np
import argparse
import pandas as pd
from functools import partial
import torch
import src.models as m
import torchvision
from liederiv.lee.e2e_lee import get_equivariance_metrics as get_lee_metrics
from liederiv.lee.loader import get_loaders, eval_average_metrics_wstd

def numparams(model):
    return sum(p.numel() for p in model.parameters())

def get_metrics(args, key, loader, model, max_mbs=400):
    lee_metrics = eval_average_metrics_wstd(
        loader, partial(get_lee_metrics, model), max_mbs=max_mbs,
    )
    metrics = pd.concat([lee_metrics], axis=1)

    metrics["dataset"] = key
    metrics["model"] = args.modelname
    metrics["params"] = numparams(model)

    return metrics

def get_args_parser():
    parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser.add_argument('--output_dir', metavar='NAME', default='equivariance_metrics_cnns',help='experiment name')
    parser.add_argument('--modelname', metavar='NAME', default='resnet18', help='model name')
    parser.add_argument('--num_datapoints', type=int, default=60, help='use pretrained model')
    parser.add_argument('--encoder', type=str, default='resnet20', help='The resnet encoder to use')
    parser.add_argument('--num_caps', type=int, default=8, help='Number of capsules')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='./data', help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument(
        "--caps_size", type=int, default=4, help="Size of capsules"
    )
    parser.add_argument(
        "--depth", type=int, default=1, help="Depth of the model"
    )

    return parser

def main(args):
    wandb.init(project="LieDerivEquivariance", config=args)
    args.__dict__.update(wandb.config)

    print(args)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print(args.modelname)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.modelname == "etcaps":
        model = m.ETCAPS(args)
    elif args.modelname == "srcaps":
        model = m.SRCAPS(args)
    elif args.modelname == "resnet18":
        model = torchvision.models.resnet18(pretrained=False)
    model.eval()

    evaluated_metrics = []

    dataset = get_test_loader(args.data_dir, args.dataset, batch_size=64,num_workers=0, num_samples=args.num_datapoints, pin_memory=False)
    evaluated_metrics += [get_metrics(args, args.dataset, dataset, model, max_mbs=args.num_datapoints)]
    gc.collect()


    df = pd.concat(evaluated_metrics)
    df.to_csv(os.path.join(args.output_dir, args.modelname + ".csv"))

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
