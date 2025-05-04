import os
import gc
from src.datasets import DATASET_CONFIGS, get_test_loader
# import wandb
import argparse
import pandas as pd
import numpy as np
from functools import partial
import torch
import src.models as m
import torchvision
from liederiv.lee.e2e_lee import get_equivariance_metrics as get_lee_metrics, eval_average_metrics_wstd

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

def compute_stats(df, metrics_columns):
    """
    Compute mean and standard deviation for each metric in the dataframe.
    
    Args:
        df: DataFrame containing metrics data
        metrics_columns: List of column names for metrics
        
    Returns:
        DataFrame with mean and std for each metric
    """
    stats_df = pd.DataFrame()
    
    # Calculate mean and std for each metric
    for metric in metrics_columns:
        stats_df[f"{metric}_mean"] = [df[metric].mean()]
        stats_df[f"{metric}_std"] = [df[metric].std()]
    
    # Add dataset and model info
    if len(df) > 0:
        stats_df["dataset"] = df["dataset"].iloc[0]
        stats_df["model"] = df["model"].iloc[0]
        stats_df["params"] = df["params"].iloc[0]
    
    return stats_df

def get_args_parser():
    parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser.add_argument('--output_dir', metavar='NAME', default='equivariance_error',help='experiment name')
    parser.add_argument('--modelname', metavar='NAME', default='etcaps', help='model name', choices=['etcaps', 'srcaps', 'resnet20', 'et'])
    parser.add_argument('--model_path', metavar='PATH', help='path to model')
    parser.add_argument('--num_datapoints', type=int, default=100, help='use pretrained model')
    parser.add_argument('--encoder', type=str, default='resnet20', help='The resnet encoder to use')
    parser.add_argument('--num_caps', type=int, default=32, help='Number of capsules')
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
    # wandb.init(project="LieDerivEquivariance", config=args)
    # args.__dict__.update(wandb.config)

    print(args)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print(args.modelname)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model_path, map_location='cpu')['model_state_dict']
    
    if args.modelname == "etcaps":
        model = m.ETCAPS(args)
    elif args.modelname == "srcaps":
        model = m.SRCAPS(args)
    elif args.modelname == "resnet20":
        model = m.ResNet(args)
    elif args.modelname == "et":
        model = m.Transformer(args)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()

    evaluated_metrics = []

    dataset = get_test_loader(args.data_dir, args.dataset, batch_size=64,num_workers=0, num_samples=args.num_datapoints, pin_memory=False)
    evaluated_metrics += [get_metrics(args, args.dataset, dataset, model, max_mbs=args.num_datapoints)]
    gc.collect()

    df = pd.concat(evaluated_metrics)
    
    # Define the metrics columns
    metrics_columns = ['trans_x_deriv', 'trans_y_deriv', 'rot_deriv', 
                      'shear_x_deriv', 'shear_y_deriv', 
                      'stretch_x_deriv', 'stretch_y_deriv', 'saturate_err']
    
    # Compute statistics for metrics
    stats_df = compute_stats(df, metrics_columns)
    
    # Save detailed metrics to CSV
    df.to_csv(os.path.join(args.output_dir, args.modelname + ".csv"))
    print('Saved metrics to', os.path.join(args.output_dir, args.modelname + ".csv"))
    
    # Save statistics to CSV
    stats_file = os.path.join(args.output_dir, args.modelname + "_stats.csv")
    stats_df.to_csv(stats_file)
    print('Saved statistics to', stats_file)
    
    # Print summary of results
    print("\nSummary of equivariance metrics:")
    print("-" * 50)
    for metric in metrics_columns:
        mean_val = df[metric].mean()
        std_val = df[metric].std()
        print(f"{metric}: {mean_val} ± {std_val}")
    print("-" * 50)
    
if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
