from matplotlib import pyplot as plt
from scipy import stats
from src.datasets import get_test_loader
import numpy as np
from pathlib import Path
import argparse
import scipy.stats as stats
import matplotlib.pyplot as plt
from src.datasets import get_test_loader, DATASET_CONFIGS
import src.models2 as m
import numpy as np
from src.train import test
import neptune
import random
import torch


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description="Train ETCaps Model")
#Dataset Settings
parser.add_argument(
    "--data_dir", type=str, required=True, help="Path where datasets are stored"
)
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    choices=["cifar10", "svhn", "smallnorb", "3diebench"],
    help="Dataset name",
)

# Training Settings
parser.add_argument(
    "--batch_size", type=int, default=128, help="Batch size for training"
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
parser.add_argument("--test", action="store_true", help="Train the model")

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
parser.add_argument("--et", action="store_true", help="Set this flag to add a sequence of equivariant transformers before the backbone")

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
    "--root_log_dir", type=Path, default=Path("logs"), help="Root directory for logs"
)

parser.add_argument(
 "--path_to_model", type=Path, help="path to model"

)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)    
    
def main():
    
    args = parser.parse_args()
    test_loader = get_test_loader(data_dir=args.data_dir, dataset=args.dataset, batch_size=args.batch_size)
    errors = []
    seeds = [42, 43, 44, 45, 46]
    for i in range(5):
        set_seed(seeds[i])
        error = test(args.path_to_model, test_loader, args)
        errors.append(error)

    errors = np.array(errors)
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    confidence = 0.95
    n = len(errors)
    stderr = std_error / np.sqrt(n)
    margin_of_error = stderr * stats.t.ppf((1 + confidence) / 2., n - 1)

    print(mean_error, std_error)

    plt.figure(figsize=(6, 4))
    plt.bar(1, mean_error, yerr=margin_of_error, capsize=10, color='skyblue')
    plt.xticks([1], ['Model'])
    plt.ylabel('Classification Error')
    plt.title('Classification Error with 95% Confidence Interval')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    main()
