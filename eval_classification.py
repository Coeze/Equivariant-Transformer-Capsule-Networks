from matplotlib import pyplot as plt
from scipy import stats
from src.datasets import get_test_loader
import numpy as np
from pathlib import Path
import argparse
import scipy.stats as stats
import matplotlib.pyplot as plt
from src.datasets import get_test_loader, DATASET_CONFIGS
import src.models as m
import numpy as np
from src.train import test
# import neptune
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
    choices=["cifar10", "svhn", "smallnorb"],
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

# Model Architecture Settings
parser.add_argument(
    "--model_name", type=str, default='etcaps', help="name of model", choices=['etcaps', 'et', 'srcaps', 'resnet20']
)
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
parser.add_argument(
 "--path_to_model", type=Path, help="path to model"

)
parser.add_argument(
    "--exp", type=str, default='full', help="experiment type", choices=['full', 'azimuth', 'elevation']
)
parser.add_argument(
    "--familiar", action='store_true', help="Use seen data in smallnorb evaluation"
)
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)    
    
def main():
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    checkpoint = torch.load(args.path_to_model, map_location=device)    
    
    if args.model_name == 'resnet20':
        model = m.ResNet(args).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
    elif args.model_name == 'srcaps':
        model = m.SRCAPS(args).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
    elif args.model_name == 'et':
        model = m.Transformer(args).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
    elif args.model_name == 'etcaps':
        model = m.ETCAPS(args).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise ValueError("Invalid model name. Choose from ['etcaps', 'et', 'srcaps', 'resnet20']")
    
    test_loader = get_test_loader(data_dir=args.data_dir, dataset=args.dataset, batch_size=args.batch_size, exp=args.exp, familiar=args.familiar)
    errors = []
    for i in range(5):
        set_seed(i+41)
        error = test(model, test_loader)
        errors.append(error)

    errors = np.array(errors)
    mean_error = np.mean(errors)

    print(f'Mean Error: {mean_error}')

    
if __name__ == "__main__":
    main()
