import torch
import pandas as pd
from .lie_derivs import *
import tqdm

def get_equivariance_metrics(model, minibatch):
    x, y = minibatch
    if torch.cuda.is_available():
        model = model.cuda()
        x, y = x.cuda(), y.cuda()

    model = model.eval()

    model_probs = lambda x: F.softmax(model(x), dim=-1)

    errs = {
        "trans_x_deriv": translation_lie_deriv(model_probs, x, axis="x"),
        "trans_y_deriv": translation_lie_deriv(model_probs, x, axis="y"),
        "rot_deriv": rotation_lie_deriv(model_probs, x),
        "shear_x_deriv": shear_lie_deriv(model_probs, x, axis="x"),
        "shear_y_deriv": shear_lie_deriv(model_probs, x, axis="y"),
        "stretch_x_deriv": stretch_lie_deriv(model_probs, x, axis="x"),
        "stretch_y_deriv": stretch_lie_deriv(model_probs, x, axis="y"),
        "saturate_err": saturate_lie_deriv(model_probs, x),
    }
    
    metrics = {x: pd.Series(errs[x].abs().cpu().data.numpy().mean(-1)) for x in errs}
    df = pd.DataFrame.from_dict(metrics)
    return df



def eval_average_metrics_wstd(loader, metrics, max_mbs=None):
    total = len(loader) if max_mbs is None else min(max_mbs, len(loader))
    dfs = []
    with torch.no_grad():
        for idx, minibatch in tqdm.tqdm(enumerate(loader), total=total):
            dfs.append(metrics(minibatch))
            if max_mbs is not None and idx >= max_mbs:
                break
    df = pd.concat(dfs)
    return df
