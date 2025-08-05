import torch
from torch import nn
import copy
import os
import glob
from datetime import datetime


def prune_model(model: nn.Module, threshold: float) -> nn.Module:
    """
    Prunes a SINDy model by removing basis functions with coefficients below a threshold.

    Args:
        model (nn.Module): SINDy model with `coef` and `library` attributes.
        threshold (float): Minimum absolute value to keep a coefficient.

    Returns:
        nn.Module: The pruned model with reduced basis.
    """
    with torch.no_grad():
        coef = model.coef.detach()
        active_mask = torch.any(torch.abs(coef) > threshold, dim=1)
        active_idx = torch.nonzero(active_mask, as_tuple=False).view(-1)

        if active_idx.numel() == 0:
            print("⚠️ Warning: All terms were pruned.")

        idx_list = active_idx.tolist()
        model.library.library = [model.library.library[i] for i in idx_list]
        model.library.function_names = [model.library.function_names[i] for i in idx_list]
        model.library.shape = (len(idx_list), model.library.shape[1])

        pruned_coef = coef[active_idx]
        model.coef = torch.nn.Parameter(pruned_coef.clone(), requires_grad=True)
    return model


def map_to_new_fx(trained_fx: nn.Module, base_fx: nn.Module) -> nn.Module:
    """
    Maps a trained SINDy model's pruned coefficients into a full library model.

    Args:
        trained_fx (nn.Module): Trained SINDy model with pruned terms.
        base_fx (nn.Module): Full library SINDy model to receive weights.

    Returns:
        nn.Module: Copy of `base_fx` with mapped coefficients.
    """

    def normalize_expr(expr: str) -> str:
        expr = expr.replace(" ", "")
        if "*" in expr:
            terms = expr.split("*")
            return " * ".join(sorted(terms))
        return expr

    mapped_fx = copy.deepcopy(base_fx)

    base_names_raw = mapped_fx.library.function_names
    trained_names_raw = trained_fx.library.function_names

    base_names = [normalize_expr(name) for name in base_names_raw]
    trained_names = [normalize_expr(name) for name in trained_names_raw]

    coef_full = torch.zeros((len(base_names), trained_fx.coef.shape[1]),
                            device=trained_fx.coef.device)

    for i, t_name in enumerate(trained_names):
        if t_name in base_names:
            j = base_names.index(t_name)
            coef_full[j] = trained_fx.coef[i]
        else:
            print(f"⚠️ Warning: term '{trained_names_raw[i]}' not found in full library")

    mapped_fx.coef = torch.nn.Parameter(coef_full, requires_grad=False)
    return mapped_fx


def save_model_with_timestamp(model: nn.Module, base_name: str, directory: str) -> str:
    """
    Save model state_dict to a timestamped file.

    Args:
        model (nn.Module): Model to save.
        base_name (str): Prefix name for file.
        directory (str): Output directory.

    Returns:
        str: Full path to saved file.
    """
    os.makedirs(directory, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_name}_{timestamp}.pth"
    path = os.path.join(directory, filename)
    torch.save(model.state_dict(), path)
    print(f"✅ Model saved to {path}")
    return path


def load_latest_model(model: nn.Module, base_name: str, directory: str) -> nn.Module:
    """
    Load the most recently saved model with a given base name.

    Args:
        model (nn.Module): Empty model instance to load weights into.
        base_name (str): Prefix of saved files.
        directory (str): Directory to search.

    Returns:
        nn.Module: Model with loaded weights.
    """
    pattern = os.path.join(directory, f"{base_name}_*.pth")
    model_files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)

    if not model_files:
        raise FileNotFoundError(f"No saved model with base name '{base_name}' found in '{directory}'.")

    latest_file = model_files[0]
    model.load_state_dict(torch.load(latest_file))
    print(f"✅ Loaded model from: {latest_file}")
    return model
