import torch
from torch import nn
import copy
import os
from datetime import datetime
import glob


def prune_model(model, threshold):
    """
    Removes functions and coefficients from the model whose absolute values are below the threshold.

    Args:
        model: The model with `fx_policy_1` and `fx_policy_2`, each having `.coef` and `.library`.
        threshold: Float value below which coefficients are pruned.
    """
    with torch.no_grad():
        coef = model.coef.detach()

        # Identify active function indices (any row with abs(coef) > threshold)
        active_mask = torch.any(torch.abs(coef) > threshold, dim=1)
        active_idx = torch.nonzero(active_mask, as_tuple=False).view(-1)

        if active_idx.numel() == 0:
            print(f"Warning: All terms are pruned.")

        # Prune the library functions
        idx_list = active_idx.tolist()
        model.library.library = [model.library.library[i] for i in idx_list]
        model.library.function_names = [model.library.function_names[i] for i in idx_list]
        model.library.shape = (len(idx_list), model.library.shape[1])

        # Prune the coefficients
        pruned_coef = coef[active_idx]
        model.coef = torch.nn.Parameter(pruned_coef.clone(), requires_grad=True)
    return model

def map_to_new_fx(trained_fx, base_fx):
    """
    Map a trained SINDy model's coefficients to a full function library with potentially more terms.

    Args:
        trained_fx: a trained SINDy model (with pruned library)
        base_fx: a reference SINDy model with the full library (untrained)

    Returns:
        A deepcopy of base_fx with coefficients filled from trained_fx
    """

    def normalize_expr(expr):
        """
        Normalize monomial expressions like 'x1 * x0^2' to 'x0^2 * x1' for consistent matching.
        """
        expr = expr.replace(" ", "")
        if "*" in expr:
            terms = expr.split("*")
            return " * ".join(sorted(terms))
        else:
            return expr

    # Create a deep copy of base_fx so the original remains untouched
    mapped_fx = copy.deepcopy(base_fx)

    # Normalize function names for safe comparison
    base_names_raw = mapped_fx.library.function_names
    trained_names_raw = trained_fx.library.function_names

    base_names = [normalize_expr(name) for name in base_names_raw]
    trained_names = [normalize_expr(name) for name in trained_names_raw]

    # Initialize zero-filled coefficient matrix
    coef_full = torch.zeros((len(base_names), trained_fx.coef.shape[1]),
                            device=trained_fx.coef.device)

    # Map coefficients from trained model into the full model
    for i, t_name in enumerate(trained_names):
        if t_name in base_names:
            j = base_names.index(t_name)
            coef_full[j] = trained_fx.coef[i]
        else:
            print(f"⚠️ Warning: term '{trained_names_raw[i]}' not found in full library")

    mapped_fx.coef = torch.nn.Parameter(coef_full, requires_grad=False)
    return mapped_fx

def save_model_with_timestamp(model:nn.Module, base_name:str, directory:str):
    """
    Saves the model's state_dict with a unique timestamped filename.

    Args:
        model (nn.Module): The PyTorch model to save.
        base_name (str): Base name for the file, e.g., 'fx1_trained'.
        directory (str): Directory to save the model in.
    """
    os.makedirs(directory, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_name}_{timestamp}.pth"
    path = os.path.join(directory, filename)
    torch.save(model.state_dict(), path)
    print(f"✅ Model saved to {path}")
    return path  # Optional: return path for logging

def load_latest_model(model:nn.Module, base_name:str, directory:str):
    """
    Loads the latest saved model state_dict based on timestamped filenames.

    Args:
        model (nn.Module): An instance of the model class to load into.
        base_name (str): Base filename prefix, e.g., 'fx1_trained'.
        directory (str): Directory where models are saved.

    Returns:
        model (nn.Module): The model loaded with the latest weights.
    """
    pattern = os.path.join(directory, f"{base_name}_*.pth")
    model_files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    print(model_files)
    if not model_files:
        raise FileNotFoundError(f"No saved model found with base name '{base_name}' in '{directory}'.")

    latest_file = model_files[0]
    model.load_state_dict(torch.load(latest_file))
    print(f"✅ Loaded model from: {latest_file}")
    return model
