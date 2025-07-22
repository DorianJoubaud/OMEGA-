from pycatch22 import catch22_all

import os
import numpy as np
import torch
import torch.distributions as dist
from scipy import fft
import torch.nn.functional as F
import matplotlib.pyplot as plt
import umap

# ------------------------
# GAUSSIAN SAMPLING IN LATENT SPACE
# ------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
def sample_varied_gaussian(mu, cov, alpha_min=0.8, alpha_max=1.2, n=200):
    mvn = dist.MultivariateNormal(mu, covariance_matrix=alpha_min * cov)
    z_samples = mvn.sample((n,)).to(device)
    return z_samples, alpha_min


def is_valid_with_stats_per_class(x, class_idx, catch22_bounds, envelope_bounds):
    # x : (N, L) ou (L,)
    if x.ndim == 1:
        x = x[None, :]
    # catch22
    # print(class_idx)
    lower_c22, upper_c22 = catch22_bounds[class_idx]
    lower_env, upper_env = envelope_bounds[class_idx]
    is_valid_list = []
    for xi in x:
        # 1. catch22
        feats = np.array(catch22_all(xi.cpu().numpy())["values"])
        valid_c22 = np.all((feats >= lower_c22) & (feats <= upper_c22))
        # 2. envelope
        valid_env = np.all((xi.cpu().numpy() >= lower_env) & (xi.cpu().numpy() <= upper_env))
        is_valid_list.append(valid_c22)
    return torch.tensor(is_valid_list, device=x.device)

def is_valid_with_stats(x, catch22_bounds, envelope_bounds):
    # x : (N, L) ou (L,)
    if x.ndim == 1:
        x = x[None, :]
    # catch22
    # print(class_idx)
    is_valid_list = []
    lower_c22, upper_c22 = catch22_bounds
    lower_env, upper_env = envelope_bounds
    for xi in x:
        # 1. catch22
        feats = np.array(catch22_all(xi.cpu().numpy())["values"])
        valid_c22 = np.all((feats >= lower_c22) & (feats <= upper_c22))
        # 2. envelope
        valid_env = np.all((xi.cpu().numpy() >= lower_env) & (xi.cpu().numpy() <= upper_env))
        is_valid_list.append(valid_c22)
    return torch.tensor(is_valid_list, device=x.device)



def compute_feature_bounds_per_class(X, y, features="catch22"):
    bounds_per_class = {}
    for cls in np.unique(y):
        X_cls = X[y==cls]
        if features == "catch22":
            feats = np.stack([list(catch22_all(xi)["values"]) for xi in X_cls])
            lower = np.min(feats, axis=0)
            upper = np.max(feats, axis=0)
            bounds_per_class[cls] = (lower, upper)
        elif features == "envelope":
            mean = X_cls.mean(axis=0)
            std = X_cls.std(axis=0)
            # 99% envelope
            lower = mean - 2.58*std
            upper = mean + 2.58*std
            bounds_per_class[cls] = (lower, upper)
    return bounds_per_class

def compute_feature_bounds(X, y, features="catch22"):
    bounds = {}
    if features == "catch22":
        feats = np.stack([list(catch22_all(xi)["values"]) for xi in X])
        lower = np.min(feats, axis=0)
        upper = np.max(feats, axis=0)
        bounds = (lower, upper)
    elif features == "envelope":
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        # 99% envelope
        lower = mean - 2.58*std
        upper = mean + 2.58*std
        bounds = (lower, upper)
    return bounds





# ------------------------
# LOAD 
# ------------------------
def load_data(data_dir):
    def _load(split):
        print(os.getcwd())
        path = os.path.join(data_dir, f'{data_dir}_{split}.tsv')
        arr = np.loadtxt(path)
        X = arr[:,1:]
        y = arr[:,0]
        # 1 = normal, -1 = abnormal
        y = (y == 1).astype(int)
        return X, y

    X_tr, y_tr = _load('TRAIN')
    X_te, y_te = _load('TEST')
    # X = np.vstack([X_tr, X_te])
    # y = np.concatenate([y_tr, y_te])
    
    X = X_tr.copy()
    y = y_tr.copy()
    # min max normalize X
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

    print(f"X.max(): {X.max()}, X.min(): {X.min()}")

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

# ------------------------
# VALIDITY CHECK (CONSTRAINTS)
# ------------------------

def is_valid(x, A_max=1.0, delta_max=0.55, noise_energy_min=1e-6):
    """
    Vérifie les contraintes 'hard' suivantes sur une ou plusieurs séries temporelles x:
    1. Amplitude bornée: max_t |x_t| <= A_max
    2. Variation locale limitée: max_t |x_t - x_{t-1}| <= delta_max
    3. Énergie du bruit (hautes fréquences) >= noise_energy_min

    Args:
        x: Tensor de forme (L,) ou (N, L)
        A_max: amplitude maximale autorisée
        delta_max: variation maximale entre échantillons consécutifs
        noise_energy_min: énergie minimale dans la bande haute fréquence
    Retourne:
        mask: BoolTensor indiquant pour chaque série si toutes les contraintes sont satisfaites
    """
    # Mise en forme en batch
    if x.dim() == 1:
        xs = x.unsqueeze(0)
    else:
        xs = x

    mask = []
    for xi in xs:
        # 1) Amplitude bornée
        amp = xi.abs().max().item()
        if amp > A_max:
            # print(f"Invalid amplitude {amp} > {A_max} for series.")
            mask.append(False)
            continue

        # 2) Variation locale limitée
        diffs = (xi[1:] - xi[:-1]).abs().max().item()
        if diffs > delta_max:
            # print(f"Invalid local variation {diffs} > {delta_max} for series.")
            mask.append(False)
            continue

        # 3) Énergie du bruit: calcul par FFT, on considère les fréquences > 0.5 * Nyquist
        Xf = fft.rfft(xi)
        freqs = torch.linspace(0, 1.0, Xf.size(-1), device=xi.device)
        # bande haute fréquence
        hf_mask = freqs > 0.5
        noise_energy = (Xf[hf_mask].abs().pow(2).sum() / Xf.abs().pow(2).sum()).item()
        if noise_energy < noise_energy_min:
            # print(f"Invalid noise energy {noise_energy} < {noise_energy_min} for series.")
            mask.append(False)
            continue

        # Si toutes les contraintes sont satisfaites
        mask.append(True)

    return torch.tensor(mask, dtype=torch.bool, device=x.device)
