import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from models import CVAE
from utils import load_data, is_valid_with_stats, compute_feature_bounds

from torch.distributions import Categorical, MixtureSameFamily, MultivariateNormal

def main(data_dir, model_path, n_samples, output_path):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1) Load real data
    x, y = load_data(data_dir)
    x_np = x.cpu().numpy()
    y_np = y.cpu().numpy()

    # 2) Compute validity bounds
    catch22_bounds   = compute_feature_bounds(x_np, y_np, features='catch22')
    envelope_bounds  = compute_feature_bounds(x_np, y_np, features='envelope')

    # 3) Load trained cVAE
    cvae = CVAE(input_dim=x.size(1), latent_dim=2, num_classes=2, y_emb=1).to(device)
    cvae.load_state_dict(torch.load(model_path, map_location=device))
    cvae.eval()

    # 4) Encode real valid data to build mixtures
    valid_mask = is_valid_with_stats(x, catch22_bounds, envelope_bounds)
    ds_real = TensorDataset(x[valid_mask], y[valid_mask], torch.ones(valid_mask.sum(), dtype=torch.long))
    loader  = DataLoader(ds_real, batch_size=64)

    mus_list, logvars_list = [], []
    with torch.no_grad():
        for xb, yb, yb_r in loader:
            xb, yb, yb_r = xb.to(device), yb.to(device), yb_r.to(device)
            mu_b, logvar_b = cvae.encode(xb, yb, yb_r)
            mus_list.append(mu_b.cpu())
            logvars_list.append(logvar_b.cpu())

    mu_all     = torch.cat(mus_list, dim=0)
    logvar_all = torch.cat(logvars_list, dim=0)
    covs_all   = torch.stack([torch.diag(torch.exp(lv)) for lv in logvar_all], dim=0)

    # 5) Create a mixture distribution per class
    mixtures = {}
    for cls in [0, 1]:
        idx = (y_np[valid_mask.cpu().numpy()] == cls)
        mus_c  = mu_all[idx].to(device)
        covs_c = covs_all[idx].to(device)
        probs  = torch.ones(len(mus_c), device=device) / len(mus_c)
        mixtures[cls] = MixtureSameFamily(
            Categorical(probs),
            MultivariateNormal(loc=mus_c, covariance_matrix=covs_c)
        )

    # 6) Sample n valid synthetic data points per class
    x_synth = {0: [], 1: []}
    for cls, mix in mixtures.items():
        samples = []
        # Oversample then filter by validity
        while len(samples) < n_samples:
            z_new = mix.sample((n_samples * 2,))
            yc    = torch.full((z_new.size(0),), cls, dtype=torch.long, device=device)
            yv    = torch.ones_like(yc)
            with torch.no_grad():
                x_dec = cvae.decode(z_new.to(device), yc, yv).cpu().numpy()

            valid_flags = is_valid_with_stats(torch.from_numpy(x_dec), catch22_bounds, envelope_bounds)
            for xi, valid in zip(x_dec, valid_flags):
                if valid and len(samples) < n_samples:
                    samples.append(xi)
            # continue until enough
        x_synth[cls] = np.stack(samples, axis=0)

    # 7) Prepare real signals per class
    real_signals = {
        cls: x_np[(y_np == cls) & (valid_mask.cpu().numpy())]
        for cls in [0, 1]
    }

    # 8) Plot mean +/- std for real vs synthetic per class
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, cls in zip(axes, [0, 1]):
        real = real_signals[cls]
        synth = x_synth[cls]

        mean_r = real.mean(axis=0)
        std_r  = real.std(axis=0)
        mean_s = synth.mean(axis=0)
        std_s  = synth.std(axis=0)

        t = np.arange(mean_r.shape[0])
        # Real
        ax.plot(t, mean_r, label='Real mean')
        ax.fill_between(t, mean_r - std_r, mean_r + std_r, alpha=0.3)
        # Synth
        ax.plot(t, mean_s, label='Synthetic mean', linestyle='--')
        ax.fill_between(t, mean_s - std_s, mean_s + std_s, alpha=0.3)

        ax.set_title(f'Class {cls}')
        ax.set_xlabel('Time index')
        ax.set_ylabel('Signal value')
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Omega: compare real vs synthetic signals')
    parser.add_argument('--data_dir',   type=str, default='GunPoint', help='Path to data directory')
    parser.add_argument('--model_path', type=str, default='best_model.pt', help='Path to saved cVAE model')
    parser.add_argument('--n_samples',  type=int, default=100, help='Number of valid samples per class')
    parser.add_argument('--output',     type=str, default='test_omega.png', help='Output plot file')
    args = parser.parse_args()

    main(args.data_dir, args.model_path, args.n_samples, args.output)
