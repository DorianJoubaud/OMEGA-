import os
import umap
import matplotlib.pyplot as plt
import numpy as np
from pycatch22 import catch22_all


def plot_umap_latents(
        Z_real_init,     # (N0, D) np.ndarray – latents réels initiaux
        Z_all,           # (N, D)  np.ndarray – latents pool complet
        labels,          # (N,)    np.ndarray – labels classe (0/1)
        valid_mask,      # (N,)    np.ndarray – validité (0/1)
        is_synth,        # (N,)    np.ndarray – synthétique (1) ou réel (0)
        cycle,           # int     – numéro de cycle
        outdir="visu"
    ):
    """
    UMAP: fit sur latents réels initiaux, transform sur tout le pool latent
    """
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    umap_model = umap.UMAP(n_components=2, random_state=0)
    umap_model.fit(Z_real_init)
    proj = umap_model.transform(Z_all)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    colors = {0: 'blue', 1: 'red'}
    markers = {0: 'o', 1: 'x'}
    val_colors = {0: 'grey', 1: 'green'}

    # Plot 1 : classe + synth/real
    for cls in [0, 1]:
        for synth in [0, 1]:
            mask = (labels == cls) & (is_synth == synth)
            if mask.sum() == 0:
                continue
            label = f"{'réel' if synth == 0 else 'synth'}·cls{cls}"
            ax1.scatter(proj[mask,0], proj[mask,1],
                        c=colors[cls], marker=markers[synth],
                        label=label, alpha=0.7, s=30, 
                        edgecolor='k' if synth else None)
    ax1.set_title("UMAP latents – Couleur=classe, marker=réel/synth")
    ax1.legend(fontsize='small')

    # Plot 2 : validité oracle + synth/real
    for valid in [0, 1]:
        for synth in [0, 1]:
            mask = (valid_mask == valid) & (is_synth == synth)
            if mask.sum() == 0:
                continue
            label = f"{'réel' if synth == 0 else 'synth'}·val{valid}"
            ax2.scatter(proj[mask,0], proj[mask,1],
                        c=val_colors[valid], marker=markers[synth],
                        label=label, alpha=0.7, s=30,
                        edgecolor='k' if synth else None)
    ax2.set_title("UMAP latents – Couleur=valide (oracle), marker=réel/synth")
    ax2.legend(fontsize='small')

    plt.tight_layout()
    plt.savefig(f"{outdir}/umap_latents_cycle_{cycle}.png")
    plt.close(fig)


def plot_umap_signals(X_real_init, X_dec, labels, valid_mask, is_synth, cycle, outdir="visu"):
    """
    UMAP: fit sur les données de base, transform sur tout le pool
    X_real_init : (N0, L) - signaux de base (array)
    X_dec       : (N, L)  - tous les signaux à projeter (array)
    labels      : (N,)    - classe 0/1 (array)
    valid_mask  : (N,)    - validité oracle 0/1 (array)
    is_synth    : (N,)    - 0 = réel, 1 = synthétique (array)
    cycle       : int     - numéro de cycle
    outdir      : str     - dossier d’export
    """
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    # Fit UMAP sur les signaux de base (ex: les 200 premiers du pool initial)
    umap_model = umap.UMAP(n_components=2, random_state=0)
    umap_model.fit(X_real_init)
    proj = umap_model.transform(X_dec)    # shape (N, 2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    colors = {0: 'blue', 1: 'red'}
    markers = {0: 'o', 1: 'x'}
    val_colors = {0: 'grey', 1: 'green'}

    # --- Plot 1 : classe + synth/real
    for cls in [0, 1]:
        for synth in [0, 1]:
            mask = (labels == cls) & (is_synth == synth)
            if mask.sum() == 0:
                continue
            label = f"{'réel' if synth == 0 else 'synth'}·cls{cls}"
            ax1.scatter(proj[mask,0], proj[mask,1],
                        c=colors[cls], marker=markers[synth],
                        label=label, alpha=0.7, s=30, 
                        edgecolor='k' if synth else None)
    ax1.set_title("UMAP séries – Couleur=classe, marker=réel/synth")
    ax1.legend(fontsize='small')

    # --- Plot 2 : validité oracle + synth/real
    for valid in [0, 1]:
        for synth in [0, 1]:
            mask = (valid_mask == valid) & (is_synth == synth)
            if mask.sum() == 0:
                continue
            label = f"{'réel' if synth == 0 else 'synth'}·val{valid}"
            ax2.scatter(proj[mask,0], proj[mask,1],
                        c=val_colors[valid], marker=markers[synth],
                        label=label, alpha=0.7, s=30,
                        edgecolor='k' if synth else None)
    ax2.set_title("UMAP séries – Couleur=valide (oracle), marker=réel/synth")
    ax2.legend(fontsize='small')

    plt.tight_layout()
    plt.savefig(f"{outdir}/umap_signals_cycle_{cycle}.png")
    plt.close(fig)
    
def plot_class_time_series(X_init, X_all, y_all, y_valid_all, is_synth_all, outdir="visu", prefix="cycle"):
    """
    Pour chaque classe, trace :
      • la moyenne ±1σ des séries initiales (non synthétiques)
      • la moyenne ±1σ des séries synthétiques valides
    X_init         : np.ndarray (n_real, L)  – signaux initiaux
    X_all          : np.ndarray (N,   L)    – tout le pool (initiaux + générés)
    y_all          : np.ndarray (N,)       – labels de classe 0/1
    y_valid_all    : np.ndarray (N,)       – validité oracle (0/1)
    is_synth_all   : np.ndarray (N,)       – indicateur synthétique (True/False)
    """
    os.makedirs(outdir, exist_ok=True)
    classes = np.unique(y_all)
    t = np.arange(X_init.shape[1])
    
    for cls in classes:
        # Masques
        mask_init = (y_all == cls) & (~is_synth_all)
        mask_gen  = (y_all == cls) & ( is_synth_all) & (y_valid_all==1)

        if mask_init.sum()==0 or mask_gen.sum()==0:
            continue

        # Stats init
        Xi = X_all[mask_init]
        mu_i  = Xi.mean(axis=0)
        sigma_i = Xi.std(axis=0)

        # Stats gen valides
        Xg = X_all[mask_gen]
        mu_g  = Xg.mean(axis=0)
        sigma_g = Xg.std(axis=0)

        plt.figure(figsize=(8,4))
        # init
        plt.plot(t, mu_i, label="Initiales",    linestyle='-')
        plt.fill_between(t, mu_i - sigma_i, mu_i + sigma_i, alpha=0.3)
        # synth validées
        plt.plot(t, mu_g, label="Synth-valides", linestyle='--')
        plt.fill_between(t, mu_g - sigma_g, mu_g + sigma_g, alpha=0.3)

        plt.title(f"Classe {cls} — cycle {prefix}")
        plt.xlabel("Temps")
        plt.ylabel("Valeur normalisée")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{outdir}/class_{cls}_{prefix}_mean_std.png")
        plt.close()
