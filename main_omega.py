import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pyro
import pyro.distributions as dist
from torch.distributions import Categorical, MixtureSameFamily, MultivariateNormal
import umap
import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from pycatch22 import catch22_all
from losses import loss_fn, contrastive_loss
from models import CVAE, ValidityNet, EarlyStopping
from utils import load_data, is_valid, is_valid_with_stats, compute_feature_bounds, sample_varied_gaussian
from plots import plot_umap_latents, plot_umap_signals, plot_class_time_series, plot_gaussian_ellipse

# ------------------------
# PIPELINE D'ENTRAÎNEMENT cVAE AVEC MIXTURE DE GAUSSIENNES LOCALES
# ------------------------
def train_cVAE_pipeline(data_dir):
    # 1. Warm-up
    torch.manual_seed(0)
    x, y = load_data(data_dir)
    X_np, y_np = x.cpu().numpy(), y.cpu().numpy()
    catch22_bounds = compute_feature_bounds(X_np, y_np, features="catch22")
    envelope_bounds = compute_feature_bounds(X_np, y_np, features="envelope")

    print(f"Nombre de données initiales chargées : {len(x)}")
    valid_init = is_valid_with_stats(x, catch22_bounds, envelope_bounds)
    print(f"Nombre de données valides initiales : {valid_init.sum().item()} sur {len(x)}")
    if not valid_init.all():
        print("Attention : certaines données initiales ne sont pas valides !")
        exit(1)

    y_real = torch.ones(len(x), dtype=torch.long)
    is_synth_all = torch.zeros(len(x), dtype=torch.bool, device=device)
    n_real = x.shape[0]

    ds_real = TensorDataset(x, y, y_real)
    n_val = int(len(ds_real)*0.1)
    train_ds, val_ds = random_split(ds_real, [len(ds_real)-n_val, n_val])
    dl_train = DataLoader(train_ds, batch_size=32, shuffle=True)
    dl_val   = DataLoader(val_ds, batch_size=32)

    cvae = CVAE(input_dim=x.size(1), latent_dim=2, num_classes=2, y_emb=1).to(device)
    V_net = ValidityNet(input_dim=x.size(1)).to(device)
    optimizer = torch.optim.Adam(cvae.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.75)
    stopper = EarlyStopping(patience=100, min_delta=1e-4, mode='min')

    for epoch in range(1, 1000):
        cvae.train(); V_net.train()
        train_loss = 0.0
        for xb, yb, yb_r in dl_train:
            xb, yb, yb_r = xb.to(device), yb.to(device), yb_r.to(device)
            optimizer.zero_grad()
            x_hat, mu, logvar, z, _, _ = cvae(xb, yb, yb_r)
            val_pred = V_net(x_hat)
            loss_val = F.binary_cross_entropy_with_logits(val_pred, yb_r.float(), reduction='sum')
            loss_vae, loss_cls, loss_valcls = loss_fn(xb, x_hat, mu, logvar, z, yb, yb_r,
                                                       cvae.cls_label, cvae.cls_cons, lambda_cons=0)
            loss = loss_vae + loss_val
            loss.backward(); optimizer.step()
            train_loss += loss.item()
        sched.step()

        cvae.eval(); V_net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb, yb_r in dl_val:
                xb, yb, yb_r = xb.to(device), yb.to(device), yb_r.to(device)
                x_hat, mu, logvar, z, _, _ = cvae(xb, yb, yb_r)
                val_pred = V_net(x_hat)
                loss_val = F.binary_cross_entropy_with_logits(val_pred, yb_r.float(), reduction='sum')
                vloss, vcls, vvalcls = loss_fn(xb, x_hat, mu, logvar, z, yb, yb_r,
                                                cvae.cls_label, cvae.cls_cons, lambda_cons=0)
                val_loss += (vloss + loss_val).item()
        avg_val = val_loss / len(dl_val.dataset)
        print(f"[Warm-up] Époch {epoch} → Train loss: {train_loss/len(dl_train.dataset):.4f}, "
              f"Val loss: {avg_val:.4f}")
        stopper(avg_val, cvae)
        if stopper.should_stop:
            print(f"→ Early stopping warm-up à l'époque {epoch}")
            cvae.load_state_dict(torch.load('best_model.pt'))
            break

    # Préparation du pool global
    x_all, y_all = x.clone(), y.clone()
    y_valid_all = y_real.clone()

    num_cycles = 100
    samples_per_cycle = 10
    alpha = 1.5

    for cycle in range(1, num_cycles+1):
        print(f"\n=== Cycle {cycle} ===")
        print(f"Avant génération : total = {len(x_all)}, synthétiques = {is_synth_all.sum().item()}")
        print(f"Répartition y=1: {(y_all==1).sum().item()} | y=0: {(y_all==0).sum().item()} | Valides: {y_valid_all.sum().item()}")

        # Encodage complet
        cvae.eval()
        mus, logvars = [], []
        with torch.no_grad():
            for xb, yb, yb_r in DataLoader(TensorDataset(x_all, y_all, y_valid_all), batch_size=64):
                xb, yb, yb_r = xb.to(device), yb.to(device), yb_r.to(device)
                mu_b, logvar_b = cvae.encode(xb, yb, yb_r)
                mus.append(mu_b.cpu()); logvars.append(logvar_b.cpu())
        mu_all = torch.cat(mus, dim=0)
        logvar_all = torch.cat(logvars, dim=0)

        
        # Construction des mixtures (scalée et non-scalée)
        covs_all = torch.stack([torch.diag(torch.exp(logvar_all[i])) for i in range(len(logvar_all))], dim=0)
        mixtures_scaled   = {}
        mixtures_unscaled = {}
        for c in torch.unique(y_all):
            mask = (y_all==c) & (y_valid_all==1)
            idx = mask.nonzero(as_tuple=False).view(-1)
            if len(idx)==0: continue
            mus_c  = mu_all[idx].to(device)      # [K, D]
            covs_c = covs_all[idx].to(device)    # [K, D, D]
            probs  = torch.ones(len(idx), device=device)/len(idx)

            # mixture scalée (comme avant)
            mix_s = MixtureSameFamily(Categorical(probs),
                                    MultivariateNormal(loc=mus_c,
                                                        covariance_matrix=alpha*covs_c))
            mixtures_scaled[int(c.item())] = mix_s

            # mixture _non_-scalée
            mix_u = MixtureSameFamily(Categorical(probs),
                                    MultivariateNormal(loc=mus_c,
                                                        covariance_matrix=covs_c))
            mixtures_unscaled[int(c.item())] = mix_u


        # Échantillonnage et décodage
        x_gen, y_gen, yv_gen = [], [], []
        for c, mix in mixtures_scaled.items():
            z_new = mix.sample((samples_per_cycle,))
            yc = torch.full((len(z_new),), c, dtype=torch.long, device=device)
            yv = torch.ones_like(yc)
            with torch.no_grad():
                x_dec = cvae.decode(z_new, yc, yv).cpu()
            valid_oracle = is_valid_with_stats(x_dec, catch22_bounds, envelope_bounds)
            kept = valid_oracle.sum().item()
            print(f"Classe {c}: {kept}/{len(x_dec)} synthétiques valides")
            for xi, val in zip(x_dec, valid_oracle):
                x_gen.append(xi.unsqueeze(0)); y_gen.append(c); yv_gen.append(int(val))

        # Ajout au pool
        if x_gen:
            x_new = torch.cat(x_gen, dim=0)
            y_new = torch.tensor(y_gen, dtype=torch.long)
            yv_new= torch.tensor(yv_gen, dtype=torch.long)
            x_all = torch.cat([x_all, x_new], dim=0)
            y_all = torch.cat([y_all, y_new], dim=0)
            y_valid_all = torch.cat([y_valid_all, yv_new], dim=0)
            is_synth_all = torch.cat([is_synth_all, torch.ones(len(x_new), dtype=torch.bool, device=device)], dim=0)
            print(f"Après génération : total = {len(x_all)}, nouveaux = {len(x_new)}")

        # Ré-entraînement
        ds_all = TensorDataset(x_all, y_all, y_valid_all)
        n_val = int(len(ds_all)*0.1)
        tr_ds, vl_ds = random_split(ds_all, [len(ds_all)-n_val, n_val])
        dl_tr = DataLoader(tr_ds, batch_size=16, shuffle=True)
        dl_vl = DataLoader(vl_ds, batch_size=16)

        stopper_c = EarlyStopping(patience=10, min_delta=1e-4, mode='min')
        optimizer = torch.optim.Adam(cvae.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        for epoch in range(1,1000):
            cvae.train(); V_net.train()
            train_l = 0; cnt=0
            for xb, yb, yb_r in dl_tr:
                xb, yb, yb_r = xb.to(device), yb.to(device), yb_r.to(device)
                optimizer.zero_grad()
                x_hat, mu, logvar, z, _, _ = cvae(xb, yb, yb_r)
                val_pred = V_net(x_hat)
                loss_val = F.binary_cross_entropy_with_logits(val_pred, yb_r.float(), reduction='sum')
                l_vae, l_cls, l_valcls = loss_fn(xb, x_hat, mu, logvar, z, yb, yb_r,
                                                  cvae.cls_label, cvae.cls_cons, lambda_cons=2)
                loss = l_vae + loss_val
                loss.backward(); optimizer.step()
                train_l += loss.item(); cnt+=1
            sched.step()
            # validation
            cvae.eval()
            vl = 0
            with torch.no_grad():
                for xb, yb, yb_r in dl_vl:
                    xb, yb, yb_r = xb.to(device), yb.to(device), yb_r.to(device)
                    x_hat, mu, logvar, z, _, _ = cvae(xb, yb, yb_r)
                    val_pred = V_net(x_hat)
                    loss_val = F.binary_cross_entropy_with_logits(val_pred, yb_r.float(), reduction='sum')
                    l_vae, l_cls, l_valcls = loss_fn(xb, x_hat, mu, logvar, z, yb, yb_r,
                                                      cvae.cls_label, cvae.cls_cons, lambda_cons=2)
                    vl += (l_vae.item() + loss_val.item())
            avg_vl = vl/len(dl_vl.dataset)
            # print(f"Cycle {cycle} [Epoch {epoch}] → Val loss: {avg_vl:.4f}")
            stopper_c(avg_vl, cvae)
            if stopper_c.should_stop:
                # print(f"→ Early stopping cycle {cycle} à l'époque {epoch}")
                cvae.load_state_dict(torch.load('best_model.pt'))
                break

        # Visualisation tous les 10 cycles
        if cycle % 10 == 0 or cycle == num_cycles :
            cvae.eval()
            with torch.no_grad():
                mus_batches = []
                labels_batches = []
                valid_labels_batches = []
                for xb, yb, yb_valid in DataLoader(TensorDataset(x_all, y_all, y_valid_all), batch_size=64):
                    xb, yb, yb_valid = xb.to(device), yb.to(device), yb_valid.to(device)
                    mu_b, _ = cvae.encode(xb, yb, yb_valid)
                    mus_batches.append(mu_b.cpu().numpy())
                    labels_batches.append(yb.cpu().numpy())
                    valid_labels_batches.append(yb_valid.cpu().numpy())
                mus_all = np.vstack(mus_batches)       # (N_total, 2)
                labels_all = np.concatenate(labels_batches)  # (N_total,)
                valid_labels_all = np.concatenate(valid_labels_batches)  # (N_total,)
            is_synth_all_np = is_synth_all.cpu().numpy()  
         
         # -- Juste après la création de mus_all, labels_all, valid_labels_all --
            Z_all = mus_all   # (N_total, latent_dim) (déjà np.ndarray si tu fais .cpu().numpy())
            labels = labels_all
            valid_mask = valid_labels_all
            is_synth = is_synth_all_np

            # Pour Z_real_init, prends les vrais de départ (dans l’ordre de concaténation)
            Z_real_init = mus_all[:n_real]   # <-- n_real = len(x) initial, ou ton nombre de réels d'origine

            # plot_umap_latents(Z_real_init, Z_all, labels, valid_mask, is_synth, cycle)

            # 1) Ajuster la PCA sur TOUT votre espace latent
            # pca = PCA(n_components=2)
            Z2 = mus_all  # (N_total, 2)
            mu_mean_per_class = []
            cov_per_class = []
            for c in torch.unique(torch.tensor(labels_all)):
                mask = (labels_all == c) & (valid_labels_all == 1)
                if mask.sum() == 0:
                    continue
                mu_c = mus_all[mask].mean(axis=0)
                cov_c = np.cov(mus_all[mask], rowvar=False)
                mu_mean_per_class.append(torch.tensor(mu_c, dtype=torch.float32))
                cov_per_class.append(torch.tensor(cov_c, dtype=torch.float32))
            # 2) Scatter des points réels / synthétiques dans le plan 2D
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
             # 3) Créer la grille 2D dans l’espace PCA
            n_pts = 100
            x_min, x_max = Z2[:,0].min()-1.0, Z2[:,0].max()+1.0
            y_min, y_max = Z2[:,1].min()-1.0, Z2[:,1].max()+1.0
            xx, yy = np.meshgrid(np.linspace(x_min,x_max,n_pts),
                                np.linspace(y_min,y_max,n_pts))
            grid2d = np.stack([xx.ravel(), yy.ravel()], axis=1)  # (n_pts², 2)

            # 4) Inverser la PCA pour remonter en ℝᵈ
            # latent_grid = pca.inverse_transform(grid2d)          # (n_pts², latent_dim)
            z_grid = torch.from_numpy(grid2d).float().to(device)

            # ─── Tracé des contours PDF non-scalés ───
            for class_label, mix in mixtures_unscaled.items():
                with torch.no_grad():
                    logp     = mix.log_prob(z_grid)                     # (n_pts²,)
                    pdf_vals = logp.exp().cpu().numpy().reshape(xx.shape)
                level = pdf_vals.max() * 0.8  # par exemple 10% du max

                ax1.contour(
                    xx, yy, pdf_vals,
                    levels=[level],
                    colors=[f"C{class_label}"],
                    linestyles='-',
                    linewidths=1.5,
                    label=f"Unscaled PDF cls{class_label}"
                )

            # ─── Tracé des contours PDF scalés (avec α) ───
            for class_label, mix in mixtures_scaled.items():
                with torch.no_grad():
                    logp     = mix.log_prob(z_grid)
                    pdf_vals = logp.exp().cpu().numpy().reshape(xx.shape)
                level = pdf_vals.max() * 0.8

                ax1.contour(
                    xx, yy, pdf_vals,
                    levels=[level],
                    colors=[f"C{class_label}"],
                    linestyles='--',   # tirets pour distinction
                    linewidths=1.5,
                    label=f"Scaled PDF cls{class_label}"
                )


            # ex. Plot 1 : mêmes sélections qu’avant mais en 2D
            for cls, marker in zip([0,1], ['o','^']):
                for synth, color in [(False,'blue'), (True,'red')]:
                    mask = (labels_all==cls) & (is_synth_all_np==synth) & (valid_labels_all==1)
                    ax1.scatter(Z2[mask,0], Z2[mask,1],
                                marker=marker, color=color,
                                label=f"{'réel' if not synth else 'synth'}·cls{cls}",
                                s=30, alpha=0.7)
            ax1.set_title("Valides seulement — Classe @0.5")
            ax1.set_xlabel("PC1"); ax1.set_ylabel("PC2")
            ax1.legend(fontsize='small')

           
            # 5) Évaluer vos classifieurs sur ces z_grid
            with torch.no_grad():
                probs_class = torch.sigmoid(cvae.cls_label(z_grid)).cpu().numpy().reshape(xx.shape)
                probs_valid = torch.sigmoid(cvae.cls_cons(z_grid)).cpu().numpy().reshape(xx.shape)

            # 6) Tracer les frontières de décision
            ax1.contour(xx, yy, probs_class, levels=[0.5], colors='k', linestyles='--')
            # 6) Oracle validité en décodant z_grid (on force yv=1)
            valid_oracle = []
            batch_size = n_pts**2
            yc = torch.zeros(batch_size, dtype=torch.long, device=device)
            yv = torch.ones( batch_size, dtype=torch.long, device=device)
            with torch.no_grad():
                for i in range(0, grid2d.shape[0], batch_size):
                    zb = z_grid[i:i+batch_size]
                    lbl_clf = torch.sigmoid(cvae.cls_label(zb))
                    lbl_cons = torch.sigmoid(cvae.cls_cons(zb))
                    # sigmoid to label
                    lbl_clf = (lbl_clf > 0.5).long()
                    lbl_cons = (lbl_cons > 0.5).long()
                    print(lbl_clf.shape, lbl_cons.shape, zb.shape)
                    

                    xy = cvae.decode(zb, yc, yv).cpu()
                    valid_oracle.append(is_valid_with_stats(xy, catch22_bounds, envelope_bounds).numpy())
            valid_oracle = np.concatenate(valid_oracle).reshape(xx.shape)

            # 7) Tracer les frontières dans le plan PCA
            ax2.contour(xx, yy, probs_valid,  levels=[0.5], colors='k', linestyles='--')
            # ax2.contour(xx, yy, valid_oracle, levels=[0.5], colors='r', linestyles='-')

            # Scatter global
            for synth, color in [(False,'blue'), (True,'orange'), (True,'red')]:
                if not synth:
                    mask = ~is_synth_all_np
                    lbl = "réel"
                else:
                    mask = is_synth_all_np & (valid_labels_all==1) if color=='orange' else is_synth_all_np & (valid_labels_all==0)
                    lbl = "synth-val" if color=='orange' else "synth-inv"
                for cls, marker in zip([0,1], ['o','^']):
                    m = mask & (labels_all==cls)
                    ax2.scatter(Z2[m,0], Z2[m,1],
                                marker=marker, color=color,
                                label=f"{lbl}·cls{cls}",
                                s=25, alpha=0.6)
            ax2.set_title("Toutes données — Validité @0.5")
            ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2")
            ax2.legend(fontsize='small')
            plt.tight_layout()
            plt.savefig(f"visu/combined_plots_cycle_{cycle}_pca.png")
            plt.close(fig)
            # Après la génération/ajout au pool, récupère les infos

            # Juste avant la visualisation
            X_real_init = x[:n_real].cpu().numpy()      # Si tu as gardé x de base
            X_dec = x_all.cpu().numpy()
            labels = y_all.cpu().numpy()
            valid_mask = y_valid_all.cpu().numpy()
            is_synth = is_synth_all.cpu().numpy()

            plot_umap_signals(X_real_init, X_dec, labels, valid_mask, is_synth, cycle)
            plot_class_time_series(
                X_init     = X_real_init,
                X_all      = X_dec,
                y_all      = labels_all,
                y_valid_all= valid_mask,
                is_synth_all=is_synth,
                outdir     ="visu",
                prefix     =f"cycle_{cycle}"
            )





    print("\n→ Pipeline cVAE avec mixture terminé.")

if __name__=='__main__':
    train_cVAE_pipeline('GunPoint')