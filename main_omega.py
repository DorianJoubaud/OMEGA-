
# ------------------------
# IMPORTS & DEVICE
# ------------------------
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pyro
import pyro.distributions as dist
import torch.fft as fft
import umap
import wandb
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from pycatch22 import catch22_all
from losses import loss_fn, contrastive_loss
from models import CVAE, ValidityNet, EarlyStopping
from utils import load_data, is_valid, is_valid_with_stats, compute_feature_bounds, sample_varied_gaussian
from plots import  plot_umap_latents, plot_umap_signals, plot_class_time_series

# ------------------------# ------------------------
# PIPELINE D'ENTRAÎNEMENT cVAE AVEC EARLY STOPPING
# ------------------------
def train_cVAE_pipeline(data_dir):
    # ————— 1. Génération initiale & Warm-up —————
    torch.manual_seed(0)
    x, y = load_data(data_dir)
    # Après load_data
    X_np = x.cpu().numpy()
    y_np = y.cpu().numpy()
    catch22_bounds = compute_feature_bounds(X_np, y_np, features="catch22")
    envelope_bounds = compute_feature_bounds(X_np, y_np, features="envelope")

    # sampler aleatoire 50 données de chaque classe
    # idx_0 = torch.where(y == 0)[0]
    # idx_1 = torch.where(y == 1)[0]
    # idx_0 = idx_0[torch.randperm(len(idx_0))[:100]]
    # idx_1 = idx_1[torch.randperm(len(idx_1))[:100]]
    # idx = torch.cat([idx_0, idx_1])
    
    # x = x[idx] # (100, 96)
    # y = y[idx] # (100,)
    # print(f'{len(x)} données initiales chargées dont {len(idx_0)} de classe 0 et {len(idx_1)} de classe 1.')
    y_real = torch.ones(len(x), dtype=torch.long)   # tous valides = 1
    # print le nombre de données x qui sont valides
    print(f"Nombre de données valides initiales : {is_valid_with_stats(x, catch22_bounds, envelope_bounds).sum().item()} sur {len(x)}")
    if not is_valid_with_stats(x, catch22_bounds, envelope_bounds).all():
        print("Attention : certaines données initiales ne sont pas valides !")
        exit(1)
    is_synth_all = torch.zeros(len(x), dtype=torch.bool, device=device)
    n_real = x.shape[0]


    # Création du dataset et split train/val pour le warm-up
    ds_real = TensorDataset(x, y, y_real)
    n_val = int(len(ds_real) * 0.1)
    n_train = len(ds_real) - n_val
    train_ds, val_ds = random_split(ds_real, [n_train, n_val])
    dl_train = DataLoader(train_ds, batch_size=32, shuffle=True)
    dl_val = DataLoader(val_ds, batch_size=32)

    # Instanciation du cVAE et de l'EarlyStopping pour le warm-up
    cvae = CVAE(input_dim=150, latent_dim=2, num_classes=2, y_emb=1).to(device)
    V_net = ValidityNet(input_dim=150).to(device)
    optimizer = torch.optim.Adam(cvae.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    stopper = EarlyStopping(patience=10, min_delta=1e-4, mode='min')

    # Warm-up (max 100 époques, arrêt précoce possible)
    for epoch in range(1, 1000):
        cvae.train()
        loss_total, loss_classif, loss_valid_classif = 0.0, 0.0, 0.0
        
        for xb, yb, y_r_b in dl_train:
            xb, yb, y_r_b = xb.to(device), yb.to(device), y_r_b.to(device)
            optimizer.zero_grad()
            # print(f"xb shape: {xb.shape}, yb shape: {yb.shape}, y_r_b shape: {y_r_b.shape}")
            x_hat, mu, logvar, z, _, _ = cvae(xb, yb, y_r_b)
            validity_pred = V_net(x_hat)
            loss_validity = F.binary_cross_entropy_with_logits(validity_pred, y_r_b.float(), reduction='sum')
            loss_total, loss_classif, loss_valid_classif = loss_fn(xb, x_hat, mu, logvar, z, yb, y_r_b, cvae.cls_label, cvae.cls_cons)

            loss_total += loss_validity
            loss_total.backward()

            optimizer.step()
            loss_total += loss_total.item()
            loss_classif += loss_classif.item()
            loss_valid_classif += loss_valid_classif.item()
        
        loss_total /= len(dl_train.dataset)
        loss_classif /= len(dl_train.dataset)
        loss_valid_classif /= len(dl_train.dataset)
        sched.step()

        # Calcul de la validation loss
        cvae.eval()
        V_net.eval()
        val_loss_total, val_loss_classif, val_loss_valid_classif = 0.0, 0.0, 0.0
        val_validity_total = 0.0
        with torch.no_grad():
            for xb_val, yb_val, yb_r_val in dl_val:
                xb_val, yb_val, yb_r_val = xb_val.to(device), yb_val.to(device), yb_r_val.to(device)
                x_hat_val, mu_val, logvar_val, z_val, _, _ = cvae(xb_val, yb_val, yb_r_val)
                validity_pred_val = V_net(x_hat_val)
                val_loss, val_classif_loss, val_valid_classif_loss = loss_fn(xb_val, x_hat_val, mu_val, logvar_val, z_val, yb_val, yb_r_val , cvae.cls_label, cvae.cls_cons)
                val_loss_total += val_loss.item()
                val_loss_classif += val_classif_loss.item()
                val_loss_valid_classif += val_valid_classif_loss.item()
                val_validity_total += F.binary_cross_entropy_with_logits(validity_pred_val, yb_r_val.float(), reduction='sum').item()
                
        avg_val_loss = val_loss_total / len(dl_val.dataset)
        avg_val_classif_loss = val_loss_classif / len(dl_val.dataset)
        avg_val_valid_classif_loss = val_loss_valid_classif / len(dl_val.dataset)
        avg_val_validity_loss = val_validity_total / len(dl_val.dataset)

        # print(f"[Warm-up] Epoch {epoch} → Train ELBO: {total_train_loss/len(dl_train.dataset):.4f}, "
        #       f"Val ELBO: {avg_val_loss:.4f}")
        print(f" [Warm-up] Epoch {epoch} Loss VAE: {loss_total:.4f}, "
              f"Loss Classif: {loss_classif:.4f}, "
              f"Loss Valid Classif: {loss_valid_classif:.4f} | "
              f"Val Loss VAE: {avg_val_loss:.4f}, "
              f"Val Loss Classif: {avg_val_classif_loss:.4f}, "
              f"Val Loss Valid Classif: {avg_val_valid_classif_loss:.4f}" 
                f"Val Loss Validity: {avg_val_validity_loss:.4f}")
        
        wandb.log({
            "pretrain/epoch": epoch,
            "pretrain/train_loss": loss_total,
            "pretrain/train_classif_loss": loss_classif,
            "pretrain/train_valid_classif_loss": loss_valid_classif,
            "pretrain/val_loss": avg_val_loss,
            "pretrain/val_classif_loss": avg_val_classif_loss,
            "pretrain/val_valid_classif_loss": avg_val_valid_classif_loss,
            "pretrain/val_validity_loss": avg_val_validity_loss,
        })

        stopper(avg_val_loss, cvae)
        if stopper.should_stop:
            print(f"→ Early stopping warm-up à l'époque {epoch}")
            cvae.load_state_dict(torch.load('best_model.pt'))
            break

    # Préparation des listes « globales »
    x_all = x.clone()   # (N_real, 100)
    y_all = y.clone()   # (N_real,)
    y_valid_all = y_real.clone()  # (N_real,)
    # ————— 2. Boucle d’augmentation cyclique —————
    num_cycles = 100
    samples_per_cycle = 100
    alpha_min, alpha_max = 1.1, 3  # échantillonnage autour de la moyenne
    # avant num_cycles…
    samples_per_cycle = 100      # N échantillons « large »
    n_local           = 50       # n échantillons par point invalidé
    cov_scale_local   = 0.05     # échelle (petite) de la covariance locale


    for cycle in range(1, num_cycles + 1):
        print(f"\n=== Cycle {cycle} ===")
        # print le nombre de données avant augmentation avec le nomre par classe et le nombre valide pas validité
        print(f" Avant génération : total samples = {len(x_all)}, "
              f"dont synthétiques = {is_synth_all.sum().item()}")
        print(f" Répartition y=1 : {int((y_all == 1).sum().item())} | "
              f"y=0 : {int((y_all == 0).sum().item())} | "
              f"Validés : {int((y_valid_all == 1).sum().item())} sur {len(y_valid_all)}")

        # 2.1) Encoder tout pour estimer mu_mean, cov
        # cvae.eval()
        with torch.no_grad():
            mus_batches = []
            for xb_all, yb_all, yb_valid_all in DataLoader(TensorDataset(x_all, y_all, y_valid_all), batch_size=64):
                xb_all, yb_all, yb_valid_all = xb_all.to(device), yb_all.to(device), yb_valid_all.to(device)
                mu_b, _ = cvae.encode(xb_all, yb_all, yb_valid_all)  # mu_b : (batch, 2)
                mus_batches.append(mu_b.cpu())
            mus_all = torch.cat(mus_batches, dim=0)  # (N_total, 2)
            
        # calcule le mu mean et covariance par classe
        mu_mean_per_class = list()
        cov_per_class = list()
        for class_label in np.unique(y_all):
            mask_valide = (y_all == class_label) & (y_valid_all == 1)
            mu_mean = mus_all[mask_valide].mean(dim=0)  # (2,)
            cov = torch.from_numpy(np.cov(mus_all[mask_valide].T.numpy())).float()  # covariance pour la classe
            mu_mean_per_class.append(mu_mean.to(device))  # (2,)
            cov_per_class.append(cov.to(device))  # (2,2)
        
        
        
        mu_mean = mus_all.mean(dim=0)               # (2,)
        cov = torch.from_numpy(np.cov(mus_all.T.numpy())).float()  # (2,2)

        # 2.2) Générer des z autour de la moyenne pour chaque classe
        # z_candidates = []  
        # for mu_c, cov_c in zip(mu_mean_per_class, cov_per_class):  
        #     z_cand, _ = sample_varied_gaussian(mu_c.to(device), cov_c.to(device),  
        #                                     alpha_min=alpha_min, alpha_max=alpha_max,  
        #                                     n=samples_per_cycle)  
        #     z_candidates.append(z_cand)  # (samples_per_cycle, latent_dim)
        eps    = 1e-4
        margin = 0.1

        z_candidates = []
        for class_label, (mu_c, cov_c) in enumerate(zip(mu_mean_per_class, cov_per_class)):
            # 1) régularisation et mise à l'échelle de la covariance
            mu_c = mu_c.to(device)
            cov_c = cov_c.to(device)
            cov_reg = cov_c + eps * torch.eye(cov_c.size(0), device=device)
            cov_reg = cov_reg * alpha_min  # ou alpha si vous voulez contrôler l'étendue

            # 2) MVN target
            mvn_tgt = dist.MultivariateNormal(mu_c.to(device), covariance_matrix=cov_reg)
            # 3) échantillonnage
            z_samples = mvn_tgt.sample((samples_per_cycle,))  # (n, latent_dim)

            # 4) log-prob target
            log_p_tgt = mvn_tgt.log_prob(z_samples)  # (n,)

            # 5) log-prob pour les autres classes
            log_ps_others = []
            for other_label, (mu_o, cov_o) in enumerate(zip(mu_mean_per_class, cov_per_class)):
                if other_label == class_label:
                    continue
                # print cov o reg device
                # print(f"Covariance matrix for class {other_label} on device: {cov_o.device}")
                cov_o_reg = cov_o + eps * torch.eye(cov_o.size(0), device=device)
                cov_o_reg = cov_o_reg * alpha_min
                mvn_o = dist.MultivariateNormal(mu_o.to(device), covariance_matrix=cov_o_reg)
                log_ps_others.append(mvn_o.log_prob(z_samples))

            log_ps_others = torch.stack(log_ps_others, dim=0)          # (num_classes-1, n)
            max_log_p_oth, _ = torch.max(log_ps_others, dim=0)         # (n,)

            # 6) filtrage par log-ratio
            keep1 = log_p_tgt > (max_log_p_oth + margin)         # (n,)
            # keep 2 is log target > 0.5
            keep2 = log_p_tgt > torch.tensor(0.5, device=device)  # (n,)
            keep = keep1 & keep2                                   # (n,)
            z_filtered = z_samples[keep]
            print(f"Data generated for class {class_label}: {len(z_filtered)} samples kept out of {samples_per_cycle}.")

            if len(z_filtered) > 0:
                z_candidates.append(z_filtered)

        

        x_gen_list = []
        y_gen_list = []
        y_valid_gen_list = []

        
        # 2.3) Filtrer ceux que le modèle juge valid  
        x_gen_list, y_gen_list, y_valid_gen_list = [], [], []  
        # for class_label, z_cand in enumerate(z_candidates):  
        #     # prédictions de validité  
        #     with torch.no_grad():  
        #         logits_val = cvae.cls_cons(z_cand).view(-1)  
        #         p_valid   = torch.sigmoid(logits_val)         # (samples_per_cycle,)  
        #         mask_pred = p_valid > 0.5                     # on ne garde que les z jugés valides  

        #     if mask_pred.sum() == 0:  
        #         continue  

        #     z_valid = z_cand[mask_pred]                     # (n_kept, latent_dim)  
        #     # on force yc à la vraie classe, yv à 1 pour décoder  
        #     yc = torch.full((len(z_valid),), class_label, dtype=torch.long, device=device)  
        #     yv = torch.ones_like(yc)  

        #     # 2.3a) décodage  
        #     with torch.no_grad():  
        #         x_dec = cvae.decode(z_valid, yc, yv).cpu()   # (n_kept, input_dim)  

        #     # 2.3b) oracle de validité  

        #     valid_oracle = is_valid_with_stats(x_dec, catch22_bounds, envelope_bounds)
        #     # valid_oracle = is_valid(x_dec)              # BoolTensor CPU (n_kept,)  

        #     # 2.3c) construction des exemples avec vrai label y_valid  
        #     for xi, is_val in zip(x_dec, valid_oracle):  
        #         x_gen_list.append(xi.unsqueeze(0))          # (1, input_dim)  
        #         y_gen_list.append(class_label)              # on garde la classe originale  
        #         y_valid_gen_list.append(int(is_val))        # 1 si valide, 0 sinon  
        for class_label, z_cand in enumerate(z_candidates):
            yc = torch.full((len(z_cand),), class_label, dtype=torch.long, device=device)
            yv = torch.ones_like(yc)
            with torch.no_grad():
                x_dec = cvae.decode(z_cand, yc, yv).cpu()
            valid_oracle = is_valid_with_stats(x_dec, catch22_bounds, envelope_bounds)
            for xi, is_val in zip(x_dec, valid_oracle):
                x_gen_list.append(xi.unsqueeze(0))
                y_gen_list.append(class_label)
                y_valid_gen_list.append(int(is_val))

            # 2.4) Ajout au pool  
        if x_gen_list:  
            x_gen_all       = torch.cat(x_gen_list, dim=0)                     # (N_gen, input_dim)  
            y_gen_all       = torch.tensor(y_gen_list, dtype=torch.long)       # (N_gen,)  
            y_valid_gen_all = torch.tensor(y_valid_gen_list, dtype=torch.long) # (N_gen,)  

            x_all       = torch.cat([x_all, x_gen_all], dim=0)  
            y_all       = torch.cat([y_all, y_gen_all], dim=0)  
            y_valid_all = torch.cat([y_valid_all, y_valid_gen_all], dim=0)  
            is_synth_all = torch.cat([  
                is_synth_all,  
                torch.ones(len(x_gen_all), dtype=torch.bool, device=device)  
            ], dim=0)  
        # print(f" Après génération : total samples = {len(x_all)}, dont synthétiques = {len(x_gen_all)}")
        # print(" Répartition y=1 :", int((y_all == 1).sum().item()), "| y=0 :", int((y_all == 0).sum().item()))

        # 2.5) Split train/val pour ce cycle
        ds_all = TensorDataset(x_all, y_all, y_valid_all)
        n_val = int(len(ds_all) * 0.1)
        n_train = len(ds_all) - n_val
        train_ds, val_ds = random_split(ds_all, [n_train, n_val])
        dl_train_all = DataLoader(train_ds, batch_size=16, shuffle=True)
        dl_val_all = DataLoader(val_ds, batch_size=16)

        # Nouvel EarlyStopping pour ce cycle
        stopper_cycle = EarlyStopping(patience=10, min_delta=1e-4, mode='min')
        optimizer = torch.optim.Adam(cvae.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        

        # Ré-entraînement du cVAE sur tout (x_all, y_all) avec early stopping
        for epoch in range(1, 1000):
            cvae.train()

            total_train_vae, total_train_cls, total_train_valid_cls = 0.0, 0.0, 0.0
            count_batches = 0
            for xb, yb, yb_valid in dl_train_all:
                xb, yb, yb_valid = xb.to(device), yb.to(device), yb_valid.to(device)
                optimizer.zero_grad()
                x_hat, mu, logvar, z, _, _ = cvae(xb, yb, yb_valid)
                validity_pred = V_net(xb)
                loss_validity = F.binary_cross_entropy_with_logits(validity_pred, yb_valid.float(), reduction='sum')
                # print(f"xb shape: {xb.shape}, yb shape
                loss_total, loss_cls, loss_val_cls = loss_fn(
                    xb, x_hat, mu, logvar, z, yb, yb_valid, cvae.cls_label, cvae.cls_cons)
                loss_total += 5*loss_validity
                loss_total.backward()
                optimizer.step()
                total_train_vae += loss_total.item()
                total_train_cls += loss_cls.item()
                total_train_valid_cls += loss_val_cls.item()
                count_batches += 1
            avg_train_vae = total_train_vae / count_batches
            avg_train_cls = total_train_cls / count_batches
            avg_train_valid_cls = total_train_valid_cls / count_batches
            sched.step()
            # print(f" Cycle {cycle} [Epoch {epoch}] → Loss VAE: {avg_train_vae:.4f}, Loss Classif: {avg_train_cls:.4f}")
            # Calcul de la validation loss
            cvae.eval()
            tot_val_loss = 0.0
            tot_val_cls_loss = 0.0
            tot_val_valid_cls_loss = 0.0
            with torch.no_grad():
                for xb_val, yb_val, yb_valid_val in dl_val_all:
                    xb_val, yb_val, yb_valid_val = xb_val.to(device), yb_val.to(device), yb_valid_val.to(device)
                    x_hat_val, mu_val, logvar_val, z_val, _, _ = cvae(xb_val, yb_val, yb_valid_val)
                    validity_pred_val = V_net(x_hat_val)
                    loss_validity_val = F.binary_cross_entropy_with_logits(validity_pred_val, yb_valid_val.float(), reduction='sum')
                    # print(f"xb_val shape: {xb_val
                    val_loss, val_cls_loss, val_valid_cls_loss = loss_fn(
                        xb_val, x_hat_val, mu_val, logvar_val, z_val, yb_val, yb_valid_val,
                        cvae.cls_label, cvae.cls_cons)
                    tot_val_loss += val_loss.item() 
                    tot_val_loss += loss_validity_val.item()

                    tot_val_cls_loss += val_cls_loss.item()
                    tot_val_valid_cls_loss += val_valid_cls_loss.item()
            avg_val_loss = tot_val_loss / len(dl_val_all.dataset)
            avg_val_cls_loss = tot_val_cls_loss / len(dl_val_all.dataset)
            avg_val_valid_cls_loss = tot_val_valid_cls_loss / len(dl_val_all.dataset)
            # print(f" Cycle {cycle} [Epoch {epoch}] → Val Loss: {avg_val_loss:.4f}")

           
            wandb.log({
                "cycle/epoch": epoch,
                "cycle/cycle": cycle,
                "cycle/train_loss": avg_train_vae,
                "cycle/train_classif_loss": avg_train_cls,
                "cycle/train_valid_classif_loss": avg_train_valid_cls,
                "cycle/val_loss": avg_val_loss,
                "cycle/val_classif_loss": avg_val_cls_loss,
                "cycle/val_valid_classif_loss": avg_val_valid_cls_loss,
            })

            stopper_cycle(avg_val_loss, cvae)
            if stopper_cycle.should_stop:
                # print(f"→ Early stopping cycle {cycle} à l'époque {epoch}")
                cvae.load_state_dict(torch.load('best_model.pt'))
                break

        # 2.6) Tracer la frontière latente, en distinguant base vs synthétiques
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

            # 2) Scatter des points réels / synthétiques dans le plan 2D
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))

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






    print("\n→ Pipeline C-VAE terminé.")
if __name__=='__main__':
    wandb.init(project="OMEGA")
    train_cVAE_pipeline('GunPoint')
