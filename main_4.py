
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------
# EARLY STOPPING
# ------------------------
class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-5, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = float('inf') if mode=='min' else -float('-inf')
        self.counter = 0
        self.should_stop = False

    def __call__(self, current, model):
        improved = ((self.mode=='min' and current < self.best - self.min_delta) or
                    (self.mode=='max' and current > self.best + self.min_delta))
        if improved:
            self.best = current
            self.counter = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

# ------------------------
# LOAD ECG200
# ------------------------
def load_ecg200(data_dir):
    def _load(split):
        print(os.getcwd())
        path = os.path.join(data_dir, f'ECG200_{split}.tsv')
        arr = np.loadtxt(path)
        X = arr[:,1:]
        y = arr[:,0]
        # 1 = normal, -1 = abnormal
        y = (y == 1).astype(int)
        return X, y

    X_tr, y_tr = _load('TRAIN')
    X_te, y_te = _load('TEST')
    X = np.vstack([X_tr, X_te])
    y = np.concatenate([y_tr, y_te])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

# ------------------------
# VALIDITY CHECK (ECG CONSTRAINTS)
# ------------------------
def is_valid_ecg(x, fs=100,
                 thr_factor=0.85,     # seuil à 0.3 * max(abs(x))
                 qrs_min_pts=2,      # 0.08 s à 100 Hz → 8 échantillons
                 qrs_max_pts=11,     # 0.12 s → 12 échantillons
                 st_thr=0.95):        # ±0.3σ toléré sur le segment ST
    """
    Vérifie pour chaque série x (len=96, centrée et σ-normalisée) :
    - exactement 1 pic R (max local dépassant thr_factor * max(abs_x))
    - durée QRS entre qrs_min_pts et qrs_max_pts
    - segment ST dans ±st_thr
    """

    # S'assure du format batchable
    if x.dim() == 1:
        xs = x.unsqueeze(0)
    else:
        xs = x

    mask = []
    for xi in xs:
        # 1. longueur fixe
        if xi.shape[0] != 96:
            # print(f"Invalid length {xi.shape[0]} for ECG series, expected 96.")
            mask.append(False)
            continue

        # 2. z-normalisation implicite (moyenne ≈0, σ≈1)
        x0 = xi - xi.mean()
        abs_x = x0.abs()

        # 3. détection du pic R
        peak_idx = int(torch.argmax(abs_x))
        thr = thr_factor * abs_x.max()
        # compte des maxima locaux > seuil
        # locs = sum(
        #     1 for i in range(1, 95)
        #     if x0[i] > x0[i-1] and x0[i] > x0[i+1] and abs_x[i] > thr
        # )
        # if locs != 1:
        #     print(f"Invalid number of R-peaks: {locs} found, expected 1.")
        #     mask.append(False)
        #     continue

        # 4. durée QRS en échantillons
        # on remonte/descend depuis peak jusqu’à ce que abs_x < thr
        l, r = peak_idx, peak_idx
        while l > 0   and abs_x[l] > thr: l -= 1
        while r < 95  and abs_x[r] > thr: r += 1
        dur_pts = r - l
        if not (qrs_min_pts <= dur_pts <= qrs_max_pts):
            # print(f"Invalid QRS duration: {dur_pts} pts, expected between {qrs_min_pts} and {qrs_max_pts}.")
            mask.append(False)
            continue

        # # 5. segment ST (juste après QRS) doit être proche de 0
        # st_start = peak_idx + int(0.04 * fs)    # ~4 points après R
        # st_end   = st_start + int(0.02 * fs)    # +2 points
        # if st_end > 95:
        #     print(f"Invalid ST segment range: {st_start}-{st_end} exceeds series length.")
        #     mask.append(False)
        #     continue
        # # st_mean = x0[st_start:st_end].mean().abs()
        # if st_mean > st_thr:
        #     mask.append(False)
        #     continue

        # si tout est OK
        mask.append(True)

    mask = torch.tensor(mask, dtype=torch.bool, device=x.device)
    return mask[0] if x.dim()==1 else mask


# ------------------------
# GAUSSIAN SAMPLING IN LATENT SPACE
# ------------------------

        
def sample_varied_gaussian(mu, cov, alpha_min=0.8, alpha_max=1.2, n=200):
    mvn = dist.MultivariateNormal(mu, covariance_matrix=alpha_min * cov)
    z_samples = mvn.sample((n,)).to(device)
    return z_samples, alpha_min

# ------------------------
# CVAE MODEL
# ------------------------
class CVAE(nn.Module):
    def __init__(self, input_dim=96, latent_dim=2, num_classes=2, y_emb=8):
        super().__init__()
        self.embed = nn.Embedding(num_classes, y_emb)
        self.valid_embed = nn.Embedding(2, y_emb)  # pour y=0 et y=1
        self.encoder = nn.Sequential(
            nn.Linear(input_dim+2,64), nn.ReLU(),
            nn.Linear(64,32),             nn.ReLU()
        )
        self.mu     = nn.Linear(32, latent_dim)
        self.logvar = nn.Linear(32, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim+2,32), nn.ReLU(),
            nn.Linear(32,64),                nn.ReLU(),
            nn.Linear(64,input_dim)
        )
        self.cls_ecg = nn.Sequential(
            nn.Linear(latent_dim,16), nn.ReLU(), nn.Linear(16,1)
        )
        self.cls_val = nn.Sequential(
            nn.Linear(latent_dim,16), nn.ReLU(), nn.Linear(16,1)
        )

    def encode(self,x,yc, yv):
        y_e = self.embed(yc)
        y_v_e = self.valid_embed(yv)  # pour y=0 et y=1
        # print(torch.cat([x, y_e, y_v_e], dim=1).shape)
        h  = self.encoder(torch.cat([x, y_e, y_v_e], dim=1))
        return self.mu(h), self.logvar(h)
    
    def reparam(self,mu,lv):
        std = torch.exp(0.5*lv)
        return mu + torch.randn_like(std)*std
    def decode(self,z , yc, yv):
        y_e = self.embed(yc)
        y_v_e = self.valid_embed(yv)  # pour y=0 et y=
        return self.decoder(torch.cat([z, y_e, y_v_e], dim=1))
    def forward(self,x,yc,yv):
        mu, lv = self.encode(x,yc,yv)
        z       = self.reparam(mu,lv)
        xh      = self.decode(z,yc,yv)
        le      = self.cls_ecg(z).view(-1)
        lv2     = self.cls_val(z).view(-1)
        return xh, mu, lv, z, le, lv2

# ------------------------
# LOSS FUNCTION
# ------------------------

def contrastive_loss(z, labels, margin=1.0):
        # z : (batch, latent_dim)
        # labels : (batch,) 0 ou 1
        # calcule toutes les distances inter-paquets
        pairwise_distances = torch.cdist(z, z, p=2)  # (batch, batch)
        # mask positif / négatif
        positive_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        negative_mask = ~positive_mask
        # loss² pour positives, hinge pour négatives
        positive_loss = positive_mask * pairwise_distances.pow(2)
        negative_loss = negative_mask * F.relu(margin - pairwise_distances).pow(2)
        # moyenne sur tous les termes
        return (positive_loss + negative_loss).mean()
def loss_fn(x,x_hat,mu,log_var, z, y, y_valid, classifier, validator, lambda_ecg=0.0, lambda_val=10.0 , lambda_contrastive=1.0):
    recon = F.mse_loss(x_hat,x,reduction='sum')
    kld   = -0.5*torch.sum(1+log_var-mu.pow(2)-log_var.exp())
    logits_ecg = classifier(z).view(-1)
    logits_val = validator(z).view(-1)
    l_ecg = F.binary_cross_entropy_with_logits(logits_ecg,y.float(),reduction='sum')
    l_val = F.binary_cross_entropy_with_logits(logits_val,y_valid.float(),reduction='sum')
    contrastive_loss_value = contrastive_loss(z, y)
    return recon + kld + lambda_ecg*l_ecg + lambda_val*l_val + lambda_contrastive * contrastive_loss_value, l_ecg, l_val

# ------------------------
# TRAIN & AUGMENTATION PIPELINE
# ------------------------# ------------------------
# PIPELINE D'ENTRAÎNEMENT cVAE AVEC EARLY STOPPING
# ------------------------
def train_cVAE_pipeline(data_dir):
    # ————— 1. Génération initiale & Warm-up —————
    torch.manual_seed(0)
    x, y = load_ecg200(data_dir)
    # sampler aleatoire 50 données de chaque classe
    # idx_0 = torch.where(y == 0)[0]
    # idx_1 = torch.where(y == 1)[0]
    # idx_0 = idx_0[torch.randperm(len(idx_0))[:10]]
    # idx_1 = idx_1[torch.randperm(len(idx_1))[:10]]
    # idx = torch.cat([idx_0, idx_1])
    
    # x = x[idx] # (100, 96)
    # y = y[idx] # (100,)
    # print(f'{len(x)} données initiales chargées dont {len(idx_0)} de classe 0 et {len(idx_1)} de classe 1.')
    y_real = torch.ones(len(x), dtype=torch.long)   # tous valides = 1
    # print le nombre de données x qui sont valides
    print(f"Nombre de données valides initiales : {is_valid_ecg(x).sum().item()} sur {len(x)}")
    if not is_valid_ecg(x).all():
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
    cvae = CVAE(input_dim=96, latent_dim=8, num_classes=2, y_emb=1).to(device)
    optimizer = torch.optim.Adam(cvae.parameters(), lr=1e-3)
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
            loss_total, loss_classif, loss_valid_classif = loss_fn(xb, x_hat, mu, logvar, z, yb, y_r_b, cvae.cls_ecg, cvae.cls_val, lambda_ecg=0.0, lambda_val=10.0)
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
        val_loss_total, val_loss_classif, val_loss_valid_classif = 0.0, 0.0, 0.0
        with torch.no_grad():
            for xb_val, yb_val, yb_r_val in dl_val:
                xb_val, yb_val, yb_r_val = xb_val.to(device), yb_val.to(device), yb_r_val.to(device)
                x_hat_val, mu_val, logvar_val, z_val, _, _ = cvae(xb_val, yb_val, yb_r_val)
                val_loss, val_classif_loss, val_valid_classif_loss = loss_fn(xb_val, x_hat_val, mu_val, logvar_val, z_val, yb_val, yb_r_val , cvae.cls_ecg, cvae.cls_val, lambda_ecg=0.0, lambda_val=10.0)
                val_loss_total += val_loss.item()
                val_loss_classif += val_classif_loss.item()
                val_loss_valid_classif += val_valid_classif_loss.item()
                
        avg_val_loss = val_loss_total / len(dl_val.dataset)
        avg_val_classif_loss = val_loss_classif / len(dl_val.dataset)
        avg_val_valid_classif_loss = val_loss_valid_classif / len(dl_val.dataset)

        # print(f"[Warm-up] Epoch {epoch} → Train ELBO: {total_train_loss/len(dl_train.dataset):.4f}, "
        #       f"Val ELBO: {avg_val_loss:.4f}")
        print(f" [Warm-up] Epoch {epoch} Loss VAE: {loss_total:.4f}, "
              f"Loss Classif: {loss_classif:.4f}, "
              f"Loss Valid Classif: {loss_valid_classif:.4f} | "
              f"Val Loss VAE: {avg_val_loss:.4f}, "
              f"Val Loss Classif: {avg_val_classif_loss:.4f}, "
              f"Val Loss Valid Classif: {avg_val_valid_classif_loss:.4f}")

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
    samples_per_cycle = 10
    alpha_min, alpha_max = 1.5, 3  # échantillonnage autour de la moyenne

    for cycle in range(1, num_cycles + 1):
        print(f"\n=== Cycle {cycle} ===")
        # print le nombre de données avant augmentation avec le nomre par classe et le nombre valide pas validité
        print(f" Avant génération : total samples = {len(x_all)}, "
              f"dont synthétiques = {is_synth_all.sum().item()}")
        print(f" Répartition y=1 : {int((y_all == 1).sum().item())} | "
              f"y=0 : {int((y_all == 0).sum().item())} | "
              f"Validés : {int((y_valid_all == 1).sum().item())} sur {len(y_valid_all)}")

        # 2.1) Encoder tout pour estimer mu_mean, cov
        cvae.eval()
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
            mu_mean = mus_all[y_all == class_label].mean(dim=0)  # moyenne pour la classe
            cov = torch.from_numpy(np.cov(mus_all[y_all == class_label].T.numpy())).float()  # covariance pour la classe
            mu_mean_per_class.append(mu_mean)
            cov_per_class.append(cov)
        
        
        
        mu_mean = mus_all.mean(dim=0)               # (2,)
        cov = torch.from_numpy(np.cov(mus_all.T.numpy())).float()  # (2,2)

        # 2.2) Générer des z autour de la moyenne pour chaque classe
        z_candidates = []  
        for mu_c, cov_c in zip(mu_mean_per_class, cov_per_class):  
            z_cand, _ = sample_varied_gaussian(mu_c.to(device), cov_c.to(device),  
                                            alpha_min=alpha_min, alpha_max=alpha_max,  
                                            n=samples_per_cycle)  
            z_candidates.append(z_cand)  # (samples_per_cycle, latent_dim)
        

        x_gen_list = []
        y_gen_list = []
        y_valid_gen_list = []

        
        # 2.3) Filtrer ceux que le modèle juge valid®  
        x_gen_list, y_gen_list, y_valid_gen_list = [], [], []  
        for class_label, z_cand in enumerate(z_candidates):  
            # prédictions de validité  
            with torch.no_grad():  
                logits_val = cvae.cls_val(z_cand).view(-1)  
                p_valid   = torch.sigmoid(logits_val)         # (samples_per_cycle,)  
                mask_pred = p_valid > 0.5                     # on ne garde que les z jugés valides  

            if mask_pred.sum() == 0:  
                continue  

            z_valid = z_cand[mask_pred]                     # (n_kept, latent_dim)  
            # on force yc à la vraie classe, yv à 1 pour décoder  
            yc = torch.full((len(z_valid),), class_label, dtype=torch.long, device=device)  
            yv = torch.ones_like(yc)  

            # 2.3a) décodage  
            with torch.no_grad():  
                x_dec = cvae.decode(z_valid, yc, yv).cpu()   # (n_kept, input_dim)  

            # 2.3b) oracle de validité  
            valid_oracle = is_valid_ecg(x_dec)              # BoolTensor CPU (n_kept,)  

            # 2.3c) construction des exemples avec vrai label y_valid  
            for xi, is_val in zip(x_dec, valid_oracle):  
                x_gen_list.append(xi.unsqueeze(0))          # (1, input_dim)  
                y_gen_list.append(class_label)              # on garde la classe originale  
                y_valid_gen_list.append(int(is_val))        # 1 si valide, 0 sinon  
                    

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
        dl_val_all = DataLoader(val_ds, batch_size=316)

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
                loss_total, loss_cls, loss_val_cls = loss_fn(
                    xb, x_hat, mu, logvar, z, yb, yb_valid, cvae.cls_ecg, cvae.cls_val, lambda_ecg=10.0, lambda_val=10.0)
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
                    val_loss, val_cls_loss, val_valid_cls_loss = loss_fn(
                        xb_val, x_hat_val, mu_val, logvar_val, z_val, yb_val, yb_valid_val,
                        cvae.cls_ecg, cvae.cls_val, lambda_ecg=10.0, lambda_val=10.0)
                    tot_val_loss += val_loss.item()
                    tot_val_cls_loss += val_cls_loss.item()
                    tot_val_valid_cls_loss += val_valid_cls_loss.item()
            avg_val_loss = tot_val_loss / len(dl_val_all.dataset)
            avg_val_cls_loss = tot_val_cls_loss / len(dl_val_all.dataset)
            avg_val_valid_cls_loss = tot_val_valid_cls_loss / len(dl_val_all.dataset)
            # print(f" Cycle {cycle} [Epoch {epoch}] → Val Loss: {avg_val_loss:.4f}")

           
            

            stopper_cycle(avg_val_loss, cvae)
            if stopper_cycle.should_stop:
                # print(f"→ Early stopping cycle {cycle} à l'époque {epoch}")
                cvae.load_state_dict(torch.load('best_model.pt'))
                break

        # 2.6) Tracer la frontière latente, en distinguant base vs synthétiques
        if cycle % 1 == 0 or cycle == num_cycles:
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
            # # Calcul des limites pour la grille
            # x_min, x_max = mus_all[:,0].min() - 0.2, mus_all[:,0].max() + 0.2
            # y_min, y_max = mus_all[:,1].min() - 0.2, mus_all[:,1].max() + 0.2

            # # 1) Créer une grille
            # # Résolution réduite pour la rapidité
            # # Filtrer les valides
            # mask_val = (y_valid_all == 1)
            # Z = mus_all[mask_val]          # shape (N_val,2)
            # C = labels_all[mask_val]       # 0 ou 1
            # S = is_synth_all_np[mask_val]     # False=réel, True=synth
    
            # # Grille pour la decision boundary
            # n_pts = 100
            # x_min, x_max = Z[:,0].min()-0.2, Z[:,0].max()+0.2
            # y_min, y_max = Z[:,1].min()-0.2, Z[:,1].max()+0.2
            # xx, yy = np.meshgrid(np.linspace(x_min, x_max, n_pts),
            #                     np.linspace(y_min, y_max, n_pts))
            # zz = np.stack([xx.ravel(), yy.ravel()], axis=1)

            #     # 1) calcul des deux cartes de probabilité
            # with torch.no_grad():
            #     z_grid = torch.from_numpy(zz).float().to(device)
            #     probs_class = torch.sigmoid(cvae.cls_ecg(z_grid).view(-1)).cpu().numpy().reshape(xx.shape)
            #     probs_valid = torch.sigmoid(cvae.cls_val(z_grid).view(-1)).cpu().numpy().reshape(xx.shape)

            # # 2) oracle validité
            # batch_size = 2000
            # yc_full = torch.zeros(batch_size, dtype=torch.long, device=device)
            # yv_full = torch.ones (batch_size, dtype=torch.long, device=device)

            # valid_oracle = []
            # with torch.no_grad():
            #     for i in range(0, zz.shape[0], batch_size):
            #         z_b = torch.from_numpy(zz[i:i+batch_size]).float().to(device)
            #         yc_b = yc_full[:z_b.size(0)]
            #         yv_b = yv_full[:z_b.size(0)]
            #         # on décode en supposant “valid” (yv=1)
            #         x_dec = cvae.decode(z_b, yc_b, yv_b).cpu()
            #         # on applique l’oracle is_valid_ecg
            #         valid_oracle.append(is_valid_ecg(x_dec).numpy().astype(float))

            # valid_oracle = np.concatenate(valid_oracle).reshape(xx.shape)

            # # 3) plot combiné
            # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

            # # — Plot 1 : seules les données VALIDES, frontière classe —
            # ax1.contour(xx, yy, probs_class, levels=[0.5], colors='k', linestyles='--')
            # mask_val = (valid_labels_all == 1)
            # Zv = mus_all[mask_val]; Cv = labels_all[mask_val]; Sv = is_synth_all_np[mask_val]
            # for cls in [0, 1]:
            #     for synth in [False, True]:
            #         m = (Cv == cls) & (Sv == synth)
            #         ax1.scatter(
            #             Zv[m,0], Zv[m,1],
            #             marker='o' if cls==0 else '^',
            #             color='blue' if not synth else 'red',
            #             label=f"{'réel' if not synth else 'synth'}·cls{cls}",
            #             s=30, alpha=0.7
            #         )
            # ax1.set_title("Valides seulement — Classe @0.5")
            # ax1.set_xlabel("z₁"); ax1.set_ylabel("z₂")
            # ax1.legend(fontsize='small', loc='best')

            # # — Plot 2 : toutes données, frontière validité —
            # ax2.contour(xx, yy, probs_valid, levels=[0.5], colors='k', linestyles='--')
            # ax2.contour(xx, yy, valid_oracle, levels=[0.5], colors='r', linestyles='-')
            # for synth, col in [(False,'blue'), (True,'orange'), (True,'red')]:
            #     if not synth:
            #         mask = ~is_synth_all_np
            #         lbl = "réel"
            #     else:
            #         mask = is_synth_all_np & (valid_labels_all==1) if col=='orange' else is_synth_all_np & (valid_labels_all==0)
            #         lbl = "synth-val" if col=='orange' else "synth-inv"
            #     for cls in [0,1]:
            #         m = mask & (labels_all==cls)
            #         ax2.scatter(
            #             mus_all[m,0], mus_all[m,1],
            #             marker='o' if cls==0 else '^',
            #             color=col,
            #             label=f"{lbl}·cls{cls}",
            #             s=25, alpha=0.6
            #         )
            # ax2.set_title("Toutes données — Validité @0.5")
            # ax2.set_xlabel("z₁"); ax2.set_ylabel("z₂")
            # ax2.legend(fontsize='small', loc='best')

            # plt.tight_layout()
            # plt.savefig(f"combined_plots_cycle_{cycle}.png")
            # plt.close(fig)
            from sklearn.decomposition import PCA
            # ou : from sklearn.manifold import TSNE

            # 1) Ajuster la PCA sur TOUT votre espace latent
            pca = PCA(n_components=2)
            Z2 = pca.fit_transform(mus_all)       # shape (N_total, 2)

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
            x_min, x_max = Z2[:,0].min()-0.2, Z2[:,0].max()+0.2
            y_min, y_max = Z2[:,1].min()-0.2, Z2[:,1].max()+0.2
            xx, yy = np.meshgrid(np.linspace(x_min,x_max,n_pts),
                                np.linspace(y_min,y_max,n_pts))
            grid2d = np.stack([xx.ravel(), yy.ravel()], axis=1)  # (n_pts², 2)

            # 4) Inverser la PCA pour remonter en ℝᵈ
            latent_grid = pca.inverse_transform(grid2d)          # (n_pts², latent_dim)
            z_grid = torch.from_numpy(latent_grid).float().to(device)

            # 5) Évaluer vos classifieurs sur ces z_grid
            with torch.no_grad():
                probs_class = torch.sigmoid(cvae.cls_ecg(z_grid)).cpu().numpy().reshape(xx.shape)
                probs_valid = torch.sigmoid(cvae.cls_val(z_grid)).cpu().numpy().reshape(xx.shape)

            # 6) Tracer les frontières de décision
            ax1.contour(xx, yy, probs_class, levels=[0.5], colors='k', linestyles='--')
            # 6) Oracle validité en décodant z_grid (on force yv=1)
            valid_oracle = []
            batch_size = 2000
            yc = torch.zeros(batch_size, dtype=torch.long, device=device)
            yv = torch.ones( batch_size, dtype=torch.long, device=device)
            with torch.no_grad():
                for i in range(0, latent_grid.shape[0], batch_size):
                    zb = z_grid[i:i+batch_size]
                    xy = cvae.decode(zb, yc[:zb.size(0)], yv[:zb.size(0)]).cpu()
                    valid_oracle.append(is_valid_ecg(xy).numpy().astype(float))
            valid_oracle = np.concatenate(valid_oracle).reshape(xx.shape)

            # 7) Tracer les frontières dans le plan PCA
            ax2.contour(xx, yy, probs_valid,  levels=[0.5], colors='k', linestyles='--')
            ax2.contour(xx, yy, valid_oracle, levels=[0.5], colors='r', linestyles='-')

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
            plt.savefig(f"combined_plots_cycle_{cycle}_pca.png")
            plt.close(fig)



    print("\n→ Pipeline C-VAE terminé.")
if __name__=='__main__':
    train_cVAE_pipeline('ECG200')
