# ------------------------
# IMPORTS & DEVICE
# ------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ------------------------
# EARLY STOPPING CLASS
# ------------------------
class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-5, mode='min'):
        """
        patience: nombre d'époques sans amélioration avant arrêt
        min_delta: amélioration minimale considérée
        mode: 'min' pour val_loss (on veut minimiser), 'max' pour métrique à maximiser
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = float('inf') if mode == 'min' else -float('inf')
        self.counter = 0
        self.should_stop = False

    def __call__(self, current, model):
        """
        current: valeur courante du critère (ex: validation loss)
        model: modèle dont on sauve l'état si amélioration
        """
        improved = (
            (self.mode == 'min' and current < self.best - self.min_delta) or
            (self.mode == 'max' and current > self.best + self.min_delta)
        )
        if improved:
            self.best = current
            self.counter = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


# ------------------------
# DATA GENERATION & VALIDATION
# ------------------------
def is_valid(x, amp_max=1.2, decay_ratio_thresh=0.8):
    """
    Vérifie si un signal x (shape [batch, L]) est « valide » :
    - L'amplitude maximale est inférieure à amp_max
    - Le ratio de l’amplitude située dans la moitié finale < decay_ratio_thresh
    """
    x0 = x - x.mean(dim=1, keepdim=True)
    amp_vals = x0.abs().max(dim=1).values
    mid = x0.shape[1] // 2
    decay_vals = x0[:, mid:].abs().max(dim=1).values
    return (amp_vals < amp_max) & ((decay_vals / (amp_vals + 1e-8)) < decay_ratio_thresh)

def generate_valid_damped_signals(n=100, L=100, dt=0.01,
                                  amp_bounds=(0.5, 1.2),
                                  decay_ratio_thresh=0.8,
                                  noise_std=0.05):
    """
    Génère n signaux amortis de longueur L avec bruit,
    et ne conserve que ceux validés par is_valid.
    """
    t = torch.linspace(0, dt * (L - 1), L)
    signals = []
    while len(signals) < n:
        A = torch.rand(1) * (amp_bounds[1] - amp_bounds[0]) + amp_bounds[0]
        omega0 = torch.rand(1) * 5 + 5
        zeta = torch.rand(1) * 0.3
        phi = torch.rand(1) * 2 * np.pi

        omega_d = omega0 * torch.sqrt(1 - zeta**2)
        x = A * torch.exp(-zeta * omega0 * t) * torch.cos(omega_d * t + phi)
        x_noisy = x + torch.randn_like(x) * noise_std

        if is_valid(x_noisy.unsqueeze(0),
                    amp_max=amp_bounds[1],
                    decay_ratio_thresh=decay_ratio_thresh)[0]:
            signals.append(x_noisy)
    return torch.stack(signals)  # shape (n, L)

def sample_varied_gaussian(mu, cov, alpha_min=0.8, alpha_max=1.2, n=200):
    """
    Échantillonne n vecteurs z ~ N(mu, cov) en appliquant un facteur alpha 
    différent par dimension et par échantillon.
    Retourne :
      - z_samples : tensor (n, latent_dim)
      - alphas : tensor (n, latent_dim)
    """
    device = mu.device
    cov = cov.to(device)
    eigvals, eigvecs = torch.linalg.eigh(cov)       # (d,), (d,d)
    sqrt_eig = torch.sqrt(eigvals).unsqueeze(0)      # (1, d)

    alphas = torch.rand(n, eigvals.size(0), device=device)
    alphas = alpha_min + (alpha_max - alpha_min) * alphas

    eps = torch.randn(n, eigvals.size(0), device=device)
    scaled = eps * alphas * sqrt_eig                 # (n, d)

    return mu + (scaled @ eigvecs.T), alphas


# ------------------------
# CVAE CLASS
# ------------------------
class CVAE(nn.Module):
    def __init__(self, input_dim=100, latent_dim=2, num_classes=2, y_embed_dim=8):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # Embedding du label y → vecteur de dimension `y_embed_dim`
        self.embed_y = nn.Embedding(num_classes, y_embed_dim)

        # --- ENCODEUR ---
        # On concatène x (input_dim) + embedding(y) (y_embed_dim) → couche 64 → 32
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + y_embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.mu     = nn.Linear(32, latent_dim)
        self.logvar = nn.Linear(32, latent_dim)

        # --- DÉCODEUR ---
        # On concatène z (latent_dim) + embedding(y) (y_embed_dim) → 32 → 64 → input_dim
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + y_embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

        # Classifieur z → label (optionnel, pour monitoring)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def encode(self, x, y):
        """
        Encode (x, y) en mu, logvar.
        x : tensor (batch_size, input_dim)
        y : tensor (batch_size,) entiers {0,1}
        """
        y_emb = self.embed_y(y)                     # (batch_size, y_embed_dim)
        xy = torch.cat([x, y_emb], dim=1)           # (batch_size, input_dim + y_embed_dim)
        h = self.encoder(xy)                        # (batch_size, 32)
        return self.mu(h), self.logvar(h)           # chacun (batch_size, latent_dim)

    def reparameterize(self, mu, logvar):
        """
        Applique le reparameterization trick pour échantillonner z.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        """
        Décode (z, y) → x_hat.
        z : tensor (batch_size, latent_dim)
        y : tensor (batch_size,)
        """
        y_emb = self.embed_y(y)                     # (batch_size, y_embed_dim)
        zy = torch.cat([z, y_emb], dim=1)           # (batch_size, latent_dim + y_embed_dim)
        return self.decoder(zy)                     # (batch_size, input_dim)

    def forward(self, x, y):
        """
        Passe complète : encode → reparam → decode. Renvoie aussi logits du classifieur.
        """
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, y)
        logits_cls = self.classifier(z)             # (batch_size, 1)
        return x_hat, mu, logvar, z, logits_cls


# ------------------------
# LOSS ELBO CONDITIONNEL
# ------------------------
def loss_cVAE(x, x_hat, mu, logvar, z, y, classifier, lambda_cls=1.0):
    """
    - x, x_hat, mu, logvar : comme avant pour ELBO
    - z : le code latent (batch_size, latent_dim)
    - y : labels {0,1} (batch_size,)
    - classifier : nn.Module retournant un logit pour z
    - lambda_cls : pondération de la loss de classification
    """

    # (1) Reconstruction + KL (ELBO)
    recon_loss = F.mse_loss(x_hat, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss_vae = recon_loss + kld

    # (2) Classification latente
    # On prédit y_hat = sigmoid(classifier(z))
    logits = classifier(z).view(-1)
    loss_cls = F.binary_cross_entropy_with_logits(logits, y.float(), reduction='sum')

    return loss_vae + lambda_cls * loss_cls, loss_vae, loss_cls



# ------------------------
# PIPELINE D'ENTRAÎNEMENT cVAE AVEC EARLY STOPPING
# ------------------------
def train_cVAE_pipeline():
    # ————— 1. Génération initiale & Warm-up —————
    torch.manual_seed(0)
    x_real = generate_valid_damped_signals(n=100)       # (1000, 100)
    y_real = torch.ones(len(x_real), dtype=torch.long)   # tous valides = 1
    n_real = x_real.shape[0]


    # Création du dataset et split train/val pour le warm-up
    ds_real = TensorDataset(x_real, y_real)
    n_val = int(len(ds_real) * 0.1)
    n_train = len(ds_real) - n_val
    train_ds, val_ds = random_split(ds_real, [n_train, n_val])
    dl_train = DataLoader(train_ds, batch_size=32, shuffle=True)
    dl_val = DataLoader(val_ds, batch_size=32)

    # Instanciation du cVAE et de l'EarlyStopping pour le warm-up
    cvae = CVAE(input_dim=100, latent_dim=2, num_classes=2, y_embed_dim=2).to(device)
    optimizer = torch.optim.Adam(cvae.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    stopper = EarlyStopping(patience=10, min_delta=1e-4, mode='min')

    # Warm-up (max 100 époques, arrêt précoce possible)
    for epoch in range(1, 100):
        cvae.train()
        total_train_loss = 0.0
        for xb, yb in dl_train:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            x_hat, mu, logvar, z, _ = cvae(xb, yb)
            loss_total, loss_vae, loss_cls = loss_cVAE(
            xb, x_hat, mu, logvar, z, yb, cvae.classifier, lambda_cls=10.0)
            loss_total.backward()

            optimizer.step()
            total_train_loss += loss_total.item()
        sched.step()

        # Calcul de la validation loss
        cvae.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for xb_val, yb_val in dl_val:
                xb_val, yb_val = xb_val.to(device), yb_val.to(device)
                x_hat_val, mu_val, logvar_val, z_val, _ = cvae(xb_val, yb_val)
                val_loss, _, _ = loss_cVAE(
                xb_val, x_hat_val, mu_val, logvar_val, z_val, yb_val, cvae.classifier, lambda_cls=10.0)
                total_val_loss += val_loss.item()
        avg_val_loss = total_val_loss / len(dl_val.dataset)

        # print(f"[Warm-up] Epoch {epoch} → Train ELBO: {total_train_loss/len(dl_train.dataset):.4f}, "
        #       f"Val ELBO: {avg_val_loss:.4f}")
        print(f" [Warm-up] Epoch {epoch} Loss VAE: {loss_vae.item():.4f}, Loss Classif: {loss_cls.item():.4f}")

        stopper(avg_val_loss, cvae)
        if stopper.should_stop:
            print(f"→ Early stopping warm-up à l'époque {epoch}")
            cvae.load_state_dict(torch.load('best_model.pt'))
            break

    # Préparation des listes « globales »
    x_all = x_real.clone()   # (N_real, 100)
    y_all = y_real.clone()   # (N_real,)

    # ————— 2. Boucle d’augmentation cyclique —————
    num_cycles = 100
    samples_per_cycle = 10
    alpha_min, alpha_max = 1.0, 1.5

    for cycle in range(1, num_cycles + 1):
        print(f"\n=== Cycle {cycle} ===")

        # 2.1) Encoder tout pour estimer mu_mean, cov
        cvae.eval()
        with torch.no_grad():
            mus_batches = []
            for xb_all, yb_all in DataLoader(TensorDataset(x_all, y_all), batch_size=64):
                xb_all, yb_all = xb_all.to(device), yb_all.to(device)
                mu_b, _ = cvae.encode(xb_all, yb_all)
                mus_batches.append(mu_b.cpu())
            mus_all = torch.cat(mus_batches, dim=0)  # (N_total, 2)
        mu_mean = mus_all.mean(dim=0)               # (2,)
        cov = torch.from_numpy(np.cov(mus_all.T.numpy())).float()  # (2,2)

        # 2.2) Générer des z autour de la moyenne
        z_candidates, _ = sample_varied_gaussian(
            mu_mean.to(device), cov.to(device),
            alpha_min=alpha_min, alpha_max=alpha_max,
            n=samples_per_cycle
        )  # (samples_per_cycle, 2)

        x_gen_list = []
        y_gen_list = []

        # 2.3) Décode forcé y=1 puis y=0
        for forced_y in [1, 0]:
            forced_y_batch = torch.full((samples_per_cycle,), forced_y, dtype=torch.long)
            with torch.no_grad():
                x_decoded = cvae.decode(z_candidates.to(device), forced_y_batch.to(device))
                x_decoded = x_decoded.cpu()  # (samples_per_cycle, 100)

            valid_mask = is_valid(x_decoded)  # (samples_per_cycle,)

            for i in range(samples_per_cycle):
                x_i = x_decoded[i].unsqueeze(0)  # (1, 100)
                is_valid_i = valid_mask[i].item()
                if forced_y == 1:
                    y_i = 1 if is_valid_i else 0
                else:  # forced_y == 0
                    y_i = 0 if not is_valid_i else 1
                x_gen_list.append(x_i)
                y_gen_list.append(y_i)

        # 2.4) Concaténation aux données globales
        x_gen_all = torch.cat(x_gen_list, dim=0)             # (2 * samples_per_cycle, 100)
        y_gen_all = torch.tensor(y_gen_list, dtype=torch.long)  # (2 * samples_per_cycle,)

        x_all = torch.cat([x_all, x_gen_all], dim=0)
        y_all = torch.cat([y_all, y_gen_all], dim=0)

        # print(f" Après génération : total samples = {len(x_all)}, dont synthétiques = {len(x_gen_all)}")
        # print(" Répartition y=1 :", int((y_all == 1).sum().item()), "| y=0 :", int((y_all == 0).sum().item()))

        # 2.5) Split train/val pour ce cycle
        ds_all = TensorDataset(x_all, y_all)
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
            total_train_vae, total_train_cls = 0.0, 0.0
            count_batches = 0
            for xb, yb in dl_train_all:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                x_hat, mu, logvar, z, _ = cvae(xb, yb)
                loss_total, loss_vae, loss_cls = loss_cVAE(
                xb, x_hat, mu, logvar, z, yb, cvae.classifier, lambda_cls=10.0)
                loss_total.backward()
                optimizer.step()
                total_train_vae += loss_vae.item()
                total_train_cls += loss_cls.item()
                count_batches += 1
            avg_train_vae = total_train_vae / count_batches
            avg_train_cls = total_train_cls / count_batches
            sched.step()
            # print(f" Cycle {cycle} [Epoch {epoch}] → Loss VAE: {avg_train_vae:.4f}, Loss Classif: {avg_train_cls:.4f}")
            # Calcul de la validation loss
            cvae.eval()
            tot_val_loss = 0.0
            with torch.no_grad():
                for xb_val, yb_val in dl_val_all:
                    xb_val, yb_val = xb_val.to(device), yb_val.to(device)
                    x_hat_val, mu_val, logvar_val, z_val, _ = cvae(xb_val, yb_val)
                    val_loss, _, _ = loss_cVAE(
                    xb_val, x_hat_val, mu_val, logvar_val, z_val, yb_val, cvae.classifier, lambda_cls=10.0)
                    tot_val_loss += val_loss.item()
            avg_val_loss = tot_val_loss / len(dl_val_all.dataset)
            # print(f" Cycle {cycle} [Epoch {epoch}] → Val Loss: {avg_val_loss:.4f}")

           
            

            stopper_cycle(avg_val_loss, cvae)
            if stopper_cycle.should_stop:
                print(f"→ Early stopping cycle {cycle} à l'époque {epoch}")
                cvae.load_state_dict(torch.load('best_model.pt'))
                break

        # 2.6) Tracer la frontière latente, en distinguant base vs synthétiques
        if cycle % 1 == 0 or cycle == num_cycles:
            cvae.eval()
            with torch.no_grad():
                mus_batches = []
                labels_batches = []
                for xb, yb in DataLoader(TensorDataset(x_all, y_all), batch_size=64):
                    xb, yb = xb.to(device), yb.to(device)
                    mu_b, _ = cvae.encode(xb, yb)   # mu_b : (batch, 2)
                    mus_batches.append(mu_b.cpu().numpy())
                    labels_batches.append(yb.cpu().numpy())
                mus_all = np.vstack(mus_batches)       # (N_total, 2)
                labels_all = np.concatenate(labels_batches)  # (N_total,)

            # Calcul des limites pour la grille
            x_min, x_max = mus_all[:,0].min() - 0.2, mus_all[:,0].max() + 0.2
            y_min, y_max = mus_all[:,1].min() - 0.2, mus_all[:,1].max() + 0.2

            # 1) Créer une grille
            n_pts = 500
            xx, yy = np.meshgrid(
                np.linspace(x_min, x_max, n_pts),
                np.linspace(y_min, y_max, n_pts)
            )
            zz = np.stack([xx.ravel(), yy.ravel()], axis=1)  # (n_pts^2, 2)

            # 2) Passer la grille dans le classifieur pour obtenir proba(z)
            with torch.no_grad():
                z_grid = torch.from_numpy(zz).float().to(device)        # (n_pts^2, 2)
                logits_grid = cvae.classifier(z_grid).view(-1)          # (n_pts^2,)
                probs_grid = torch.sigmoid(logits_grid).cpu().numpy()   # (n_pts^2,)

            # 3) Reshape pour le contour
            probs_grid = probs_grid.reshape(xx.shape)  # (n_pts, n_pts)

            # 4) Tracé
            plt.figure(figsize=(6,6))
            # a) fond coloré selon p(y=1|z)
            plt.contourf(xx, yy, probs_grid, levels=50, cmap='RdBu_r', alpha=0.6)

            # b) ligne de décision p=0.5
            plt.contour(xx, yy, probs_grid, levels=[0.5], colors='k',
                        linestyles='--', linewidths=1.5)

            # --- Séparation base vs synthétiques ---
            # indices des données de base
            real_idx = np.arange(n_real)

            # indices des synthétiques : ceux après n_real
            synth_idx = np.arange(n_real, mus_all.shape[0])

            # parmi les synthétiques, distinguer valide (label==1) / non valide (label==0)
            synth_labels = labels_all[n_real:]                # shape = (N_synth,)
            synth_valid_mask = (synth_labels == 1)
            synth_invalid_mask = (synth_labels == 0)

            # indices réels à tracer
            idx_real = real_idx

            # indices synthétiques valides
            idx_synth_valid = synth_idx[synth_valid_mask]

            # indices synthétiques non valides
            idx_synth_invalid = synth_idx[synth_invalid_mask]

            # tracé des points
            plt.scatter(mus_all[idx_real, 0], mus_all[idx_real, 1],
                        c='tab:blue',   s=10, alpha=0.6, marker='o',
                        label='données de base')

            plt.scatter(mus_all[idx_synth_valid, 0], mus_all[idx_synth_valid, 1],
                        c='tab:blue', marker='x', s=30, alpha=0.8,
                        label='synthétique valide')

            plt.scatter(mus_all[idx_synth_invalid, 0], mus_all[idx_synth_invalid, 1],
                        c='tab:orange', marker='x', s=30, alpha=0.8,
                        label='synthétique non valide')

            plt.legend(loc='upper right')
            plt.title(f"Frontière latente apprise (cycle {cycle})")
            plt.xlabel("z₁")
            plt.ylabel("z₂")
            plt.tight_layout()
            plt.savefig(f"cycle_{cycle}.png")
            
            # 1. Recalculer tous les mu(x, y) pour x_all
            cvae.eval()
            with torch.no_grad():
                mus_batches = []
                for xb, yb in DataLoader(TensorDataset(x_all, y_all), batch_size=64):
                    xb, yb = xb.to(device), yb.to(device)
                    mu_b, _ = cvae.encode(xb, yb)
                    mus_batches.append(mu_b.cpu().numpy())
                mus_all = np.vstack(mus_batches)  # (N_total, 2)

            # 2. Définir la grille couvrant l'espace latent
            x_min, x_max = mus_all[:, 0].min() - 0.2, mus_all[:, 0].max() + 0.2
            y_min, y_max = mus_all[:, 1].min() - 0.2, mus_all[:, 1].max() + 0.2
            n_pts = 200
            xx, yy = np.meshgrid(
                np.linspace(x_min, x_max, n_pts),
                np.linspace(y_min, y_max, n_pts)
            )
            zz = np.stack([xx.ravel(), yy.ravel()], axis=1)  # (n_pts^2, 2)

            # 3. Calculer la probabilité apprise p(y=1 | z) via le classifieur latent
            with torch.no_grad():
                z_grid = torch.from_numpy(zz).float().to(device)       # (n_pts^2, 2)
                logits_grid = cvae.classifier(z_grid).view(-1)         # (n_pts^2,)
                learned_probs = torch.sigmoid(logits_grid).cpu().numpy().reshape(xx.shape)  # (200,200)

            # 4. Calculer la "frontière réelle" : pour chaque z, décoder avec y=1, puis appliquer is_valid
            batch_size = 500
            valid_oracle = []
            with torch.no_grad():
                for i in range(0, zz.shape[0], batch_size):
                    z_batch = torch.from_numpy(zz[i:i+batch_size]).float().to(device)
                    y1 = torch.ones(z_batch.shape[0], dtype=torch.long).to(device)
                    x_decoded = cvae.decode(z_batch, y1).cpu()  # (batch_size, input_dim)
                    valid = is_valid(x_decoded)  # tensor(bool) (batch_size,)
                    valid_oracle.append(valid.numpy().astype(float))

            valid_oracle = np.concatenate(valid_oracle).reshape(xx.shape)  # (200,200)

            # 5. Tracer les deux frontières et les points latents
            plt.figure(figsize=(6, 6))

            # Frontière apprise (p=0.5)
            plt.contour(
                xx, yy, learned_probs,
                levels=[0.5], colors='k', linestyles='--', linewidths=1.5
            )

            # Frontière "réelle" (is_valid = 0.5)
            plt.contour(
                xx, yy, valid_oracle,
                levels=[0.5], colors='r', linestyles='-', linewidths=1.5
            )

            # Points latents : valides en bleu, invalides en orange
            mus_valid = mus_all[y_all.cpu().numpy() == 1]
            mus_invalid = mus_all[y_all.cpu().numpy() == 0]
            plt.scatter(mus_valid[:, 0], mus_valid[:, 1],
                        c='tab:blue', marker='o', s=10, alpha=0.6, label='valide (y=1)')
            plt.scatter(mus_invalid[:, 0], mus_invalid[:, 1],
                        c='tab:orange', marker='o', s=10, alpha=0.6, label='non-valide (y=0)')

            plt.legend(loc='upper right', fontsize='small')
            plt.title("Comparaison : frontière apprise (----) vs frontière réelle (—)")
            plt.xlabel("z₁")
            plt.ylabel("z₂")
            plt.tight_layout()
            plt.savefig(f"frontiere_comparaison_cycle_{cycle}.png")

    print("\n→ Pipeline C-VAE terminé.")


if __name__ == '__main__':
    train_cVAE_pipeline()
