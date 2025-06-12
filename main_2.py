# ------------------------
# IMPORTS & DEVICE
# ------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from itertools import cycle
torch.autograd.set_detect_anomaly(True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def plot_real_vs_generated_with_boundary_2d(
    x_real, is_synth, z_gen, y_valid, vae,
    title="Latent Space"
):
    vae.eval()
    device = next(vae.parameters()).device

    # 1) Encode réels
    with torch.no_grad():
        mu_real, _ = vae.encode(x_real.to(device))
    z_real = mu_real.cpu().numpy()  # (N, 2)
    z_gen_np = z_gen.cpu().numpy()  # (M, 2)
    valid_mask = y_valid.cpu().numpy().astype(bool)

    # 2) Grid en latent space (2D direct)
    z_all = np.vstack([z_real, z_gen_np])
    x_min, x_max = z_all[:,0].min() - 0.1, z_all[:,0].max() + 0.1
    y_min, y_max = z_all[:,1].min() - 0.1, z_all[:,1].max() + 0.1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )
    grid = np.stack([xx.ravel(), yy.ravel()]).T  # (40000, 2)

    # 3) Classifieur
    grid_torch = torch.from_numpy(grid).float().to(device)
    with torch.no_grad():
        probs = torch.sigmoid(vae.classifier(grid_torch)).cpu().numpy().reshape(xx.shape)
        
    z_real_true = z_real[is_synth == False]  # (N_valid, 2)
    z_real_gen = z_real[is_synth == True]    # (N_synth, 2)
    

    # --- Plot ---
    plt.figure(figsize=(6,6))

    

    # --- Points réels ---
    plt.scatter(z_real_true[:,0], z_real_true[:,1], s=10, alpha=0.4, label="Réels valides")
    plt.scatter(z_real_gen[:,0], z_real_gen[:,1], s=10, alpha=0.4, label="Synthétique valides")
    


   # --- Synthétiques valides (cycle courant) ---
    plt.scatter(z_gen_np[valid_mask,0], z_gen_np[valid_mask,1],
                marker="x", s=30, color='blue', label="Synth valides (cycle courant)")

    # --- Synthétiques invalides ---
    plt.scatter(z_gen_np[~valid_mask,0], z_gen_np[~valid_mask,1],
                marker="x", s=30, color='orange', label="Synth invalides (cycle courant)")

    # --- Frontière classifieur ---
    plt.contour(xx, yy, probs, levels=[0.5],
                colors='k', linestyles='--', linewidths=2)

    # --- Final ---
    plt.title(title)
    plt.xlabel("Latent dim 1")
    plt.ylabel("Latent dim 2")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f"{title}.png")
    plt.close()


def plot_real_vs_generated_with_boundary(
    x_real, z_gen, y_valid, vae,
    title="Latent Space"
):
    vae.eval()
    device = next(vae.parameters()).device

    # 1) Encode réels et récupérer mu_real: (N, d)
    with torch.no_grad():
        mu_real, _ = vae.encode(x_real.to(device))
    z_real = mu_real.cpu().numpy()
    z_gen_np = z_gen.cpu().numpy()
    valid_mask = y_valid.cpu().numpy().astype(bool)

    # 2) t-SNE sur tous les points
    all_z = np.vstack([z_real, z_gen_np])
    tsne = TSNE(n_components=2, random_state=0, perplexity=30)
    z_tsne_all = tsne.fit_transform(all_z)
    zr2 = z_tsne_all[:len(z_real)]  # (N, 2)
    zg2 = z_tsne_all[len(z_real):]  # (M, 2)

    # 3) Grid pour la frontière
    x_min, x_max = z_tsne_all[:,0].min() - 0.1, z_tsne_all[:,0].max() + 0.1
    y_min, y_max = z_tsne_all[:,1].min() - 0.1, z_tsne_all[:,1].max() + 0.1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )
    grid = np.stack([xx.ravel(), yy.ravel()]).T  # (40000, 2)

    # 4) Frontier: on entraîne un LogisticRegression sur le latent 2D (zr2 + zg2)
    from sklearn.linear_model import LogisticRegression
    X_train = np.vstack([zr2, zg2])
    y_train = np.concatenate([
        np.ones(len(zr2)),   # réels → label 1
        valid_mask.astype(int)  # synthétiques valides=1, invalides=0
    ])
    clf_2d = LogisticRegression().fit(X_train, y_train)

    # Prédire la frontière
    probs = clf_2d.predict_proba(grid)[:,1].reshape(xx.shape)

    # 5) Tracé
    plt.figure(figsize=(6,6))
    plt.scatter(zr2[:,0], zr2[:,1], s=10, alpha=0.4, label="Réels")
    plt.scatter(zg2[valid_mask,0], zg2[valid_mask,1],
                marker="x", s=30, label="Synth valides")
    plt.scatter(zg2[~valid_mask,0], zg2[~valid_mask,1],
                marker="x", s=30, label="Synth invalides")

    # Frontière p=0.5
    plt.contour(xx, yy, probs, levels=[0.5],
                colors='k', linestyles='--', linewidths=2)

    plt.title(title)
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f"{title}.png")
    plt.close()


# ------------------------
# DATA GEN + VALIDITY
# ------------------------
def generate_valid_damped_signals(n=100, L=100, dt=0.01,
                                  amp_bounds=(0.5,1.2),
                                  decay_ratio_thresh=0.8,
                                  noise_std=0.05):
    t = torch.linspace(0, dt*(L-1), L)
    signals = []
    while len(signals) < n:
        A      = torch.rand(1)*(amp_bounds[1]-amp_bounds[0]) + amp_bounds[0]
        omega0 = torch.rand(1)*5 + 5
        zeta   = torch.rand(1)*0.3
        phi    = torch.rand(1)*2*np.pi

        omega_d = omega0 * torch.sqrt(1 - zeta**2)
        x = A * torch.exp(-zeta*omega0*t) * torch.cos(omega_d*t + phi)
        x_noisy = x + torch.randn_like(x)*noise_std

        if is_valid(x_noisy.unsqueeze(0),
                    amp_max=amp_bounds[1],
                    decay_ratio_thresh=decay_ratio_thresh)[0]:
            signals.append(x_noisy)
    return torch.stack(signals)


def is_valid(x, amp_max=1.2, decay_ratio_thresh=0.8):
    x0 = x - x.mean(dim=1, keepdim=True)
    amp_vals   = x0.abs().max(dim=1).values
    mid        = x0.shape[1] // 2
    decay_vals = x0[:,mid:].abs().max(dim=1).values
    return (amp_vals < amp_max) & ((decay_vals/(amp_vals+1e-8)) < decay_ratio_thresh)

def sample_extended_gaussian(mu, cov, alphas, n=200):
    dev = mu.device
    cov    = cov.to(dev)
    alphas = alphas.to(dev)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    scaled = torch.diag(alphas * torch.sqrt(eigvals))
    eps    = torch.randn(n, len(eigvals), device=dev)
    return  mu + (eps @ scaled @ eigvecs.T)
def sample_varied_gaussian(mu, cov, alpha_min=0.8, alpha_max=1.2, n=200):
    """
    Échantillonne n vecteurs z ~ N(mu, cov) en appliquant un
    facteur alpha différent par dimension et par échantillon.
    """
    device = mu.device
    cov = cov.to(device)
    # décomposition spectrale
    eigvals, eigvecs = torch.linalg.eigh(cov)        # (d,), (d,d)
    sqrt_eig = torch.sqrt(eigvals).unsqueeze(0)      # (1, d)

    # tirage des alphas: shape (n, d)
    alphas = torch.rand(n, eigvals.size(0), device=device)
    alphas = alpha_min + (alpha_max - alpha_min) * alphas
    # print(f'alphas:{alphas}')

    # tirage eps ~ N(0, I) puis mise à l’échelle
    eps = torch.randn(n, eigvals.size(0), device=device)
    scaled = eps * alphas * sqrt_eig                  # (n, d)

    # passage retour à l’espace latent
    return mu +(scaled @ eigvecs.T), alphas

def train_vae(vae, x_all, val_frac=0.1,
              epochs=50, lr=1e-3, batch_size=32, patience=5):

    ds = TensorDataset(x_all)
    n_val = int(len(ds)*val_frac)
    train_ds, val_ds = random_split(ds, [len(ds)-n_val, n_val])
    tr = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    vl = DataLoader(val_ds,   batch_size=batch_size)

    opt   = torch.optim.Adam(vae.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
    stopper = EarlyStopping(patience=patience, mode='min')

    for ep in range(1, epochs+1):
        vae.train()
        for (xb,) in tr:
            xb = xb.to(device)
            recon, mu, logvar, *_ = vae(xb)
            rec  = F.mse_loss(recon, xb)
            kld  = -0.5*torch.mean(1+logvar-mu.pow(2)-logvar.exp())
            loss = rec + kld
            opt.zero_grad(); loss.backward(); opt.step()

        # validation
        vae.eval(); tot=0.0
        with torch.no_grad():
            for (xb,) in vl:
                xb = xb.to(device)
                recon, mu, logvar, *_ = vae(xb)
                rec  = F.mse_loss(recon, xb)
                kld  = -0.5*torch.mean(1+logvar-mu.pow(2)-logvar.exp())
                tot += (rec+kld).item()*xb.size(0)
        val_loss = tot/len(vl.dataset)

        stopper(val_loss, vae)
        if stopper.should_stop:
            print(f"→ Early stopping warm-up VAE at epoch {ep}")
            vae.load_state_dict(torch.load('best_model.pt'))
            break
        sched.step()

# ------------------------
# VAE + CLASSIFIER
# ------------------------
class VAE(nn.Module):
    def __init__(self, input_dim=100, latent_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU()
        )
        self.mu = nn.Linear(32, latent_dim)
        self.logvar = nn.Linear(32, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, input_dim)
        )
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z, self.classifier(z)

# ------------------------
# CONTRASTIVE LOSS
# ------------------------
def nt_xent_loss(z, labels, temperature=0.5, eps=1e-6):
    N = z.size(0)
    sim = (z @ z.T) / temperature
    exp_sim = torch.exp(sim)
    mask_self = torch.eye(N, device=z.device).bool()
    mask_pos = (labels.unsqueeze(1) == labels.unsqueeze(0)) & ~mask_self

    pos_sum = (exp_sim * mask_pos).sum(1)
    all_sum = (exp_sim * ~mask_self).sum(1)

    ratio = pos_sum / (all_sum + eps)
    return -torch.log(ratio).mean()

# ------------------------
# EARLY STOPPING
# ------------------------
class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-5, mode='min'):
        self.patience, self.min_delta = patience, min_delta
        self.mode = mode
        self.best = float('inf') if mode == 'min' else -float('inf')
        self.counter = 0
        self.should_stop = False

    def __call__(self, current, model):
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
# PLOT LATENT COVARIANCE
# ------------------------
def plot_latent_covariance(mu_real, title="latent_covariance"):
    mu_np = mu_real.cpu().numpy()
    cov = np.cov(mu_np.T)
    plt.figure(figsize=(5,5))
    plt.imshow(cov, cmap="viridis")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{title}.png")
    plt.close()

# ------------------------
# TRAIN JOINT (amélioré)
# ------------------------
def train_joint(vae, x_real, z_gen, y_valid,
                val_frac=0.1,
                epochs=30,
                real_bs=64,
                synth_bs=64,
                lambda_bce=1.0,
                lambda_contra=1.0,
                lambda_center=0.1,
                lambda_cov=0.1,
                temperature=0.5,
                lr=1e-3,
                patience=10):

    real_ds = TensorDataset(x_real)
    synth_ds = TensorDataset(z_gen, y_valid)

    n_val_r = int(len(real_ds) * val_frac)
    rt, rv = random_split(real_ds, [len(real_ds) - n_val_r, n_val_r])
    n_val_s = int(len(synth_ds) * val_frac)
    st, sv = random_split(synth_ds, [len(synth_ds) - n_val_s, n_val_s])

    print(f"→ Real dataset: {len(rt)} train, {len(rv)} val")
    print(f"→ Synth dataset: {len(st)} train, {len(sv)} val")

    rl = DataLoader(rt, batch_size=real_bs, shuffle=True)
    sl = DataLoader(st, batch_size=synth_bs, shuffle=True)
    vl = DataLoader(rv, batch_size=real_bs)
    vs = DataLoader(sv, batch_size=synth_bs)

    opt = torch.optim.Adam(vae.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.5)
    stopper = EarlyStopping(patience=patience, mode='min')

    for ep in range(1, epochs + 1):
        vae.train()
        sit = cycle(sl)
        for (xb_real,) in rl:
            xb_real = xb_real.to(device)
            z_b, y_b = next(sit)
            z_b, y_b = z_b.to(device), y_b.to(device)

            # --- Encode réels ---
            recon, mu_r, logvar_r, _, _ = vae(xb_real)
            loss_rec = F.mse_loss(recon, xb_real)
            loss_kld = -0.5 * torch.mean(1 + logvar_r - mu_r.pow(2) - logvar_r.exp())

            # --- Latent center ---
            latent_center = mu_r.mean(0)
            loss_center = F.mse_loss(latent_center, torch.zeros_like(latent_center))

            # --- Latent covariance ---
           # --- Isotropie de la covariance ---
            if mu_r.shape[0] > 1:
                cov_latent = torch.cov(mu_r.T)
                eye = torch.eye(cov_latent.size(0), device=device)
                loss_cov = F.mse_loss(cov_latent, eye)
            else:
                loss_cov = torch.tensor(0.0, device=device, requires_grad=True)

            # --- Classifier sur réels + synth ---
            z_cls = torch.cat([mu_r, z_b], dim=0)
            labels = torch.cat([
                torch.ones(mu_r.size(0), device=device),
                y_b
            ], dim=0)

            logits = vae.classifier(z_cls).view(-1)
            loss_bce_real = F.binary_cross_entropy_with_logits(logits, labels)

            # --- Feedback classifier explicite sur z_gen ---
            # logits_gen = vae.classifier(z_b).view(-1)
            # loss_bce_gen = F.binary_cross_entropy_with_logits(logits_gen, y_b)

            # --- Contrastive ---
            z_norm = F.normalize(z_cls, dim=1)
            counts = torch.stack([
                (labels == 0).sum(),
                (labels == 1).sum()
            ])
            if (counts >= 2).all():
                loss_contra = nt_xent_loss(z_norm, labels.long(), temperature)
            else:
                loss_contra = torch.tensor(0.0, device=device, requires_grad=True)

            # --- Total loss ---
            loss = loss_rec + loss_kld \
                   + lambda_bce * loss_bce_real \
                   + lambda_contra * loss_contra \
                #    + lambda_center * loss_center \
                #    + lambda_cov * loss_cov

            # --- Backprop ---
            opt.zero_grad()
            loss.backward()
            opt.step()

        sched.step()

        # --- Validation ---
        vae.eval()
        with torch.no_grad():
            vr, vk = 0.0, 0.0
            for (xb,) in vl:
                xb = xb.to(device)
                recon, mu, logvar, *_ = vae(xb)
                vr += F.mse_loss(recon, xb, reduction='sum').item()
                vk += (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())).item()
            vr /= len(vl.dataset)
            vk /= len(vl.dataset)

            all_logits, all_labels = [], []
            for z_b, y_b in vs:
                # print(f"→ val {len(z_b)} synth samples")
                z_b, y_b = z_b.to(device), y_b.to(device)
                all_logits.append(vae.classifier(z_b).squeeze())
                all_labels.append(y_b)
            logits = torch.cat(all_logits)
            labels = torch.cat(all_labels)
            vbce = F.binary_cross_entropy_with_logits(logits, labels)

            val_loss = vr + vk + lambda_bce * vbce

        stopper(val_loss, vae)
        if stopper.should_stop:
            # print(f"→ Early stopping joint at epoch {ep}")
            vae.load_state_dict(torch.load('best_model.pt'))
            break
        if ep % 10 == 0:
            print(f"Epoch {ep}: "
                  f"Recon Loss: {vr:.4f}, KLD: {vk:.4f}, "
                  f"BCE: {lambda_bce * vbce:.4f}, "
                  f"Total Loss: {val_loss:.4f}, "
                  f"Center: {loss_center:.4f}, "
                  f"Cov: {loss_cov:.4f}")


def main():
    torch.manual_seed(0)
    # ————— 1. Génération initiale et VAE warm-up —————
    x_real = generate_valid_damped_signals(n=1000)  # (1000, 100)
    is_synth = torch.zeros(len(x_real), dtype=torch.bool)  # tous “False” au départ

    vae = VAE(latent_dim=2).to(device)
    print("→ Warm-up VAE")
    train_vae(vae, x_real, epochs=1000, lr=1e-3, batch_size=32, patience=10)

    # Listes pour accumuler tous les z_gen et y_valid
    z_gen_all   = torch.empty((0, vae.mu.out_features))
    y_valid_all = torch.empty(0)

    # ————— 2. Boucle d’augmentation cyclique —————
    alpha_min, alpha_max = 1.2, 1.5  # Facteurs d’échelle initiaux
    for cycle in range(15):
        print(f"\n→ Cycle {cycle+1}, alpha_min={alpha_min:.2f}, alpha_max={alpha_max:.2f}")

        # 2.1) Encode x_real → mu_all, calcule centre et covariance latente
        with torch.no_grad():
            mu_all, _ = vae.encode(x_real.to(device))
            mu_mean = mu_all.mean(0)
            cov     = torch.from_numpy(np.cov(mu_all.cpu().numpy().T)).float()

        # 2.2) Génère des z_latents variés
        z_gen_this, _ = sample_varied_gaussian(
            mu_mean, cov, alpha_min=alpha_min, alpha_max=alpha_max, n=20
        )
        x_gen_this = vae.decode(z_gen_this.to(device)).detach().cpu()  # signaux décodés (100, 100)

        # 2.3) Filtre les signaux décodés valides
        y_valid = is_valid(x_gen_this)        # FloatTensor (100,), 1.0 si valide
        valid_mask = y_valid.bool()           # BoolTensor (100,)

        # 2.4) Concatène uniquement les **signaux décodés valides** à x_real
        x_valid_this = x_gen_this[valid_mask]               # (n_valid, 100)
        n_valid = x_valid_this.shape[0]
        if n_valid > 0:
            x_real = torch.cat([x_real, x_valid_this], dim=0)               # nouveau x_real
            is_synth = torch.cat([
                is_synth,
                torch.ones(n_valid, dtype=torch.bool)
            ], dim=0)  # on marque ces n_valid exemples comme “synthétiques”

        # 2.5) Met à jour les listes de latents et labels pour le classifieur
        if cycle == 0:
            z_gen_all   = z_gen_this.cpu()
            y_valid_all = y_valid.float().cpu()
        else:
            z_gen_all   = torch.cat([z_gen_all,   z_gen_this.cpu()],   dim=0)
            y_valid_all = torch.cat([y_valid_all, y_valid.float().cpu()], dim=0)

        print(f"→ Taille x_real après ajout : {len(x_real)}  (dont synth={is_synth.sum().item()})")

        # 2.6) Tracé avant entraînement joint
        plot_real_vs_generated_with_boundary_2d(x_real, is_synth, z_gen_all, y_valid_all, vae,
    title="Cycle{}_gen".format(cycle+1))

        # 2.7) Entraîne conjoitement (encodeur + décodeur + classifieur)
        train_joint(
            vae,
            x_real.to(device),
            z_gen_all.to(device),
            y_valid_all.to(device),
            epochs=1000,
            real_bs=8,
            synth_bs=8,
            lambda_bce=10,
            lambda_contra=1.0,
            lambda_center=0.1,
            lambda_cov=0.1,
            temperature=0.5,
            lr=1e-3,
            patience=10
        )

        # 2.8) Tracé après entraînement joint
        plot_real_vs_generated_with_boundary_2d(x_real, is_synth, z_gen_all, y_valid_all, vae,
    title="Cycle{}_gen_trained".format(cycle+1))

        print("→ Cycle terminé\n")

    print("→ Pipeline terminée.")


if __name__ == '__main__':
    main()
