
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
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
        self.cls_label = nn.Sequential(
            nn.Linear(latent_dim,16), nn.ReLU(), nn.Linear(16,1)
        )
        self.cls_cons = nn.Sequential(
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
        le      = self.cls_label(z).view(-1)
        lv2     = self.cls_cons(z).view(-1)
        return xh, mu, lv, z, le, lv2
    
class ValidityNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim,32), nn.ReLU(),
            nn.Linear(32,1)
        )
    def forward(self,x):
        return self.net(x).view(-1)