
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
    
def loss_fn(x,x_hat,mu,log_var, z, y, y_valid, classifier, validator, lambda_label=1.0, lambda_cons=1.0 , lambda_contrastive=1.0):
    recon = F.mse_loss(x_hat,x,reduction='sum')
    kld   = -0.5*torch.sum(1+log_var-mu.pow(2)-log_var.exp())
    logits_label = classifier(z).view(-1)
    logits_cons = validator(z).view(-1)
    l_label = F.binary_cross_entropy_with_logits(logits_label, y.float(),reduction='sum')
    l_cons = F.binary_cross_entropy_with_logits(logits_cons, y_valid.float(),reduction='sum')
    contrastive_loss_value = contrastive_loss(z, y)
    constrative_loss_valid = contrastive_loss(z, y_valid)
    return 2*recon + 0.5*kld + lambda_label*l_label + lambda_cons*l_cons + lambda_contrastive * contrastive_loss_value + lambda_cons * constrative_loss_valid, l_label, l_cons
