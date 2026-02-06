import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class WorkingVaDE_Metal(nn.Module):
    def __init__(self, input_dim=512, z_dim=32, n_clusters=12):
        super().__init__()
        self.z_dim = z_dim
        self.n_clusters = n_clusters

        # Энкодер
        self.encoder = nn.Sequential(      
            nn.Linear(input_dim, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(500, z_dim)
        self.fc_logvar = nn.Linear(500, z_dim)
        self.fc_gamma = nn.Linear(500, n_clusters)

        # Декодер (восстанавливает эмбеддинги)
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

        # GMM prior
        centers = torch.randn(n_clusters, z_dim) * 3.0
        self.register_buffer('gmm_mu', centers)
        self.register_buffer('gmm_logvar', torch.ones(n_clusters, z_dim) * np.log(0.3))
        self.register_buffer('gmm_pi', torch.ones(n_clusters) / n_clusters)

    def encode(self, x, temperature=0.3):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), min=-4, max=3)
        gamma_logits = self.fc_gamma(h)
        gamma = F.softmax(gamma_logits / temperature, dim=-1)
        return mu, logvar, gamma

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, temperature=0.3):
        mu, logvar, gamma = self.encode(x, temperature)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, z, gamma, mu, logvar

    def compute_gmm_prob(self, z):
        B, D = z.shape
        K = self.n_clusters
        z_exp = z.unsqueeze(1)
        mu_exp = self.gmm_mu.unsqueeze(0)
        logvar_exp = self.gmm_logvar.unsqueeze(0)
        log_prob = -0.5 * (
            D * np.log(2 * np.pi) +
            logvar_exp.sum(-1) +
            ((z_exp - mu_exp) ** 2 / torch.exp(logvar_exp)).sum(-1)
        )
        log_prob = torch.log(self.gmm_pi + 1e-10) + log_prob
        return torch.logsumexp(log_prob, dim=1)