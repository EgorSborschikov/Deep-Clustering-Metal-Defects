import torch
import torch.nn.functional as F
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import numpy as np
from tqdm import tqdm

class WorkingTrainer:
    def __init__(self, model, device, eta=0.0):
        self.model = model
        self.device = device
        self.eta = eta

    def compute_loss(self, x, x_recon, z, gamma, mu, logvar):
        B = x.shape[0]
        recon_loss = F.mse_loss(x_recon, x, reduction='sum') / B
        log_q_z = -0.5 * torch.sum(
            logvar + (z - mu) ** 2 / torch.exp(logvar) + np.log(2 * np.pi),
            dim=1
        )
        log_p_z = self.model.compute_gmm_prob(z)
        kl_loss = torch.mean(log_q_z - log_p_z)
        expected_center = torch.matmul(gamma, self.model.gmm_mu)
        center_loss = F.mse_loss(z, expected_center, reduction='mean') * 5.0
        entropy = -torch.sum(gamma * torch.log(gamma + 1e-10), dim=1).mean()
        entropy_loss = -self.eta * entropy
        total_loss = recon_loss + 20.0 * kl_loss + center_loss + entropy_loss
        confidence = gamma.max(dim=1)[0].mean()
        return total_loss, {
            'total': total_loss.item(),
            'recon': recon_loss.item(),
            'kl': kl_loss.item(),
            'center': center_loss.item(),
            'entropy': entropy.item(),
            'entropy_loss': entropy_loss.item(),
            'confidence': confidence.item()
        }

    def train_epoch(self, train_loader, optimizer, epoch, temperature=0.3):
        self.model.train()
        epoch_loss = 0
        epoch_entropy = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1:3d}')
        for batch_idx, x in enumerate(pbar):
            x = x.to(self.device)
            optimizer.zero_grad()
            x_recon, z, gamma, mu, logvar = self.model(x, temperature)
            loss, loss_dict = self.compute_loss(x, x_recon, z, gamma, mu, logvar)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += loss_dict['total']
            epoch_entropy += loss_dict['entropy']
            if batch_idx % 20 == 0:
                pbar.set_postfix({
                    'loss': f"{loss_dict['total']:.1f}",
                    'H(γ)': f"{loss_dict['entropy']:.2f}",
                    'η': f"{self.eta:.2f}"
                })
        return {'loss': epoch_loss / len(train_loader), 'entropy': epoch_entropy / len(train_loader)}

    def evaluate(self, data_loader, true_labels, temperature=0.3):
        self.model.eval()
        all_gamma = []
        with torch.no_grad():
            for x in data_loader:
                x = x.to(self.device)
                _, _, gamma, _, _ = self.model(x, temperature)
                all_gamma.append(gamma.cpu().numpy())
        all_gamma = np.vstack(all_gamma)
        pred_labels = np.argmax(all_gamma, axis=1)
        nmi = normalized_mutual_info_score(true_labels, pred_labels)
        ari = adjusted_rand_score(true_labels, pred_labels)
        entropy = -np.sum(all_gamma * np.log(all_gamma + 1e-10), axis=1).mean()
        confidence = all_gamma.max(axis=1).mean()
        return {
            'nmi': nmi,
            'ari': ari,
            'entropy': entropy,
            'confidence': confidence,
            'gamma': all_gamma,
            'pred_labels': pred_labels
        }