import os
import json
import time
import torch.optim as optim
from config import DEVICE, NUM_EPOCHS, INPUT_DIM, Z_DIM, N_CLUSTERS
from model.modified_vade import WorkingVaDE_Metal
from training.trainer import WorkingTrainer
from training.utils import pretrain_autoencoder

def run_experiment_for_eta(eta, device, num_epochs=15):
    print(f"\n{'='*60}")
    print(f"ЭКСПЕРИМЕНТ с η = {eta}")
    print(f"{'='*60}")
    start_time = time.time()

    model = WorkingVaDE_Metal(input_dim=512, z_dim=32, n_clusters=12).to(device)
    model = pretrain_autoencoder(model, train_loader, device, epochs=3)

    trainer = WorkingTrainer(model, device, eta=eta)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    history = {'train_loss': [], 'train_entropy': [], 'val_nmi': [], 'val_ari': [], 'val_entropy': []}

    for epoch in range(num_epochs):
        train_metrics = trainer.train_epoch(train_loader, optimizer, epoch, temperature=0.3)
        if epoch % 2 == 0 or epoch == num_epochs - 1:
            val_metrics = trainer.evaluate(val_loader, val_labels, temperature=0.3)
            history['train_loss'].append(train_metrics['loss'])
            history['train_entropy'].append(train_metrics['entropy'])
            history['val_nmi'].append(val_metrics['nmi'])
            history['val_ari'].append(val_metrics['ari'])
            history['val_entropy'].append(val_metrics['entropy'])
            if epoch % 5 == 0 or epoch < 3:
                print(f"Epoch {epoch+1:3d}: "
                      f"loss={train_metrics['loss']:.1f}, "
                      f"NMI={val_metrics['nmi']:.3f}, "
                      f"H(γ)={val_metrics['entropy']:.3f}")

    final_metrics = trainer.evaluate(val_loader, val_labels, temperature=0.3)
    elapsed_time = time.time() - start_time
    print(f"\nРезультаты для η={eta}:")
    print(f"  NMI: {final_metrics['nmi']:.4f}")
    print(f"  ARI: {final_metrics['ari']:.4f}")
    print(f"  H(γ): {final_metrics['entropy']:.4f}")

    return {
        'model': model,
        'history': history,
        'final_metrics': final_metrics,
        'elapsed_time': elapsed_time
    }