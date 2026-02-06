import torch.optim as optim
import torch.function as F
from tqdm import tqdm

def pretrain_autoencoder(model, train_loader, device, epochs=3):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print("Предобучение автоэнкодера...")
    for epoch in range(epochs):
        total_loss = 0
        for x in tqdm(train_loader, desc=f"Предобучение {epoch+1}/{epochs}"):
            x = x.to(device)
            optimizer.zero_grad()
            mu, logvar, gamma = model.encode(x, temperature=1.0)
            z = model.reparameterize(mu, logvar)
            x_recon = model.decode(z)
            loss = F.mse_loss(x_recon, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: средний loss = {total_loss/len(train_loader):.4f}")
    return model
