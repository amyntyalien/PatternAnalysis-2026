

"""
A simple CNN for colourising images, using the BSD500 dataset.
The model takes in a grayscale image (L channel) and predicts the ab channels.
The model is trained on the training set, validated on the validation set, and tested on the test set.
The model is trained using the Adam optimizer and the MSE loss function.
The model is evaluated using the PSNR and SSIM metrics.
The model is trained for 100 epochs, with a batch size of 16, and a learning rate of 1e-4.
The model is saved after every epoch, and the best model is saved based on the validation loss.

(thanks autocomplete for writing this docstring)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from dataset import get_dataloaders, denormalise_L, denormalise_ab, lab_to_rgb

# CNN model

class ColouriseCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),  # Output in range [-1, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



def train_model(model, train_loader, val_loader, num_epochs=100, lr = 1e-4, device='cuda'):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {'train_loss': [], 'val_loss': [], 'val_psnr': [], 'val_ssim': []}
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for L, ab in train_loader:
            L, ab = L.to(device), ab.to(device)
            optimizer.zero_grad()
            ab_pred = model(L)
            loss = criterion(ab_pred, ab)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * L.size(0)

        train_loss /= len(train_loader.dataset)
        history['train_loss'].append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        val_psnr = 0.0
        val_ssim = 0.0
        with torch.no_grad():
            for L, ab in val_loader:
                L, ab = L.to(device), ab.to(device)
                ab_pred = model(L)
                loss = criterion(ab_pred, ab)
                val_loss += loss.item() * L.size(0)

                # Calculate PSNR and SSIM
                L_denorm = denormalise_L(L.cpu().numpy())
                ab_denorm = denormalise_ab(ab.cpu().numpy())
                ab_pred_denorm = denormalise_ab(ab_pred.cpu().numpy())

                for i in range(L.size(0)):
                    lab_true = np.concatenate([L_denorm[i], ab_denorm[i]], axis=0).transpose(1, 2, 0)
                    lab_pred = np.concatenate([L_denorm[i], ab_pred_denorm[i]], axis=0).transpose(1, 2, 0)
                    
                    rgb_true = lab_to_rgb(lab_true[..., 0], lab_true[..., 1:])
                    rgb_pred = lab_to_rgb(lab_pred[..., 0], lab_pred[..., 1:])
                    
                    val_psnr += psnr_fn(rgb_true, rgb_pred)
                    val_ssim += ssim_fn(rgb_true, rgb_pred, multichannel=True)

        val_loss /= len(val_loader.dataset)
        val_psnr /= len(val_loader.dataset)
        val_ssim /= len(val_loader.dataset)

        history['val_loss'].append(val_loss)
        history['val_psnr'].append(val_psnr)
        history['val_ssim'].append(val_ssim)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val PSNR: {val_psnr:.4f}, Val SSIM: {val_ssim:.4f}')
    
    return model, history


# Testing
def test_model(model, test_loader, checkpoint_path, device='cuda'):
    model.load_state_dict(torch.load(checkpoint_path))
    model = model.to(device)
    model.eval()

    psnr_scores, ssim_scores = [], []
    with torch.no_grad():
        for L, ab in test_loader:
            L, ab = L.to(device), ab.to(device)
            ab_pred = model(L)

            # Calculate PSNR and SSIM
            L_denorm = denormalise_L(L.cpu().numpy())
            ab_denorm = denormalise_ab(ab.cpu().numpy())
            ab_pred_denorm = denormalise_ab(ab_pred.cpu().numpy())

            for i in range(L.size(0)):
                lab_true = np.concatenate([L_denorm[i], ab_denorm[i]], axis=0).transpose(1, 2, 0)
                lab_pred = np.concatenate([L_denorm[i], ab_pred_denorm[i]], axis=0).transpose(1, 2, 0)
                
                rgb_true = lab_to_rgb(lab_true[..., 0], lab_true[..., 1:])
                rgb_pred = lab_to_rgb(lab_pred[..., 0], lab_pred[..., 1:])
                
                psnr_scores.append(psnr_fn(rgb_true, rgb_pred))
                ssim_scores.append(ssim_fn(rgb_true, rgb_pred, multichannel=True))

    metrics = {
        'psnr': np.mean(psnr_scores),
        'ssim': np.mean(ssim_scores)
    }
    print(f'Test PSNR: {metrics["psnr"]:.4f}, Test SSIM: {metrics["ssim"]:.4f}')
    return metrics

# Visualise results

def visualise_results(model, test_loader, checkpoint_path, device='cuda', num_images=5, save_path="recognition/MambaIRv2_s48448284/figures"):
    model.load_state_dict(torch.load(checkpoint_path))
    model = model.to(device)
    model.eval()

    Path(save_path).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(num_images, 3, figsize=(10, 3 * num_images))
    fig.suptitle("Colourisation Results", fontsize=16)
    col_titles = ["Grayscale (L)", "Predicted (ab)", "Ground Truth (ab)"]
    for ax, col in zip(axes[0], col_titles):
        ax.set_title(col, fontsize=14)
        ax.axis('off')


    with torch.no_grad():
        for i, (L, ab) in enumerate(test_loader):
            if i >= num_images:
                break
            L, ab = L.to(device), ab.to(device)
            ab_pred = model(L)

            # Denormalise
            L_denorm = denormalise_L(L.cpu().numpy())
            ab_denorm = denormalise_ab(ab.cpu().numpy())
            ab_pred_denorm = denormalise_ab(ab_pred.cpu().numpy())

            for j in range(L.size(0)):
                lab_true = np.concatenate([L_denorm[j], ab_denorm[j]], axis=0).transpose(1, 2, 0)
                lab_pred = np.concatenate([L_denorm[j], ab_pred_denorm[j]], axis=0).transpose(1, 2, 0)
                
                rgb_true = lab_to_rgb(lab_true[..., 0], lab_true[..., 1:])
                rgb_pred = lab_to_rgb(lab_pred[..., 0], lab_pred[..., 1:])

                # Add images to the plot
                axes[i, 0].imshow(L_denorm[j][0], cmap='gray')
                axes[i, 1].imshow(rgb_pred)
                axes[i, 2].imshow(rgb_true)

    plt.tight_layout()
    plt.savefig(f"{save_path}/base_colourisation_results.png")
    plt.show()


# Main

if __name__ == "__main__":
    # Get data loaders
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=16, num_workers=4)

    # Initialize model
    model = ColouriseCNN()
    model, history = train_model(model, train_loader, val_loader, num_epochs=100, lr=1e-4)
    metrics = test_model(model, test_loader, checkpoint_path='best_model.pth')
    visualise_results(model, test_loader, checkpoint_path='best_model.pth', num_images=5)

