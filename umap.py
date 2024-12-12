import torch
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Load the model checkpoint
checkpoint_path = "/content/drive/My Drive/checkpoints/vae_epoch_100.pth" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vae = VAE().to(device)
checkpoint = torch.load(checkpoint_path, map_location=device)
vae.load_state_dict(checkpoint['model_state_dict'])
vae.eval()

latent_vectors = []
labels = []
import noisy_vae.umap as umap

with torch.no_grad():
    for batch in tqdm(val_loader):
        images, batch_labels = batch  # Unpack images and labels
        images = images.to(device)
        encoded = vae.encoder(images)
        mu = vae.fc_mu(encoded)
        latent_vectors.append(mu.cpu().numpy())
        labels.extend(batch_labels.numpy())

latent_vectors = np.concatenate(latent_vectors, axis=0)
labels = np.array(labels)



# Perform UMAP
umap_reducer = umap.UMAP(n_components=2, random_state=42)
latent_umap = umap_reducer.fit_transform(latent_vectors)

def plot_latent_space(latent_2d, method_name, labels):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap='tab10', s=5, alpha=0.7)
    cbar = plt.colorbar(scatter)
    cbar.set_label("Class Labels", fontsize=16) 
    cbar.ax.tick_params(labelsize=16)  

    plt.title(f"Latent Space using {method_name} on noisy data", fontsize=24)
    plt.title(f"Latent Space using {method_name} on normal data", fontsize=24)
    plt.xlabel("UMAP 1", fontsize=18)  
    plt.ylabel("UMAP 2", fontsize=18)  
    plt.grid(True)
    plt.savefig(f"latent_space_{method_name}.pdf")
    plt.show()


plot_latent_space(latent_umap, "UMAP", labels)

