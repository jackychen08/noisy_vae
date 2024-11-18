import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
import os
import wandb
from tqdm import tqdm

# Sweep configuration
sweep_config = {
    'method': 'grid',
    'name': 'Noise Type Sweep',
    'metric': {
        'name': 'avg_train_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'architecture': {'value': 'VAE'},
        'dataset': {'value': 'tiny-imagenet'},
        'epochs': {'value': 100},
        'batch_size': {'value': 256},
        'learning_rate': {'value': 1e-3},
        'latent_dim': {'value': 128},
        'img_channels': {'value': 3},
        'noise_config': {
            'values': [
                # Gaussian configurations
                {'type': 'gaussian', 'std': 0.05},
                {'type': 'gaussian', 'std': 0.1},
                {'type': 'gaussian', 'std': 0.15},
                {'type': 'gaussian', 'std': 0.2},
                {'type': 'gaussian', 'std': 0.25},
                # Uniform configurations
                {'type': 'uniform', 'range': 0.05},
                {'type': 'uniform', 'range': 0.1},
                {'type': 'uniform', 'range': 0.15},
                {'type': 'uniform', 'range': 0.2},
                {'type': 'uniform', 'range': 0.25}
            ]
        }
    }
}

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]['image']
        if self.transform:
            image = self.transform(image)
        return image
    

def add_noise_forward(batch, config):
    """Add noise based on wandb config"""
    noise_type = config.noise_config['type']
    
    if noise_type == "gaussian":
        std = config.noise_config['std']
        noise = torch.randn_like(batch) * std
    else:  # uniform
        range_val = config.noise_config['range']
        noise = torch.empty_like(batch).uniform_(-range_val, range_val)
    
    noisy_batch = batch + noise
    # Clip values to valid range [-1, 1]
    return torch.clamp(noisy_batch, -1, 1)

class VAE(nn.Module):
    def __init__(self, img_channels, latent_dim):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(128 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(128 * 8 * 8, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 128 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, img_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        z = self.reparameterize(mu, logvar)
        decoded = self.decoder(self.decoder_input(z).view(-1, 128, 8, 8))
        return decoded, mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld



def train():
    # Initialize wandb
    wandb.init()
    
    # Update config with sweep parameters
    config = wandb.config
    
    # Update noise configuration directly from the noise_config parameter
    config.noise = config.noise_config

    # Create experiment-specific directory
    noise_type = config.noise['type']
    noise_value = config.noise.get('std', config.noise.get('range'))
    experiment_dir = os.path.join(
        'experiments', 
        noise_type,
        f'value_{noise_value}'
    )
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Lambda(lambda img: img.convert('RGB')),  
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) 
    ])

    print("Loading data ...")
    imagenet_train = load_dataset('Maysee/tiny-imagenet', split='train')
    imagenet_val = load_dataset('Maysee/tiny-imagenet', split='valid')

    train_dataset = ImageDataset(imagenet_train, transform=transform)
    val_dataset = ImageDataset(imagenet_val, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    device = 'cuda:0'
    vae = VAE(img_channels=config.img_channels, latent_dim=config.latent_dim).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=config.learning_rate)

    # Watch the model with wandb
    wandb.watch(vae, log="all", log_freq=10)

    # Create directory for saving checkpoints
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in tqdm(range(config.epochs), desc='Epochs'):
        vae.train()
        train_loss = 0
        batch_count = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=False):
            batch = batch.to(device)

            # Add noise using config parameters
            noisy_batch = add_noise_forward(batch, config)
            
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae(noisy_batch)
            loss = vae_loss(recon_batch, batch, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            batch_count += 1

            optimizer.step()
        
        avg_epoch_loss = train_loss / len(train_loader.dataset)
        
        wandb.log({
            "epoch": epoch,
            "avg_train_loss": avg_epoch_loss,
        })
        
        if (epoch + 1) % 10 == 0 or epoch == config.epochs - 1:
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f'vae_epoch_{epoch+1}.pth'
            )
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': vae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
                'noise_config': config.noise
            }
            torch.save(checkpoint, checkpoint_path)
            wandb.save(checkpoint_path)  # Only log every 10th checkpoint
            
            print(f'Saved checkpoint at epoch {epoch+1}')
        
        print(f'Epoch {epoch+1}, Loss: {avg_epoch_loss:.4f}')

if __name__ == "__main__":
    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project="DataNoise", entity="ml_10701")
    
    # Start the sweep
    wandb.agent(sweep_id, function=train)