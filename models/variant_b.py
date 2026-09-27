import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.covariance import EllipticEnvelope
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1):
        super(LSTMAutoencoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Encoder
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Decoder
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        # x is (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.size()
        
        # Encode
        _, (hidden, cell) = self.encoder(x)
        # hidden is (num_layers, batch_size, hidden_dim)
        z = hidden[-1] # take the last layer's hidden state as embedding (batch_size, hidden_dim)
        
        # Decode
        # Repeat embedding for each time step
        z_repeated = z.unsqueeze(1).repeat(1, seq_len, 1)
        
        decoder_out, _ = self.decoder(z_repeated, (hidden, cell))
        reconstruction = self.output_layer(decoder_out)
        
        return reconstruction, z

def nt_xent_loss(z1, z2, temperature=0.5):
    """Normalized Temperature-scaled Cross Entropy Loss"""
    # z1 and z2 are batches of embeddings from two augmented views of the same batch of windows
    # Since we just have normal time series, a simple augmentation is adding Gaussian noise
    batch_size = z1.size(0)
    z = torch.cat((z1, z2), dim=0) # (2N, D)
    z = F.normalize(z, dim=1)
    
    sim_matrix = torch.matmul(z, z.T) / temperature
    
    # Mask out self-similarity
    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
    sim_matrix.masked_fill_(mask, -float('inf'))
    
    # Target labels
    targets = torch.cat((torch.arange(batch_size, 2*batch_size), 
                         torch.arange(batch_size)), dim=0).to(z.device)
    
    loss = F.cross_entropy(sim_matrix, targets)
    return loss

class VariantB:
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, lambda_nt=0.5, lr=1e-3, epochs=20, batch_size=64, device='cpu'):
        self.model = LSTMAutoencoder(input_dim, hidden_dim, num_layers).to(device)
        self.lambda_nt = lambda_nt
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.embed_scorer = EllipticEnvelope(contamination=0.01, random_state=42)
        
    def fit(self, X_benign):
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        X_tensor = torch.tensor(X_benign, dtype=torch.float32)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        for epoch in range(self.epochs):
            for batch in dataloader:
                x = batch[0].to(self.device)
                
                # Create augmented view (adding noise)
                x_aug = x + torch.randn_like(x) * 0.05
                
                optimizer.zero_grad()
                
                recon, z1 = self.model(x)
                _, z2 = self.model(x_aug)
                
                loss_mse = F.mse_loss(recon, x)
                loss_nt = nt_xent_loss(z1, z2)
                
                loss = loss_mse + self.lambda_nt * loss_nt
                loss.backward()
                optimizer.step()
                
        # Fit Gaussian on benign embeddings
        self.model.eval()
        with torch.no_grad():
            _, z_benign = self.model(X_tensor.to(self.device))
            self.embed_scorer.fit(z_benign.cpu().numpy())
            
    def score(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            recon, z = self.model(X_tensor)
            
            # Reconstruction MSE per window
            mse = torch.mean((recon - X_tensor)**2, dim=(1, 2)).cpu().numpy()
            
            # Embedding Mahalanobis distance
            z_np = z.cpu().numpy()
            embed_dist = self.embed_scorer.mahalanobis(z_np)
            
        # Total anomaly score
        return mse + embed_dist
