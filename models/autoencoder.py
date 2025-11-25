#!/usr/bin/env python3
"""
Autoencoder model for dimensionality reduction.

Uses a simple autoencoder to project segment statistics
into a latent space for anomaly detection.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler


def fit_autoencoder(X, latent=16, hidden_factor=2, epochs=100, batch_size=32, seed=42):
    """
    Fit an autoencoder to the input data and return latent representations.
    
    This is a simple autoencoder implementation using sklearn's MLPRegressor
    as a basis, or a custom numpy-based implementation for portability.
    
    Args:
        X: Input data (n_samples x n_features)
        latent: Latent dimension size
        hidden_factor: Hidden layer size = input_dim * hidden_factor
        epochs: Training epochs
        batch_size: Batch size for training
        seed: Random seed
        
    Returns:
        model: Fitted model (dict with weights)
        Z: Latent representations (n_samples x latent)
        reconstructed: Reconstructed data (n_samples x n_features)
    """
    np.random.seed(seed)
    
    n_samples, n_features = X.shape
    hidden_dim = max(latent * hidden_factor, n_features)
    
    # Standardize input
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize weights (Xavier initialization)
    def xavier_init(fan_in, fan_out):
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, (fan_in, fan_out))
    
    # Encoder: input -> hidden -> latent
    W_enc1 = xavier_init(n_features, hidden_dim)
    b_enc1 = np.zeros(hidden_dim)
    W_enc2 = xavier_init(hidden_dim, latent)
    b_enc2 = np.zeros(latent)
    
    # Decoder: latent -> hidden -> output
    W_dec1 = xavier_init(latent, hidden_dim)
    b_dec1 = np.zeros(hidden_dim)
    W_dec2 = xavier_init(hidden_dim, n_features)
    b_dec2 = np.zeros(n_features)
    
    def relu(x):
        return np.maximum(0, x)
    
    def relu_derivative(x):
        return (x > 0).astype(float)
    
    def forward(X_batch):
        # Encoder
        pre_h1 = X_batch @ W_enc1 + b_enc1
        h1 = relu(pre_h1)
        pre_z = h1 @ W_enc2 + b_enc2
        z = relu(pre_z)
        # Decoder
        pre_h2 = z @ W_dec1 + b_dec1
        h2 = relu(pre_h2)
        x_recon = h2 @ W_dec2 + b_dec2
        return pre_h1, h1, pre_z, z, pre_h2, h2, x_recon
    
    # Training loop
    learning_rate = 0.001
    
    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(n_samples)
        total_loss = 0.0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X_scaled[batch_idx]
            bs = len(batch_idx)
            
            # Forward pass
            pre_h1, h1, pre_z, z, pre_h2, h2, x_recon = forward(X_batch)
            
            # Loss (MSE)
            loss = np.mean((X_batch - x_recon) ** 2)
            total_loss += loss
            n_batches += 1
            
            # Backward pass (gradient descent)
            # Output layer gradient
            d_x_recon = 2 * (x_recon - X_batch) / bs
            
            # Decoder layer 2
            d_W_dec2 = h2.T @ d_x_recon
            d_b_dec2 = d_x_recon.sum(axis=0)
            d_h2 = d_x_recon @ W_dec2.T * relu_derivative(pre_h2)
            
            # Decoder layer 1
            d_W_dec1 = z.T @ d_h2
            d_b_dec1 = d_h2.sum(axis=0)
            d_z = d_h2 @ W_dec1.T * relu_derivative(pre_z)
            
            # Encoder layer 2
            d_W_enc2 = h1.T @ d_z
            d_b_enc2 = d_z.sum(axis=0)
            d_h1 = d_z @ W_enc2.T * relu_derivative(pre_h1)
            
            # Encoder layer 1
            d_W_enc1 = X_batch.T @ d_h1
            d_b_enc1 = d_h1.sum(axis=0)
            
            # Update weights
            W_enc1 -= learning_rate * d_W_enc1
            b_enc1 -= learning_rate * d_b_enc1
            W_enc2 -= learning_rate * d_W_enc2
            b_enc2 -= learning_rate * d_b_enc2
            W_dec1 -= learning_rate * d_W_dec1
            b_dec1 -= learning_rate * d_b_dec1
            W_dec2 -= learning_rate * d_W_dec2
            b_dec2 -= learning_rate * d_b_dec2
    
    # Get final latent representations
    _, _, _, Z, _, _, reconstructed = forward(X_scaled)
    
    # Unscale reconstructed
    reconstructed = scaler.inverse_transform(reconstructed)
    
    # Package model
    model = {
        'scaler': scaler,
        'W_enc1': W_enc1, 'b_enc1': b_enc1,
        'W_enc2': W_enc2, 'b_enc2': b_enc2,
        'W_dec1': W_dec1, 'b_dec1': b_dec1,
        'W_dec2': W_dec2, 'b_dec2': b_dec2,
        'latent': latent
    }
    
    return model, Z, reconstructed


def encode(model, X):
    """
    Encode new data using fitted autoencoder.
    
    Args:
        model: Fitted autoencoder model (dict)
        X: Input data (n_samples x n_features)
        
    Returns:
        Z: Latent representations
    """
    X_scaled = model['scaler'].transform(X)
    
    h1 = np.maximum(0, X_scaled @ model['W_enc1'] + model['b_enc1'])
    z = np.maximum(0, h1 @ model['W_enc2'] + model['b_enc2'])
    
    return z
