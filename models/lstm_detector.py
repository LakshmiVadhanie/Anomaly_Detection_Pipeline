"""
LSTM-based Anomaly Detector
PyTorch implementation for time series anomaly detection
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class LSTMAnomalyDetector(nn.Module):
    """LSTM Autoencoder for anomaly detection"""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMAnomalyDetector, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        
        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, input_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Initialize hidden states
        h0_enc = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c0_enc = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        
        # Encoder
        encoder_output, (hidden, cell) = self.encoder_lstm(x, (h0_enc, c0_enc))
        
        # Use the last hidden state as context
        context = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)
        
        # Decoder with context
        decoder_input = torch.zeros_like(x)
        decoder_output, _ = self.decoder_lstm(decoder_input, (hidden, cell))
        
        # Output reconstruction
        reconstruction = self.output_layer(self.dropout(decoder_output))
        
        return reconstruction

class LSTMTrainer:
    """Trainer class for LSTM anomaly detector"""
    
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.training_losses = []
        
    def create_sequences(self, data, sequence_length):
        """Create overlapping sequences from time series data"""
        sequences = []
        
        for i in range(len(data) - sequence_length + 1):
            seq = data[i:i + sequence_length]
            sequences.append(seq)
            
        return np.array(sequences)
    
    def train(self, X_train, sequence_length, epochs=50, batch_size=32, verbose=True):
        """Train the LSTM autoencoder"""
        # Create sequences
        sequences = self.create_sequences(X_train, sequence_length)
        
        # Convert to tensors
        train_data = TensorDataset(torch.FloatTensor(sequences))
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0
            batch_count = 0
            
            for batch_seq in train_loader:
                batch_seq = batch_seq[0]  # Extract from tuple
                
                self.optimizer.zero_grad()
                
                # Forward pass
                reconstruction = self.model(batch_seq)
                loss = self.criterion(reconstruction, batch_seq)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            avg_loss = epoch_loss / batch_count
            self.training_losses.append(avg_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
        
        return self.training_losses
    
    def get_reconstruction_errors(self, X, sequence_length):
        """Calculate reconstruction errors for anomaly detection"""
        self.model.eval()
        sequences = self.create_sequences(X, sequence_length)
        
        reconstruction_errors = []
        
        with torch.no_grad():
            for seq in sequences:
                seq_tensor = torch.FloatTensor(seq).unsqueeze(0)
                reconstruction = self.model(seq_tensor)
                
                # Calculate MSE for each timestep
                mse = nn.MSELoss(reduction='none')(reconstruction, seq_tensor)
                sequence_error = mse.mean().item()
                reconstruction_errors.append(sequence_error)
        
        return np.array(reconstruction_errors)
    
    def predict_anomalies(self, X, sequence_length, threshold_percentile=95):
        """Predict anomalies based on reconstruction error threshold"""
        errors = self.get_reconstruction_errors(X, sequence_length)
        threshold = np.percentile(errors, threshold_percentile)
        anomalies = (errors > threshold).astype(int)
        
        return anomalies, errors, threshold