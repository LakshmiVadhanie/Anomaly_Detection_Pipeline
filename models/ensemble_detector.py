"""
Ensemble Anomaly Detector
Combines LSTM and Isolation Forest for robust anomaly detection
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from models.lstm_detector import LSTMAnomalyDetector, LSTMTrainer
import joblib
import os
import torch

class EnsembleAnomalyDetector:
    """Ensemble detector combining LSTM and Isolation Forest"""
    
    def __init__(self, sequence_length=10, feature_dim=5, 
                 lstm_hidden_size=64, lstm_num_layers=2, lstm_dropout=0.2,
                 isolation_contamination=0.1, 
                 lstm_weight=0.6, isolation_weight=0.4):
        
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        
        # Scalers
        self.scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        
        # LSTM components
        self.lstm_model = LSTMAnomalyDetector(
            input_size=feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=lstm_dropout
        )
        self.lstm_trainer = LSTMTrainer(self.lstm_model)
        
        # Isolation Forest
        self.isolation_forest = IsolationForest(
            contamination=isolation_contamination,
            random_state=42,
            n_jobs=-1
        )
        
        # Ensemble weights
        self.lstm_weight = lstm_weight
        self.isolation_weight = isolation_weight
        
        # Training history
        self.training_history = {
            'lstm_losses': [],
            'is_trained': False
        }
    
    def train_lstm(self, X_train, epochs=50, batch_size=32, learning_rate=0.001):
        """Train LSTM component"""
        print("Training LSTM Autoencoder...")
        
        # Update learning rate if different
        if learning_rate != 0.001:
            self.lstm_trainer.optimizer = torch.optim.Adam(
                self.lstm_model.parameters(), lr=learning_rate
            )
        
        # Train LSTM
        losses = self.lstm_trainer.train(
            X_train, 
            sequence_length=self.sequence_length,
            epochs=epochs,
            batch_size=batch_size,
            verbose=True
        )
        
        self.training_history['lstm_losses'] = losses
        return losses
    
    def train_isolation_forest(self, X_train):
        """Train Isolation Forest component"""
        print("Training Isolation Forest...")
        self.isolation_forest.fit(X_train)
        print("Isolation Forest training completed")
    
    def train(self, X_train, lstm_epochs=50, batch_size=32, learning_rate=0.001):
        """Train both components of the ensemble"""
        print("Training Ensemble Anomaly Detector...")
        
        # Train LSTM
        lstm_losses = self.train_lstm(X_train, lstm_epochs, batch_size, learning_rate)
        
        # Train Isolation Forest
        self.train_isolation_forest(X_train)
        
        self.training_history['is_trained'] = True
        
        return lstm_losses
    
    def get_lstm_scores(self, X):
        """Get anomaly scores from LSTM"""
        if not self.training_history['is_trained']:
            raise ValueError("Models must be trained before prediction")
        
        return self.lstm_trainer.get_reconstruction_errors(X, self.sequence_length)
    
    def get_isolation_forest_scores(self, X):
        """Get anomaly scores from Isolation Forest"""
        # Get decision function scores (negative for inliers)
        scores = -self.isolation_forest.decision_function(X)
        return scores
    
    def ensemble_predict(self, X):
        """Generate ensemble predictions"""
        if not self.training_history['is_trained']:
            raise ValueError("Models must be trained before prediction")
        
        # LSTM scores (only for sequences after sequence_length)
        lstm_scores = self.get_lstm_scores(X)
        
        # Isolation Forest scores (align with LSTM by mapping each sequence to its last timestep)
        # For sequences that start at i and length L, the corresponding sample index is i + L - 1
        start_idx = self.sequence_length - 1
        X_aligned = X[start_idx:start_idx + len(lstm_scores)]
        iso_scores = self.get_isolation_forest_scores(X_aligned)

        # Normalize scores to [0, 1] range using independent scalers
        from sklearn.preprocessing import MinMaxScaler as _MinMaxScaler
        lstm_scaler = _MinMaxScaler()
        iso_scaler = _MinMaxScaler()

        lstm_scores_norm = lstm_scaler.fit_transform(
            lstm_scores.reshape(-1, 1)
        ).flatten()

        iso_scores_norm = iso_scaler.fit_transform(
            iso_scores.reshape(-1, 1)
        ).flatten()
        
        # Ensemble combination
        ensemble_scores = (
            self.lstm_weight * lstm_scores_norm + 
            self.isolation_weight * iso_scores_norm
        )
        
        return ensemble_scores, lstm_scores_norm, iso_scores_norm
    
    def predict_anomalies(self, X, threshold_percentile=95):
        """Predict anomalies with ensemble method"""
        ensemble_scores, lstm_scores, iso_scores = self.ensemble_predict(X)
        
        # Calculate threshold
        threshold = np.percentile(ensemble_scores, threshold_percentile)
        
        # Make predictions
        predictions = (ensemble_scores > threshold).astype(int)
        
        return {
            'predictions': predictions,
            'ensemble_scores': ensemble_scores,
            'lstm_scores': lstm_scores,
            'isolation_scores': iso_scores,
            'threshold': threshold
        }
    
    def save_model(self, filepath):
        """Save the trained ensemble model"""
        if not self.training_history['is_trained']:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'lstm_model_state': self.lstm_model.state_dict(),
            'isolation_forest': self.isolation_forest,
            'scaler': self.scaler,
            'minmax_scaler': self.minmax_scaler,
            'sequence_length': self.sequence_length,
            'feature_dim': self.feature_dim,
            'lstm_weight': self.lstm_weight,
            'isolation_weight': self.isolation_weight,
            'training_history': self.training_history
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained ensemble model"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file {filepath} not found")
        
        model_data = joblib.load(filepath)
        
        # Restore model components
        self.lstm_model.load_state_dict(model_data['lstm_model_state'])
        self.isolation_forest = model_data['isolation_forest']
        self.scaler = model_data['scaler']
        self.minmax_scaler = model_data['minmax_scaler']
        self.sequence_length = model_data['sequence_length']
        self.feature_dim = model_data['feature_dim']
        self.lstm_weight = model_data['lstm_weight']
        self.isolation_weight = model_data['isolation_weight']
        self.training_history = model_data['training_history']
        
        # Recreate trainer
        self.lstm_trainer = LSTMTrainer(self.lstm_model)
        
        print(f"Model loaded from {filepath}")
    
    def update_ensemble_weights(self, lstm_weight, isolation_weight):
        """Update ensemble weights (for Bayesian optimization)"""
        if abs(lstm_weight + isolation_weight - 1.0) > 1e-6:
            raise ValueError("Ensemble weights must sum to 1.0")
        
        self.lstm_weight = lstm_weight
        self.isolation_weight = isolation_weight
        print(f"Updated ensemble weights: LSTM={lstm_weight:.3f}, "
              f"Isolation Forest={isolation_weight:.3f}")
    
    def get_feature_importance(self):
        """Get feature importance from Isolation Forest"""
        if hasattr(self.isolation_forest, 'feature_importances_'):
            return self.isolation_forest.feature_importances_
        else:
            print("Feature importance not available for Isolation Forest")
            return None