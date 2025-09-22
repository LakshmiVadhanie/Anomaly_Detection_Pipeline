"""
Visualization utilities for anomaly detection results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

plt.style.use('default')
sns.set_palette("husl")

class ResultsVisualizer:
    """Comprehensive visualization for anomaly detection results"""
    
    def __init__(self, figsize=(15, 10)):
        self.figsize = figsize
        self.colors = {
            'normal': '#2E86C1',
            'anomaly': '#E74C3C',
            'lstm': '#28B463',
            'isolation': '#F39C12',
            'ensemble': '#8E44AD',
            'threshold': '#34495E'
        }
    
    def plot_training_results(self, lstm_losses, ensemble_scores, lstm_scores, 
                            iso_scores, y_true, predictions, threshold):
        """Create comprehensive training results visualization"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Anomaly Detection Pipeline Results', fontsize=16, fontweight='bold')
        
        # 1. Training Loss
        axes[0, 0].plot(lstm_losses, color=self.colors['lstm'], linewidth=2)
        axes[0, 0].set_title('LSTM Training Loss', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('MSE Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Anomaly Scores Comparison
        time_idx = range(len(ensemble_scores))
        axes[0, 1].plot(time_idx, lstm_scores, color=self.colors['lstm'], 
                       alpha=0.7, label='LSTM Scores', linewidth=1.5)
        axes[0, 1].plot(time_idx, iso_scores, color=self.colors['isolation'], 
                       alpha=0.7, label='Isolation Forest', linewidth=1.5)
        axes[0, 1].plot(time_idx, ensemble_scores, color=self.colors['ensemble'], 
                       linewidth=2, label='Ensemble')
        axes[0, 1].axhline(y=threshold, color=self.colors['threshold'], 
                          linestyle='--', label=f'Threshold ({threshold:.3f})')
        axes[0, 1].set_title('Anomaly Scores Comparison', fontweight='bold')
        axes[0, 1].set_xlabel('Time Steps')
        axes[0, 1].set_ylabel('Anomaly Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Score Distribution
        axes[0, 2].hist(ensemble_scores[y_true == 0], bins=30, alpha=0.7, 
                       color=self.colors['normal'], label='Normal', density=True)
        axes[0, 2].hist(ensemble_scores[y_true == 1], bins=30, alpha=0.7, 
                       color=self.colors['anomaly'], label='Anomaly', density=True)
        axes[0, 2].axvline(x=threshold, color=self.colors['threshold'], 
                          linestyle='--', label='Threshold')
        axes[0, 2].set_title('Score Distribution', fontweight='bold')
        axes[0, 2].set_xlabel('Ensemble Score')
        axes[0, 2].set_ylabel('Density')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
                   xticklabels=['Normal', 'Anomaly'], 
                   yticklabels=['Normal', 'Anomaly'])
        axes[1, 0].set_title('Confusion Matrix', fontweight='bold')
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        
        # 5. Time Series with Anomalies
        axes[1, 1].scatter(time_idx, ensemble_scores, 
                          c=['red' if pred else 'blue' for pred in predictions], 
                          alpha=0.6, s=10)
        axes[1, 1].axhline(y=threshold, color=self.colors['threshold'], 
                          linestyle='--', alpha=0.8)
        axes[1, 1].set_title('Detected Anomalies Over Time', fontweight='bold')
        axes[1, 1].set_xlabel('Time Steps')
        axes[1, 1].set_ylabel('Ensemble Score')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Performance Metrics
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
        precision = precision_score(y_true, predictions, zero_division=0)
        recall = recall_score(y_true, predictions, zero_division=0)
        f1 = f1_score(y_true, predictions, zero_division=0)
        
        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, ensemble_scores)
        else:
            auc = 0.0
        
        metrics_text = f"""Performance Metrics:

Precision: {precision:.3f}
Recall: {recall:.3f}
F1-Score: {f1:.3f}
AUC-ROC: {auc:.3f}

True Anomalies: {y_true.sum()}
Predicted Anomalies: {predictions.sum()}
Threshold: {threshold:.3f}"""
        
        axes[1, 2].text(0.1, 0.7, metrics_text, transform=axes[1, 2].transAxes,
                       fontsize=12, verticalalignment='top', 
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[1, 2].set_title('Performance Summary', fontweight='bold')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print("\n" + "="*60)
        print("MODEL EVALUATION SUMMARY")
        print("="*60)
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"AUC-ROC: {auc:.4f}")
        print(f"Threshold: {threshold:.4f}")
        print("="*60)
    
    def plot_streaming_results(self, streaming_data, predictions, window_size=100):
        """Visualize streaming anomaly detection results"""
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle('Real-Time Anomaly Detection Results', fontsize=16, fontweight='bold')
        
        # Prepare data
        timestamps = range(len(streaming_data))
        
        # 1. System Metrics Over Time
        axes[0].plot(timestamps, streaming_data['cpu_usage'], 
                    label='CPU Usage (%)', alpha=0.8)
        axes[0].plot(timestamps, streaming_data['memory_usage'], 
                    label='Memory Usage (%)', alpha=0.8)
        axes[0].plot(timestamps, streaming_data['response_time']/10, 
                    label='Response Time (ms/10)', alpha=0.8)
        
        # Highlight anomalies
        anomaly_indices = np.where(predictions == 1)[0]
        for idx in anomaly_indices:
            axes[0].axvline(x=idx, color='red', alpha=0.3, linewidth=1)
        
        axes[0].set_title('System Metrics with Detected Anomalies', fontweight='bold')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Metric Value')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Anomaly Detection Rate Over Time
        window_anomalies = []
        for i in range(0, len(predictions) - window_size + 1, window_size//4):
            window = predictions[i:i+window_size]
            anomaly_rate = window.sum() / len(window) * 100
            window_anomalies.append(anomaly_rate)
        
        window_timestamps = range(0, len(predictions) - window_size + 1, window_size//4)
        axes[1].bar(window_timestamps, window_anomalies, 
                   width=window_size//4, alpha=0.7, color=self.colors['anomaly'])
        axes[1].set_title(f'Anomaly Detection Rate (Window Size: {window_size})', 
                         fontweight='bold')
        axes[1].set_xlabel('Time Window')
        axes[1].set_ylabel('Anomaly Rate (%)')
        axes[1].grid(True, alpha=0.3)
        
        # 3. Cumulative Anomaly Count
        cumulative_anomalies = np.cumsum(predictions)
        axes[2].plot(timestamps, cumulative_anomalies, 
                    color=self.colors['ensemble'], linewidth=2)
        axes[2].set_title('Cumulative Anomaly Count', fontweight='bold')
        axes[2].set_xlabel('Time')
        axes[2].set_ylabel('Total Anomalies Detected')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print streaming summary
        total_records = len(predictions)
        total_anomalies = predictions.sum()
        anomaly_rate = total_anomalies / total_records * 100
        
        print("\n" + "="*60)
        print("STREAMING DETECTION SUMMARY")
        print("="*60)
        print(f"Total Records Processed: {total_records:,}")
        print(f"Total Anomalies Detected: {total_anomalies}")
        print(f"Overall Anomaly Rate: {anomaly_rate:.2f}%")
        print("="*60)
    
    def create_interactive_dashboard(self, data, scores, predictions, threshold):
        """Create interactive Plotly dashboard"""
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('System Metrics Over Time', 'Anomaly Scores',
                          'Score Distribution', 'Anomaly Timeline',
                          'Performance Heatmap', 'Detection Summary'),
            specs=[[{"secondary_y": True}, {"type": "scatter"}],
                   [{"type": "histogram"}, {"type": "scatter"}],
                   [{"type": "heatmap"}, {"type": "indicator"}]]
        )
        
        # System metrics
        time_idx = list(range(len(data)))
        fig.add_trace(
            go.Scatter(x=time_idx, y=data['cpu_usage'], name='CPU Usage',
                      line=dict(color='blue')), row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=time_idx, y=data['memory_usage'], name='Memory Usage',
                      line=dict(color='green')), row=1, col=1, secondary_y=False
        )
        
        # Anomaly scores
        colors = ['red' if pred else 'blue' for pred in predictions]
        fig.add_trace(
            go.Scatter(x=time_idx, y=scores, mode='markers', name='Anomaly Scores',
                      marker=dict(color=colors, size=4)), row=1, col=2
        )
        fig.add_hline(y=threshold, line_dash="dash", line_color="black", 
                     row=1, col=2)
        
        # Score distribution
        fig.add_trace(
            go.Histogram(x=scores, nbinsx=30, name='Score Distribution',
                        marker_color='lightblue'), row=2, col=1
        )
        
        # Anomaly timeline
        anomaly_times = [i for i, pred in enumerate(predictions) if pred == 1]
        fig.add_trace(
            go.Scatter(x=anomaly_times, y=[1]*len(anomaly_times),
                      mode='markers', name='Detected Anomalies',
                      marker=dict(color='red', size=8)), row=2, col=2
        )
        
        # Update layout
        fig.update_layout(height=900, showlegend=True, 
                         title_text="Interactive Anomaly Detection Dashboard")
        
        return fig
    
    def save_plots(self, filename_prefix="anomaly_detection"):
        """Save current plots to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {filename}")
    
    def plot_feature_importance(self, feature_names, importances):
        """Plot feature importance if available"""
        if importances is None:
            print("Feature importance data not available")
            return
        
        plt.figure(figsize=(10, 6))
        indices = np.argsort(importances)[::-1]
        
        plt.bar(range(len(importances)), importances[indices], 
                color=self.colors['ensemble'], alpha=0.7)
        plt.xticks(range(len(importances)), 
                  [feature_names[i] for i in indices], rotation=45)
        plt.title('Feature Importance (Isolation Forest)', fontweight='bold')
        plt.xlabel('Features')
        plt.ylabel('Importance Score')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()