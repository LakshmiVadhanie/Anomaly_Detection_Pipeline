"""
Evaluation metrics and utilities for anomaly detection
"""

import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, roc_curve, average_precision_score,
    precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import pandas as pd

class ModelEvaluator:
    """Comprehensive evaluation metrics for anomaly detection models"""
    
    def __init__(self):
        self.metrics_history = []
    
    def calculate_all_metrics(self, y_true, y_pred, y_scores):
        """Calculate comprehensive set of metrics"""
        
        # Basic classification metrics
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # ROC metrics (if both classes present)
        if len(np.unique(y_true)) > 1:
            auc_roc = roc_auc_score(y_true, y_scores)
            auc_pr = average_precision_score(y_true, y_scores)
        else:
            auc_roc = 0.0
            auc_pr = 0.0
        
        # Confusion matrix components
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        # False positive rate (important for anomaly detection)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'accuracy': accuracy,
            'specificity': specificity,
            'npv': npv,
            'fpr': fpr,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'total_anomalies': y_true.sum(),
            'predicted_anomalies': y_pred.sum()
        }
        
        return metrics
    
    def print_metrics(self, y_true, y_pred, y_scores, threshold):
        """Print formatted metrics summary"""
        metrics = self.calculate_all_metrics(y_true, y_pred, y_scores)
        
        print("\n" + "="*60)
        print("ANOMALY DETECTION PERFORMANCE METRICS")
        print("="*60)
        print(f"Threshold: {threshold:.4f}")
        print(f"Total Samples: {len(y_true)}")
        print(f"True Anomalies: {metrics['total_anomalies']}")
        print(f"Predicted Anomalies: {metrics['predicted_anomalies']}")
        print("-"*60)
        print("CLASSIFICATION METRICS:")
        print(f"Precision:    {metrics['precision']:.4f}")
        print(f"Recall:       {metrics['recall']:.4f}")
        print(f"F1-Score:     {metrics['f1_score']:.4f}")
        print(f"Accuracy:     {metrics['accuracy']:.4f}")
        print(f"Specificity:  {metrics['specificity']:.4f}")
        print(f"AUC-ROC:      {metrics['auc_roc']:.4f}")
        print(f"AUC-PR:       {metrics['auc_pr']:.4f}")
        print("-"*60)
        print("CONFUSION MATRIX:")
        print(f"True Positives:   {metrics['tp']}")
        print(f"True Negatives:   {metrics['tn']}")
        print(f"False Positives:  {metrics['fp']}")
        print(f"False Negatives:  {metrics['fn']}")
        print(f"False Positive Rate: {metrics['fpr']:.4f}")
        print("="*60)
        
        # Store metrics for history
        self.metrics_history.append(metrics)
        
        return metrics
    
    def plot_roc_curve(self, y_true, y_scores, title="ROC Curve"):
        """Plot ROC curve"""
        if len(np.unique(y_true)) <= 1:
            print("Cannot plot ROC curve: only one class present")
            return
        
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return fpr, tpr, thresholds
    
    def plot_precision_recall_curve(self, y_true, y_scores, title="Precision-Recall Curve"):
        """Plot Precision-Recall curve"""
        if len(np.unique(y_true)) <= 1:
            print("Cannot plot PR curve: only one class present")
            return
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        auc_pr = average_precision_score(y_true, y_scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2, label=f'PR Curve (AUC = {auc_pr:.3f})')
        plt.axhline(y=y_true.mean(), color='k', linestyle='--', 
                   label=f'Random Classifier (AP = {y_true.mean():.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title, fontweight='bold')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return precision, recall, thresholds
    
    def find_optimal_threshold(self, y_true, y_scores, metric='f1'):
        """Find optimal threshold based on specified metric"""
        if len(np.unique(y_true)) <= 1:
            print("Cannot optimize threshold: only one class present")
            return 0.5
        
        thresholds = np.linspace(0, 1, 100)
        scores = []
        
        for threshold in thresholds:
            y_pred = (y_scores >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'precision':
                score = precision_score(y_true, y_pred, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_true, y_pred, zero_division=0)
            elif metric == 'balanced':
                prec = precision_score(y_true, y_pred, zero_division=0)
                rec = recall_score(y_true, y_pred, zero_division=0)
                score = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            scores.append(score)
        
        optimal_idx = np.argmax(scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_score = scores[optimal_idx]
        
        print(f"Optimal threshold for {metric}: {optimal_threshold:.4f}")
        print(f"Optimal {metric} score: {optimal_score:.4f}")
        
        # Plot threshold vs score
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, scores, linewidth=2, color='blue')
        plt.axvline(x=optimal_threshold, color='red', linestyle='--', 
                   label=f'Optimal Threshold = {optimal_threshold:.3f}')
        plt.xlabel('Threshold')
        plt.ylabel(f'{metric.capitalize()} Score')
        plt.title(f'Threshold Optimization ({metric.capitalize()})', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return optimal_threshold
    
    def compare_models(self, model_results, model_names):
        """Compare multiple models side by side"""
        
        metrics_df = pd.DataFrame()
        
        for i, (y_true, y_pred, y_scores) in enumerate(model_results):
            metrics = self.calculate_all_metrics(y_true, y_pred, y_scores)
            metrics['model'] = model_names[i]
            metrics_df = pd.concat([metrics_df, pd.DataFrame([metrics])], ignore_index=True)
        
        # Display comparison table
        comparison_cols = ['model', 'precision', 'recall', 'f1_score', 'auc_roc', 
                          'fpr', 'total_anomalies', 'predicted_anomalies']
        print("\nMODEL COMPARISON:")
        print("-" * 80)
        print(metrics_df[comparison_cols].to_string(index=False, float_format='%.4f'))
        
        return metrics_df
    
    def calculate_detection_efficiency(self, y_true, y_scores, alert_budgets=[10, 50, 100]):
        """Calculate detection efficiency for different alert budgets"""
        
        # Sort by scores (descending)
        sorted_indices = np.argsort(y_scores)[::-1]
        sorted_true = y_true[sorted_indices]
        
        efficiency_results = []
        
        for budget in alert_budgets:
            if budget <= len(sorted_true):
                top_k_true = sorted_true[:budget]
                detected_anomalies = top_k_true.sum()
                total_anomalies = y_true.sum()
                
                detection_rate = detected_anomalies / total_anomalies if total_anomalies > 0 else 0
                precision_at_k = detected_anomalies / budget
                
                efficiency_results.append({
                    'alert_budget': budget,
                    'detected_anomalies': detected_anomalies,
                    'total_anomalies': total_anomalies,
                    'detection_rate': detection_rate,
                    'precision_at_k': precision_at_k
                })
        
        efficiency_df = pd.DataFrame(efficiency_results)
        
        print("\nDETECTION EFFICIENCY ANALYSIS:")
        print("-" * 60)
        print(efficiency_df.to_string(index=False, float_format='%.4f'))
        
        return efficiency_df
    
    def export_metrics(self, filepath, format='csv'):
        """Export metrics history to file"""
        if not self.metrics_history:
            print("No metrics to export")
            return
        
        df = pd.DataFrame(self.metrics_history)
        
        if format == 'csv':
            df.to_csv(filepath, index=False)
        elif format == 'json':
            df.to_json(filepath, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Metrics exported to {filepath}")
    
    def get_metrics_summary(self):
        """Get summary of all calculated metrics"""
        if not self.metrics_history:
            return None
        
        df = pd.DataFrame(self.metrics_history)
        summary = {
            'count': len(df),
            'avg_precision': df['precision'].mean(),
            'avg_recall': df['recall'].mean(),
            'avg_f1': df['f1_score'].mean(),
            'avg_auc_roc': df['auc_roc'].mean(),
            'best_f1': df['f1_score'].max(),
            'best_auc': df['auc_roc'].max()
        }
        
        return summary