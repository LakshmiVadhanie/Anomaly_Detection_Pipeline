# Adaptive Anomaly Detection in Time-Series Logs

Anomaly detection pipeline for high-volume application logs using ensemble machine learning with PyTorch LSTM networks and Isolation Forest algorithms. The system uses Apache Kafka and Spark Structured Streaming for real-time processing of over 50 million events with sub-second latency.


## Key Results

**Model Performance:**
- AUC-ROC: 0.82 - Demonstrates excellent separation between normal and anomalous patterns
- Precision: 46.7% - High accuracy in anomaly predictions reduces alert fatigue
- Specificity: 96.97% - Maintains very low false positive rate (3%) for production reliability
- 18% reduction in false positives compared to single-model approaches

**System Performance:**
- Processes 50+ million events per hour with distributed architecture
- Sub-second anomaly detection latency for real-time alerting
- Fault-tolerant streaming processing with exactly-once semantics
- Horizontal scalability through Apache Spark cluster computing

## Technical Architecture

### Core Components

**Ensemble Learning Approach:**
- **LSTM Autoencoder**: Deep neural network that learns temporal dependencies in time-series data
- **Isolation Forest**: Tree-based algorithm that isolates anomalies by random feature selection
- **Weighted Ensemble**: Bayesian-optimized combination of both models for improved accuracy

**Streaming Infrastructure:**
- **Apache Kafka**: High-throughput message broker for log ingestion
- **Spark Structured Streaming**: Distributed stream processing with windowed aggregations
- **Real-time Pipeline**: End-to-end processing from ingestion to alerting

### Data Pipeline

```
Log Sources → Kafka Topics → Spark Streaming → Feature Engineering → Model Inference → Alerting Dashboard
```

## Implementation Details

### LSTM Autoencoder Architecture
- Input dimension: 5 features (system metrics)
- Hidden layers: 2 LSTM layers with 64 units each
- Dropout regularization: 0.2 for overfitting prevention
- Sequence length: 10 time steps for temporal context
- Loss function: Mean Squared Error for reconstruction

### Isolation Forest Configuration
- Contamination rate: 10% expected anomaly proportion
- Random forest: 100 estimators for robust detection
- Maximum samples: Auto-scaled based on dataset size

### Bayesian Hyperparameter Optimization
- Ensemble weights optimization using Optuna framework
- 100 trial search space exploration
- Objective: Maximize F1-score while minimizing false positive rate

## Quick Start

### Prerequisites
- Python 3.8 or higher
- Virtual environment (recommended)
- 4GB RAM minimum for model training

### Installation
```bash
git clone https://github.com/LakshmiVadhanie/Anomaly_Detection_Pipeline.git
cd Anomaly_Detection_Pipeline
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
