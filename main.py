#!/usr/bin/env python3
"""
Adaptive Anomaly Detection Pipeline
Main execution script for training and streaming inference
"""

import argparse
import yaml
import numpy as np
from models.ensemble_detector import EnsembleAnomalyDetector
from data.data_generator import LogDataGenerator
from streaming.kafka_streamer import KafkaStreamer
from streaming.spark_processor import SparkProcessor
from utils.visualization import ResultsVisualizer
from utils.metrics import ModelEvaluator
import warnings
warnings.filterwarnings('ignore')

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def run_training_pipeline(config):
    """Execute the training pipeline"""
    print("=" * 60)
    print("ADAPTIVE ANOMALY DETECTION TRAINING PIPELINE")
    print("=" * 60)
    
    # Initialize components
    detector = EnsembleAnomalyDetector(
        sequence_length=config['model']['sequence_length'],
        feature_dim=config['model']['feature_dim']
    )
    data_generator = LogDataGenerator()
    visualizer = ResultsVisualizer()
    evaluator = ModelEvaluator()
    
    # Generate data
    print("\n1. Generating synthetic log data...")
    normal_data = data_generator.generate_normal_data(config['data']['n_samples'])
    full_data, anomaly_metadata = data_generator.inject_anomalies(
        normal_data, 
        anomaly_rate=config['data']['anomaly_rate']
    )
    
    print(f"Generated {len(full_data)} records with {full_data['is_anomaly'].sum()} anomalies")
    
    # Prepare features
    feature_cols = config['data']['feature_columns']
    X = full_data[feature_cols].values
    y = full_data['is_anomaly'].values
    
    # Train-test split
    train_size = int(config['training']['train_ratio'] * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Normalize data
    detector.scaler.fit(X_train)
    X_train_scaled = detector.scaler.transform(X_train)
    X_test_scaled = detector.scaler.transform(X_test)
    
    # Train models
    print("\n2. Training ensemble models...")
    lstm_losses = detector.train_lstm(
        X_train_scaled, 
        epochs=config['training']['lstm_epochs'],
        batch_size=config['training']['batch_size']
    )
    detector.train_isolation_forest(X_train_scaled)
    # Mark ensemble as trained when individual components are trained
    detector.training_history['is_trained'] = True
    
    # Evaluate
    print("\n3. Evaluating models...")
    ensemble_scores, lstm_scores, iso_scores = detector.ensemble_predict(X_test_scaled)
    
    # Calculate metrics
    threshold = np.percentile(ensemble_scores, config['evaluation']['threshold_percentile'])
    predictions = (ensemble_scores > threshold).astype(int)
    # Align ground truth with ensemble scores. ensemble_scores correspond to sequences
    # whose last timestep index starts at sequence_length - 1
    start_idx = detector.sequence_length - 1
    y_test_aligned = y_test[start_idx:start_idx + len(ensemble_scores)]
    
    # Print results
    evaluator.print_metrics(y_test_aligned, predictions, ensemble_scores, threshold)
    
    # Visualize results
    visualizer.plot_training_results(
        lstm_losses, ensemble_scores, lstm_scores, iso_scores,
        y_test_aligned, predictions, threshold
    )
    
    return detector, full_data

def run_streaming_pipeline(config, detector, data):
    """Execute the streaming pipeline"""
    print("\n" + "=" * 60)
    print("REAL-TIME STREAMING ANOMALY DETECTION")
    print("=" * 60)
    
    kafka_streamer = KafkaStreamer(config['kafka']['bootstrap_servers'])
    
    if config['streaming']['simulation_mode']:
        print("\nRunning in simulation mode...")
        
        feature_cols = config['data']['feature_columns']
        anomaly_count = 0
        processed_count = 0
        
        batch_size = config['streaming']['batch_size']
        for i in range(0, len(data), batch_size):
            batch = data.iloc[i:i+batch_size]
            batch_features = detector.scaler.transform(batch[feature_cols].values)
            
            if len(batch_features) >= detector.sequence_length:
                ensemble_scores, _, _ = detector.ensemble_predict(batch_features)
                threshold = np.percentile(ensemble_scores, 
                                        config['evaluation']['threshold_percentile'])
                batch_anomalies = (ensemble_scores > threshold).sum()
                
                anomaly_count += batch_anomalies
                processed_count += len(ensemble_scores)
                
                print(f"Batch {i//batch_size + 1}: Processed {len(batch)} records, "
                      f"Found {batch_anomalies} anomalies")
        
        print(f"\nStreaming Summary:")
        print(f"Total records processed: {processed_count}")
        print(f"Total anomalies detected: {anomaly_count}")
        print(f"Anomaly rate: {anomaly_count/processed_count*100:.2f}%")
        
    else:
        print("\n4. Starting real Kafka streaming...")
        spark_processor = SparkProcessor()
        
        # Start producer and consumer
        import threading
        producer_thread = threading.Thread(
            target=kafka_streamer.produce_logs,
            args=(data, config['streaming']['batch_size'])
        )
        producer_thread.start()
        
        # Start Spark streaming
        query = spark_processor.process_stream(
            config['kafka']['bootstrap_servers'],
            config['kafka']['topic']
        )
        
        if query:
            import time
            time.sleep(config['streaming']['duration'])
            query.stop()
        
        producer_thread.join()

def main():
    parser = argparse.ArgumentParser(description='Adaptive Anomaly Detection Pipeline')
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('--mode', choices=['train', 'stream', 'both'], default='both',
                       help='Pipeline mode')
    parser.add_argument('--simulate', action='store_true', 
                       help='Run in simulation mode without Kafka/Spark')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    if args.simulate:
        config['streaming']['simulation_mode'] = True
    
    detector = None
    data = None
    
    # Run training pipeline
    if args.mode in ['train', 'both']:
        detector, data = run_training_pipeline(config)
    
    # Run streaming pipeline
    if args.mode in ['stream', 'both']:
        if detector is None or data is None:
            print("Error: Training must be completed before streaming")
            return
        run_streaming_pipeline(config, detector, data)
    
    print("\nPipeline execution completed!")

if __name__ == "__main__":
    main()