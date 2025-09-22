"""
Synthetic Log Data Generator
Creates realistic time-series log data with injected anomalies
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json

class LogDataGenerator:
    """Generate synthetic log data for anomaly detection testing"""
    
    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)
        self.features = ['cpu_usage', 'memory_usage', 'network_io', 'disk_io', 'response_time']
    
    def generate_normal_data(self, n_samples=1000, start_time=None):
        """Generate normal operational log data with realistic patterns"""
        if start_time is None:
            start_time = datetime(2025, 8, 1)
        
        # Time index for cyclical patterns
        time_index = np.arange(n_samples)
        
        # Generate realistic system metrics with seasonal patterns
        
        # CPU Usage (0-100%) - workload patterns
        cpu_base = 35 + 15 * np.sin(time_index * 2 * np.pi / 144)  # Daily cycle
        cpu_weekly = 5 * np.sin(time_index * 2 * np.pi / (144 * 7))  # Weekly cycle
        cpu_noise = np.random.normal(0, 8, n_samples)
        cpu_usage = np.clip(cpu_base + cpu_weekly + cpu_noise, 5, 95)
        
        # Memory Usage (0-100%) - gradual increase with GC cycles
        memory_base = 45 + 10 * np.sin(time_index * 2 * np.pi / 200)
        memory_gc = 15 * np.abs(np.sin(time_index * 2 * np.pi / 50))  # GC spikes
        memory_trend = time_index * 0.01  # Gradual increase
        memory_noise = np.random.normal(0, 5, n_samples)
        memory_usage = np.clip(memory_base + memory_gc + memory_trend + memory_noise, 10, 90)
        
        # Network I/O (MB/s) - correlated with CPU
        network_base = 25 + 0.3 * cpu_usage
        network_burst = 20 * np.random.exponential(0.1, n_samples)  # Occasional bursts
        network_burst[network_burst > 15] = 0  # Make bursts rare
        network_noise = np.random.normal(0, 5, n_samples)
        network_io = np.clip(network_base + network_burst + network_noise, 1, None)
        
        # Disk I/O (MB/s) - correlated with memory and periodic backup patterns
        disk_base = 15 + 0.2 * memory_usage
        disk_backup = 30 * (np.sin(time_index * 2 * np.pi / 1440) > 0.9).astype(float)  # Backup cycles
        disk_noise = np.random.normal(0, 3, n_samples)
        disk_io = np.clip(disk_base + disk_backup + disk_noise, 1, None)
        
        # Response Time (ms) - inverse correlation with resources
        response_base = 150 + 2 * (100 - cpu_usage) + 1.5 * (100 - memory_usage)
        response_network = 0.5 * network_io  # Network affects response
        response_spikes = 100 * np.random.exponential(0.05, n_samples)
        response_spikes[response_spikes < 50] = 0  # Make spikes less frequent
        response_noise = np.random.normal(0, 15, n_samples)
        response_time = np.clip(
            response_base + response_network + response_spikes + response_noise, 
            50, 2000
        )
        
        # Create DataFrame
        timestamps = [start_time + timedelta(minutes=i) for i in range(n_samples)]
        
        data = pd.DataFrame({
            'timestamp': timestamps,
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'network_io': network_io,
            'disk_io': disk_io,
            'response_time': response_time,
            'is_anomaly': 0
        })
        
        return data
    
    def inject_anomalies(self, data, anomaly_rate=0.05, anomaly_types=None):
        """Inject realistic anomalies into the data"""
        if anomaly_types is None:
            anomaly_types = ['cpu_spike', 'memory_leak', 'network_drop', 'disk_thrashing', 
                           'response_degradation', 'cascade_failure']
        
        data_copy = data.copy()
        n_anomalies = int(len(data) * anomaly_rate)
        anomaly_indices = np.random.choice(len(data), n_anomalies, replace=False)
        
        print(f"Injecting {n_anomalies} anomalies of types: {anomaly_types}")
        
        for idx in anomaly_indices:
            anomaly_type = np.random.choice(anomaly_types)
            
            if anomaly_type == 'cpu_spike':
                # High CPU utilization
                data_copy.loc[idx, 'cpu_usage'] = min(98, data_copy.loc[idx, 'cpu_usage'] * 2.5)
                data_copy.loc[idx, 'response_time'] *= 2.5
                
            elif anomaly_type == 'memory_leak':
                # Gradual memory increase (affects multiple consecutive points)
                leak_duration = np.random.randint(5, 15)
                end_idx = min(idx + leak_duration, len(data_copy))
                for i in range(idx, end_idx):
                    multiplier = 1 + 0.8 * (i - idx) / leak_duration
                    data_copy.loc[i, 'memory_usage'] = min(95, 
                        data_copy.loc[i, 'memory_usage'] * multiplier)
                    data_copy.loc[i, 'response_time'] *= (1 + 0.5 * (i - idx) / leak_duration)
                    data_copy.loc[i, 'is_anomaly'] = 1
                continue  # Skip setting is_anomaly below
                
            elif anomaly_type == 'network_drop':
                # Network connectivity issues
                data_copy.loc[idx, 'network_io'] *= 0.1
                data_copy.loc[idx, 'response_time'] *= 4
                
            elif anomaly_type == 'disk_thrashing':
                # Excessive disk I/O
                data_copy.loc[idx, 'disk_io'] *= 5
                data_copy.loc[idx, 'cpu_usage'] = min(90, data_copy.loc[idx, 'cpu_usage'] * 1.5)
                data_copy.loc[idx, 'response_time'] *= 2
                
            elif anomaly_type == 'response_degradation':
                # Slow response times without clear resource issue
                data_copy.loc[idx, 'response_time'] *= 5
                
            elif anomaly_type == 'cascade_failure':
                # Multiple systems failing together
                data_copy.loc[idx, 'cpu_usage'] = min(95, data_copy.loc[idx, 'cpu_usage'] * 2)
                data_copy.loc[idx, 'memory_usage'] = min(90, data_copy.loc[idx, 'memory_usage'] * 1.8)
                data_copy.loc[idx, 'network_io'] *= 0.3
                data_copy.loc[idx, 'response_time'] *= 6
            
            data_copy.loc[idx, 'is_anomaly'] = 1
        
        # Add anomaly metadata
        anomaly_metadata = []
        for idx in anomaly_indices:
            metadata = {
                'timestamp': data_copy.loc[idx, 'timestamp'].isoformat(),
                'type': np.random.choice(anomaly_types),
                'severity': np.random.choice(['low', 'medium', 'high'], p=[0.3, 0.5, 0.2])
            }
            anomaly_metadata.append(metadata)
        
        return data_copy, anomaly_metadata
    
    def generate_kafka_message(self, row):
        """Convert data row to Kafka message format"""
        return {
            'timestamp': row['timestamp'].isoformat(),
            'metrics': {
                'cpu_usage': float(row['cpu_usage']),
                'memory_usage': float(row['memory_usage']),
                'network_io': float(row['network_io']),
                'disk_io': float(row['disk_io']),
                'response_time': float(row['response_time'])
            },
            'is_anomaly': int(row['is_anomaly']),
            'source': 'synthetic_generator'
        }
    
    def export_data(self, data, filepath, format='csv'):
        """Export generated data to file"""
        if format.lower() == 'csv':
            data.to_csv(filepath, index=False)
        elif format.lower() == 'json':
            data.to_json(filepath, orient='records', date_format='iso')
        elif format.lower() == 'parquet':
            data.to_parquet(filepath, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Data exported to {filepath} in {format} format")
    
    def generate_batch_data(self, batch_size=1000, n_batches=10, anomaly_rate=0.05):
        """Generate multiple batches of data for streaming simulation"""
        batches = []
        current_time = datetime(2025, 8, 1)
        
        for batch_num in range(n_batches):
            print(f"Generating batch {batch_num + 1}/{n_batches}...")
            
            # Generate normal data for this batch
            batch_data = self.generate_normal_data(batch_size, current_time)
            
            # Inject anomalies
            batch_data, _ = self.inject_anomalies(batch_data, anomaly_rate)
            
            # Add batch metadata
            batch_data['batch_id'] = batch_num
            
            batches.append(batch_data)
            current_time += timedelta(minutes=batch_size)
        
        # Combine all batches
        full_data = pd.concat(batches, ignore_index=True)
        return full_data, batches
    
    def add_seasonal_patterns(self, data):
        """Add more complex seasonal patterns to existing data"""
        data_copy = data.copy()
        n_samples = len(data)
        time_index = np.arange(n_samples)
        
        # Add business hours pattern (higher load during 9-5)
        business_hours = np.sin(time_index * 2 * np.pi / 480) * 0.5 + 0.5  # 8-hour cycle
        business_multiplier = 0.7 + 0.6 * business_hours
        
        data_copy['cpu_usage'] *= business_multiplier
        data_copy['memory_usage'] *= (0.8 + 0.4 * business_hours)
        data_copy['network_io'] *= business_multiplier
        data_copy['response_time'] *= (0.7 + 0.6 * business_hours)
        
        return data_copy