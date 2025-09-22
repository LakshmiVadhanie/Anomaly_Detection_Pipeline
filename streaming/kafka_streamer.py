"""
Kafka Streaming Components
Producer and Consumer for real-time log streaming
"""

import json
import time
import threading
from datetime import datetime
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KafkaStreamer:
    """Kafka producer and consumer for streaming log data"""
    
    def __init__(self, bootstrap_servers='localhost:9092', topic='log_stream'):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.producer = None
        self.consumer = None
        
    def create_producer(self):
        """Create Kafka producer with proper configuration"""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda x: json.dumps(x).encode('utf-8'),
                key_serializer=lambda x: str(x).encode('utf-8') if x else None,
                acks='all',  # Wait for all replicas to acknowledge
                retries=3,
                batch_size=16384,
                linger_ms=10,
                buffer_memory=33554432,
                compression_type='gzip'
            )
            logger.info(f"Kafka producer created for topic: {self.topic}")
            return True
        except Exception as e:
            logger.error(f"Failed to create Kafka producer: {e}")
            return False
    
    def create_consumer(self, group_id='anomaly_detection_group'):
        """Create Kafka consumer with proper configuration"""
        try:
            self.consumer = KafkaConsumer(
                self.topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=group_id,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                key_deserializer=lambda m: m.decode('utf-8') if m else None,
                auto_offset_reset='earliest',
                enable_auto_commit=True,
                auto_commit_interval_ms=1000,
                session_timeout_ms=30000,
                max_poll_records=100
            )
            logger.info(f"Kafka consumer created for topic: {self.topic}")
            return True
        except Exception as e:
            logger.error(f"Failed to create Kafka consumer: {e}")
            return False
    
    def produce_logs(self, data, batch_size=100, delay=0.1, key_column=None):
        """Produce log data to Kafka topic"""
        if not self.create_producer():
            return False
        
        try:
            total_records = len(data)
            produced_count = 0
            failed_count = 0
            
            logger.info(f"Starting to produce {total_records} log records...")
            
            for i in range(0, total_records, batch_size):
                batch = data.iloc[i:i+batch_size]
                batch_start_time = time.time()
                
                for _, row in batch.iterrows():
                    try:
                        # Create log record
                        log_record = self.format_log_record(row)
                        
                        # Determine key for partitioning
                        key = None
                        if key_column and key_column in row:
                            key = str(row[key_column])
                        
                        # Send to Kafka
                        future = self.producer.send(
                            self.topic, 
                            value=log_record, 
                            key=key
                        )
                        
                        # Add callback for monitoring
                        future.add_callback(self._on_send_success)
                        future.add_errback(self._on_send_error)
                        
                        produced_count += 1
                        
                    except Exception as e:
                        logger.error(f"Failed to send record: {e}")
                        failed_count += 1
                
                # Flush batch
                self.producer.flush()
                
                batch_time = time.time() - batch_start_time
                logger.info(f"Batch {i//batch_size + 1}: Produced {len(batch)} records "
                           f"in {batch_time:.2f}s (Total: {produced_count})")
                
                # Rate limiting
                if delay > 0:
                    time.sleep(delay)
            
            logger.info(f"Production completed. Success: {produced_count}, "
                       f"Failed: {failed_count}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in producer: {e}")
            return False
        
        finally:
            if self.producer:
                self.producer.close()
    
    def consume_logs(self, max_records=1000, timeout_ms=30000, callback=None):
        """Consume log data from Kafka topic"""
        if not self.create_consumer():
            return []
        
        try:
            logger.info("Starting to consume log records...")
            consumed_records = []
            start_time = time.time()
            
            # Set consumer timeout
            self.consumer._consumer_timeout = timeout_ms / 1000
            
            for message in self.consumer:
                try:
                    # Process message
                    record = {
                        'partition': message.partition,
                        'offset': message.offset,
                        'timestamp': datetime.fromtimestamp(message.timestamp / 1000),
                        'key': message.key,
                        'value': message.value
                    }
                    
                    consumed_records.append(record)
                    
                    # Call callback if provided
                    if callback:
                        callback(record)
                    
                    # Check limits
                    if len(consumed_records) >= max_records:
                        logger.info(f"Reached max_records limit: {max_records}")
                        break
                        
                    # Check timeout
                    if time.time() - start_time > timeout_ms / 1000:
                        logger.info("Consumption timeout reached")
                        break
                        
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    continue
            
            logger.info(f"Consumed {len(consumed_records)} records")
            return consumed_records
            
        except Exception as e:
            logger.error(f"Error in consumer: {e}")
            return []
        
        finally:
            if self.consumer:
                self.consumer.close()
    
    def format_log_record(self, row):
        """Convert pandas row to Kafka log record format"""
        log_record = {
            'timestamp': row['timestamp'].isoformat() if 'timestamp' in row else datetime.now().isoformat(),
            'metrics': {
                'cpu_usage': float(row.get('cpu_usage', 0)),
                'memory_usage': float(row.get('memory_usage', 0)),
                'network_io': float(row.get('network_io', 0)),
                'disk_io': float(row.get('disk_io', 0)),
                'response_time': float(row.get('response_time', 0))
            },
            'metadata': {
                'is_anomaly': int(row.get('is_anomaly', 0)),
                'source': 'log_generator',
                'batch_id': row.get('batch_id', 0) if 'batch_id' in row else None
            }
        }
        return log_record
    
    def _on_send_success(self, record_metadata):
        """Callback for successful message send"""
        logger.debug(f"Message sent successfully: topic={record_metadata.topic}, "
                    f"partition={record_metadata.partition}, offset={record_metadata.offset}")
    
    def _on_send_error(self, exception):
        """Callback for failed message send"""
        logger.error(f"Failed to send message: {exception}")

class StreamingProcessor:
    """High-level streaming processor with anomaly detection"""
    
    def __init__(self, kafka_config, anomaly_detector=None):
        self.kafka_streamer = KafkaStreamer(
            bootstrap_servers=kafka_config.get('bootstrap_servers', 'localhost:9092'),
            topic=kafka_config.get('topic', 'log_stream')
        )
        self.anomaly_detector = anomaly_detector
        self.processing_stats = {
            'total_processed': 0,
            'anomalies_detected': 0,
            'processing_errors': 0,
            'start_time': None
        }
    
    def start_streaming_detection(self, max_records=10000, batch_size=50):
        """Start real-time anomaly detection on streaming data"""
        logger.info("Starting streaming anomaly detection...")
        
        self.processing_stats['start_time'] = time.time()
        buffer = []
        
        def process_message(record):
            try:
                # Extract metrics from message
                metrics = record['value']['metrics']
                feature_vector = [
                    metrics['cpu_usage'],
                    metrics['memory_usage'],
                    metrics['network_io'],
                    metrics['disk_io'],
                    metrics['response_time']
                ]
                
                buffer.append({
                    'timestamp': record['timestamp'],
                    'features': feature_vector,
                    'raw_data': record['value']
                })
                
                # Process batch when buffer is full
                if len(buffer) >= batch_size:
                    self.process_batch(buffer)
                    buffer.clear()
                
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                self.processing_stats['processing_errors'] += 1
        
        # Start consuming with callback
        records = self.kafka_streamer.consume_logs(
            max_records=max_records,
            callback=process_message
        )
        
        # Process remaining buffer
        if buffer:
            self.process_batch(buffer)
        
        self.print_processing_stats()
        return records
    
    def process_batch(self, batch):
        """Process a batch of records for anomaly detection"""
        if not self.anomaly_detector:
            logger.warning("No anomaly detector provided")
            return
        
        try:
            # Extract features
            features = np.array([record['features'] for record in batch])
            
            # Normalize features
            features_scaled = self.anomaly_detector.scaler.transform(features)
            
            # Predict anomalies
            result = self.anomaly_detector.predict_anomalies(features_scaled)
            anomalies = result['predictions']
            scores = result['ensemble_scores']
            
            # Update stats
            self.processing_stats['total_processed'] += len(batch)
            self.processing_stats['anomalies_detected'] += anomalies.sum()
            
            # Log anomalies
            for i, is_anomaly in enumerate(anomalies):
                if is_anomaly:
                    logger.warning(f"ANOMALY DETECTED: "
                                 f"Score={scores[i]:.3f}, "
                                 f"Timestamp={batch[i]['timestamp']}")
        
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            self.processing_stats['processing_errors'] += len(batch)
    
    def print_processing_stats(self):
        """Print processing statistics"""
        if self.processing_stats['start_time']:
            duration = time.time() - self.processing_stats['start_time']
            throughput = self.processing_stats['total_processed'] / duration
            
            logger.info("=== Streaming Processing Statistics ===")
            logger.info(f"Total Processed: {self.processing_stats['total_processed']}")
            logger.info(f"Anomalies Detected: {self.processing_stats['anomalies_detected']}")
            logger.info(f"Processing Errors: {self.processing_stats['processing_errors']}")
            logger.info(f"Duration: {duration:.2f} seconds")
            logger.info(f"Throughput: {throughput:.2f} records/second")
            logger.info(f"Anomaly Rate: {self.processing_stats['anomalies_detected'] / max(1, self.processing_stats['total_processed']) * 100:.2f}%")