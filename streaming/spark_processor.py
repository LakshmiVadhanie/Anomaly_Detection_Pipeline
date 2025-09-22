"""
Spark Structured Streaming Processor
Real-time processing of log streams with aggregations and anomaly detection
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.streaming import *
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SparkProcessor:
    """Spark Structured Streaming processor for anomaly detection pipeline"""
    
    def __init__(self, app_name="AnomalyDetection"):
        self.app_name = app_name
        self.spark = self._create_spark_session()
        
    def _create_spark_session(self):
        """Create Spark session with optimized configuration"""
        try:
            spark = SparkSession.builder \
                .appName(self.app_name) \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                .config("spark.sql.streaming.checkpointLocation", "/tmp/spark_checkpoint") \
                .config("spark.streaming.backpressure.enabled", "true") \
                .config("spark.sql.streaming.kafka.allowNonConsecutiveOffsets", "true") \
                .getOrCreate()
            
            spark.sparkContext.setLogLevel("WARN")
            logger.info(f"Spark session created: {self.app_name}")
            return spark
            
        except Exception as e:
            logger.error(f"Failed to create Spark session: {e}")
            raise
    
    def define_log_schema(self):
        """Define schema for incoming log data"""
        return StructType([
            StructField("timestamp", StringType(), True),
            StructField("metrics", StructType([
                StructField("cpu_usage", DoubleType(), True),
                StructField("memory_usage", DoubleType(), True),
                StructField("network_io", DoubleType(), True),
                StructField("disk_io", DoubleType(), True),
                StructField("response_time", DoubleType(), True)
            ]), True),
            StructField("metadata", StructType([
                StructField("is_anomaly", IntegerType(), True),
                StructField("source", StringType(), True),
                StructField("batch_id", IntegerType(), True)
            ]), True)
        ])
    
    def create_kafka_stream(self, kafka_servers, topic, starting_offsets="latest"):
        """Create Kafka streaming DataFrame"""
        try:
            df = self.spark \
                .readStream \
                .format("kafka") \
                .option("kafka.bootstrap.servers", kafka_servers) \
                .option("subscribe", topic) \
                .option("startingOffsets", starting_offsets) \
                .option("failOnDataLoss", "false") \
                .load()
            
            logger.info(f"Kafka stream created for topic: {topic}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to create Kafka stream: {e}")
            raise
    
    def parse_log_data(self, kafka_df):
        """Parse JSON log data from Kafka messages"""
        schema = self.define_log_schema()
        
        # Parse JSON value
        parsed_df = kafka_df.select(
            col("key").cast("string").alias("message_key"),
            col("timestamp").alias("kafka_timestamp"),
            from_json(col("value").cast("string"), schema).alias("data"),
            col("partition"),
            col("offset")
        )
        
        # Flatten structure
        flattened_df = parsed_df.select(
            col("message_key"),
            col("kafka_timestamp"),
            col("partition"),
            col("offset"),
            to_timestamp(col("data.timestamp")).alias("log_timestamp"),
            col("data.metrics.cpu_usage").alias("cpu_usage"),
            col("data.metrics.memory_usage").alias("memory_usage"),
            col("data.metrics.network_io").alias("network_io"),
            col("data.metrics.disk_io").alias("disk_io"),
            col("data.metrics.response_time").alias("response_time"),
            col("data.metadata.is_anomaly").alias("is_anomaly"),
            col("data.metadata.source").alias("source"),
            col("data.metadata.batch_id").alias("batch_id")
        ).withColumn("processing_time", current_timestamp())
        
        return flattened_df
    
    def create_windowed_aggregations(self, df, window_duration="1 minute", slide_duration="30 seconds"):
        """Create windowed aggregations for monitoring"""
        
        # Add watermark for handling late data
        watermarked_df = df.withWatermark("log_timestamp", "2 minutes")
        
        # Windowed aggregations
        windowed_df = watermarked_df \
            .groupBy(
                window(col("log_timestamp"), window_duration, slide_duration),
                col("source")
            ).agg(
                count("*").alias("record_count"),
                avg("cpu_usage").alias("avg_cpu_usage"),
                max("cpu_usage").alias("max_cpu_usage"),
                avg("memory_usage").alias("avg_memory_usage"),
                max("memory_usage").alias("max_memory_usage"),
                avg("network_io").alias("avg_network_io"),
                avg("disk_io").alias("avg_disk_io"),
                avg("response_time").alias("avg_response_time"),
                max("response_time").alias("max_response_time"),
                sum("is_anomaly").alias("anomaly_count"),
                (sum("is_anomaly") / count("*") * 100).alias("anomaly_rate")
            ).select(
                col("window.start").alias("window_start"),
                col("window.end").alias("window_end"),
                col("source"),
                col("record_count"),
                round(col("avg_cpu_usage"), 2).alias("avg_cpu_usage"),
                col("max_cpu_usage"),
                round(col("avg_memory_usage"), 2).alias("avg_memory_usage"),
                col("max_memory_usage"),
                round(col("avg_network_io"), 2).alias("avg_network_io"),
                round(col("avg_disk_io"), 2).alias("avg_disk_io"),
                round(col("avg_response_time"), 2).alias("avg_response_time"),
                col("max_response_time"),
                col("anomaly_count"),
                round(col("anomaly_rate"), 2).alias("anomaly_rate")
            )
        
        return windowed_df
    
    def create_anomaly_alerts(self, df):
        """Create real-time anomaly alerts"""
        
        # Define anomaly thresholds
        anomaly_conditions = (
            (col("cpu_usage") > 90) |
            (col("memory_usage") > 85) |
            (col("response_time") > 1000) |
            (col("is_anomaly") == 1)
        )
        
        alerts_df = df.filter(anomaly_conditions).select(
            col("log_timestamp"),
            col("processing_time"),
            col("cpu_usage"),
            col("memory_usage"),
            col("network_io"),
            col("disk_io"),
            col("response_time"),
            col("is_anomaly"),
            col("source"),
            when(col("cpu_usage") > 90, "HIGH_CPU")
            .when(col("memory_usage") > 85, "HIGH_MEMORY")
            .when(col("response_time") > 1000, "SLOW_RESPONSE")
            .when(col("is_anomaly") == 1, "ML_DETECTED_ANOMALY")
            .otherwise("UNKNOWN").alias("alert_type")
        )
        
        return alerts_df
    
    def start_console_output(self, df, output_mode="update", trigger_interval="10 seconds"):
        """Start console output stream"""
        try:
            query = df.writeStream \
                .outputMode(output_mode) \
                .format("console") \
                .option("truncate", "false") \
                .option("numRows", 20) \
                .trigger(processingTime=trigger_interval) \
                .start()
            
            logger.info(f"Console output stream started with trigger: {trigger_interval}")
            return query
            
        except Exception as e:
            logger.error(f"Failed to start console output: {e}")
            raise
    
    def start_kafka_output(self, df, output_topic, kafka_servers, output_mode="append"):
        """Start Kafka output stream"""
        try:
            # Convert DataFrame to Kafka format
            kafka_output_df = df.select(
                col("source").alias("key"),
                to_json(struct(col("*"))).alias("value")
            )
            
            query = kafka_output_df.writeStream \
                .format("kafka") \
                .option("kafka.bootstrap.servers", kafka_servers) \
                .option("topic", output_topic) \
                .outputMode(output_mode) \
                .option("checkpointLocation", "/tmp/kafka_checkpoint") \
                .start()
            
            logger.info(f"Kafka output stream started to topic: {output_topic}")
            return query
            
        except Exception as e:
            logger.error(f"Failed to start Kafka output: {e}")
            raise
    
    def process_stream(self, kafka_servers="localhost:9092", input_topic="log_stream", 
                      output_topic="anomaly_alerts", processing_duration=60):
        """Main stream processing pipeline"""
        try:
            logger.info("Starting Spark Structured Streaming pipeline...")
            
            # Create Kafka input stream
            kafka_df = self.create_kafka_stream(kafka_servers, input_topic)
            
            # Parse log data
            parsed_df = self.parse_log_data(kafka_df)
            
            # Create windowed aggregations
            windowed_df = self.create_windowed_aggregations(parsed_df)
            
            # Create anomaly alerts
            alerts_df = self.create_anomaly_alerts(parsed_df)
            
            # Start multiple output streams
            queries = []
            
            # Console output for monitoring
            console_query = self.start_console_output(
                windowed_df, 
                output_mode="update",
                trigger_interval="30 seconds"
            )
            queries.append(console_query)
            
            # Kafka output for alerts
            alert_query = self.start_kafka_output(
                alerts_df,
                output_topic,
                kafka_servers,
                output_mode="append"
            )
            queries.append(alert_query)
            
            # Wait for processing duration
            logger.info(f"Processing streams for {processing_duration} seconds...")
            
            # Monitor query status
            import time
            start_time = time.time()
            while time.time() - start_time < processing_duration:
                for i, query in enumerate(queries):
                    if query.isActive:
                        progress = query.lastProgress
                        if progress:
                            logger.info(f"Query {i+1} - Processed: {progress.get('inputRowsPerSecond', 0)} rows/sec")
                    else:
                        logger.warning(f"Query {i+1} is not active")
                        if query.exception():
                            logger.error(f"Query {i+1} exception: {query.exception()}")
                
                time.sleep(10)
            
            # Stop all queries
            logger.info("Stopping streaming queries...")
            for query in queries:
                query.stop()
            
            logger.info("Spark streaming pipeline completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in stream processing: {e}")
            return False
    
    def create_batch_processor(self):
        """Create batch processing functions for historical data analysis"""
        
        def process_batch_data(input_path, output_path, format="parquet"):
            """Process historical log data in batch mode"""
            try:
                # Read historical data
                if format.lower() == "csv":
                    df = self.spark.read.csv(input_path, header=True, inferSchema=True)
                elif format.lower() == "parquet":
                    df = self.spark.read.parquet(input_path)
                elif format.lower() == "json":
                    df = self.spark.read.json(input_path)
                else:
                    raise ValueError(f"Unsupported format: {format}")
                
                # Perform aggregations
                summary_df = df.groupBy("source") \
                    .agg(
                        count("*").alias("total_records"),
                        avg("cpu_usage").alias("avg_cpu"),
                        max("cpu_usage").alias("max_cpu"),
                        avg("memory_usage").alias("avg_memory"),
                        avg("response_time").alias("avg_response_time"),
                        sum("is_anomaly").alias("total_anomalies")
                    )
                
                # Write results
                summary_df.write.mode("overwrite").parquet(output_path)
                logger.info(f"Batch processing completed: {output_path}")
                
                return summary_df
                
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                raise
        
        return process_batch_data
    
    def stop(self):
        """Stop Spark session"""
        if self.spark:
            self.spark.stop()
            logger.info("Spark session stopped")