"""Redis stream consumer for job updates"""
import logging
import time
import redis
from app.config import Config
from sync_service.sync_processor import SyncProcessor
from services.milvus_service import MilvusService

logger = logging.getLogger(__name__)


class RedisStreamConsumer:
    """Consumer for Redis streams that processes job updates"""

    def __init__(self):
        self.redis_client = redis.Redis(
            host=Config.REDIS_HOST,
            port=Config.REDIS_PORT,
            db=Config.REDIS_DB,
            decode_responses=True,
        )
        self.stream_name = Config.REDIS_STREAM_NAME
        self.consumer_group = Config.REDIS_CONSUMER_GROUP
        self.consumer_name = Config.REDIS_CONSUMER_NAME
        self.milvus_service = MilvusService()
        self.sync_processor = SyncProcessor(self.milvus_service)
        self.running = False
        self._setup_consumer_group()

    def _setup_consumer_group(self):
        """
        Setup Redis consumer group for outbox pattern
        
        Creates consumer group if it doesn't exist, following the pattern
        described in OUTBOX_PATTERN_ARCHITECTURE.md
        """
        try:
            # Check if stream exists
            try:
                stream_info = self.redis_client.xinfo_stream(self.stream_name)
                logger.info(
                    f"Stream '{self.stream_name}' exists with {stream_info.get('length', 0)} messages"
                )
            except redis.exceptions.ResponseError:
                logger.warning(
                    f"Stream '{self.stream_name}' does not exist yet. "
                    f"It will be created when first message arrives."
                )
            
            # Try to create consumer group starting from the beginning (id="0")
            # mkstream=True ensures stream is created if it doesn't exist
            self.redis_client.xgroup_create(
                name=self.stream_name,
                groupname=self.consumer_group,
                id="0",
                mkstream=True,
            )
            logger.info(
                f"✓ Created consumer group '{self.consumer_group}' for stream '{self.stream_name}'"
            )
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.info(
                    f"Consumer group '{self.consumer_group}' already exists for stream '{self.stream_name}'"
                )
                # Check pending messages
                try:
                    pending = self.redis_client.xpending_range(
                        name=self.stream_name,
                        groupname=self.consumer_group,
                        min="-",
                        max="+",
                        count=10
                    )
                    if pending:
                        logger.info(f"Found {len(pending)} pending message(s) in consumer group")
                except:
                    pass
            else:
                logger.error(f"Failed to create consumer group: {e}")
                raise

    def process_messages(self, count: int = 10, block: int = 5000):
        """
        Read and process messages from Redis stream
        
        Follows the outbox pattern implementation:
        - Reads from consumer group using xreadgroup
        - Processes messages with 8 fields (id, aggregateType, aggregateId, 
          eventType, payload, occurredAt, traceId, attempts)
        - Acknowledges messages after successful processing
        """
        try:
            # Read from stream using consumer group (">" means new messages)
            # If no new messages, this will block for 'block' milliseconds
            messages = self.redis_client.xreadgroup(
                groupname=self.consumer_group,
                consumername=self.consumer_name,
                streams={self.stream_name: ">"},
                count=count,
                block=block,
            )

            if not messages:
                logger.debug(f"No new messages in stream '{self.stream_name}'")
                return 0

            processed_count = 0
            for stream, message_list in messages:
                logger.info(f"Received {len(message_list)} message(s) from stream '{stream}'")
                
                for message_id, fields in message_list:
                    try:
                        logger.info(
                            f"Processing message {message_id} from stream '{stream}'"
                        )                        
                        # Process the message fields directly
                        # Fields contain: id, aggregateType, aggregateId, eventType,
                        # payload, occurredAt, traceId, attempts
                        result = self.sync_processor.process_stream_message(fields)
                        processed_count += result.processed

                        # Acknowledge message after successful processing
                        self.redis_client.xack(
                            self.stream_name, self.consumer_group, message_id
                        )
                        logger.info(
                            f"✓ Acknowledged message {message_id}, "
                            f"result: {result.to_dict()}"
                        )

                    except Exception as e:
                        logger.exception(
                            f"✗ Failed to process message {message_id}: {e}"
                        )
                        logger.error(f"Message fields: {fields}")
                        # Optionally: move to dead letter queue or retry logic
                        # For now, we'll still ack it to avoid infinite retries
                        # In production, you might want to:
                        # - Track failed messages
                        # - Implement retry with backoff
                        # - Move to DLQ after N attempts
                        self.redis_client.xack(
                            self.stream_name, self.consumer_group, message_id
                        )

            return processed_count

        except redis.exceptions.ConnectionError as e:
            logger.error(f"Redis connection error: {e}")
            raise
        except Exception as e:
            logger.exception(f"Error processing messages: {e}")
            return 0

    def run(self):
        """
        Run the consumer continuously
        
        Continuously reads from Redis stream and processes outbox events.
        Handles interrupts gracefully and implements retry logic for failures.
        """
        self.running = True
        logger.info(
            f"Starting Redis stream consumer: consumer='{self.consumer_name}' "
            f"group='{self.consumer_group}' stream='{self.stream_name}'"
        )

        while self.running:
            try:
                processed = self.process_messages()
                if processed > 0:
                    logger.debug(f"Processed {processed} messages in this cycle")
            except KeyboardInterrupt:
                logger.info("Received interrupt signal, shutting down gracefully...")
                self.stop()
                break
            except Exception as e:
                logger.exception(f"Error in consumer loop: {e}")
                time.sleep(5)  # Wait before retrying on error

    def stop(self):
        """Stop the consumer"""
        logger.info("Stopping Redis stream consumer...")
        self.running = False
        if self.redis_client:
            self.redis_client.close()

