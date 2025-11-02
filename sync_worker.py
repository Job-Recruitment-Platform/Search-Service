"""Entry point for the sync service worker"""
import logging
import signal
import sys
from sync_service.consumer import RedisStreamConsumer

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def signal_handler(sig, frame):
    """Handle shutdown signals"""
    logger.info("Received shutdown signal")
    if consumer:
        consumer.stop()
    sys.exit(0)


if __name__ == "__main__":
    consumer = None
    try:
        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        logger.info("=" * 60)
        logger.info("Starting Sync Worker")
        logger.info("=" * 60)
        
        # Create and run consumer
        consumer = RedisStreamConsumer()
        logger.info("Consumer initialized, starting to process messages...")
        consumer.run()

    except Exception as e:
        logger.exception(f"Failed to start sync worker: {e}")
        if consumer:
            consumer.stop()
        sys.exit(1)

