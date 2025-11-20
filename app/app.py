"""Flask application factory"""
import logging
import threading
from flask import Flask
from services.search_service import SearchService
from services.recommend import RecommendationService
from services.milvus_service import MilvusService
from app.routes import create_routes
from sync_service.consumer import RedisStreamConsumer

logger = logging.getLogger(__name__)


def create_app() -> Flask:
    """Create and configure Flask application"""
    app = Flask(__name__)

    # Initialize services
    logger.info("Initializing services...")
    try:
        milvus_service = MilvusService()
        search_service = SearchService(milvus_service)
        recommend_service = RecommendationService(milvus_service)
        logger.info("✓ Services initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}", exc_info=True)
        raise

    # Register routes
    create_routes(app, search_service, recommend_service)
    logger.info("✓ Routes registered")

    # Start Redis Stream consumer in background thread
    def start_consumer():
        """Start Redis Stream consumer with error handling"""
        try:
            logger.info("Starting Redis Stream consumer thread...")
            consumer = RedisStreamConsumer()
            logger.info("✓ Consumer initialized, starting to process messages...")
            consumer.run()
        except KeyboardInterrupt:
            logger.info("⚠️  Consumer interrupted by user")
        except Exception as e:
            logger.error(f"❌ Consumer thread error: {e}", exc_info=True)
            logger.warning("⚠️  Consumer thread stopped, but Flask app continues")
    
    consumer_thread = threading.Thread(
        target=start_consumer,
        daemon=True,
        name="RedisStreamConsumer"
    )
    consumer_thread.start()
    logger.info("✓ Redis Stream consumer thread started")

    logger.info("✓ Flask application created successfully")
    return app