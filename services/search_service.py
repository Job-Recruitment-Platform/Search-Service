"""Search service for job search operations"""
import logging
import time
from typing import List
from services.milvus_service import MilvusService
from app.config import Config
from models.search import SearchResult, SearchWeights
from models.embeddings import Embeddings

logger = logging.getLogger(__name__)


class SearchService:
    """Service for job search operations"""

    def __init__(self, milvus_service: MilvusService):
        self.milvus_service = milvus_service

    def search(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ) -> List[dict]:
        """Perform hybrid search for jobs"""
        start = time.time()

        try:
            embeddings_dict = self.milvus_service.generate_embeddings([query])
            embeddings = Embeddings.from_dict(embeddings_dict)
            logger.info(f"Generated embeddings in {time.time() - start:.2f}s")
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise

        results = self.milvus_service.hybrid_search(
            embeddings.get_dense_vector(0),
            embeddings.get_sparse_vector(0),
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            limit=limit,
            offset=offset,
        )

        logger.info(
            f"Search completed in {time.time() - start:.2f}s, found {len(results)} results"
        )

        formatted_results = []
        for hit in results:
            search_result = SearchResult.from_milvus_hit(hit)
            if search_result.score < Config.SEARCH_SCORE_THRESHOLD:
                continue
            formatted_results.append(search_result.to_dict())

        return formatted_results

