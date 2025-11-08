"""Search service for job search operations"""
import logging
import time
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime, timezone
from services.milvus_service import MilvusService
from app.config import Config
from models.search import SearchWeights
from models.embeddings import Embeddings
from models.pagination import PaginationInfo
from pymilvus import AnnSearchRequest, WeightedRanker

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
        threshold: float = Config.SEARCH_THRESHOLD,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[int], PaginationInfo]:
        """Perform hybrid search for jobs with optional filters
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            offset: Offset for pagination
            threshold: Minimum similarity score threshold
            filters: Optional filter dictionary with filterable fields
            
        Returns:
            Tuple of (job_ids, pagination_info)
        """
        query = query.lower()
        logger.info(f"Search query: '{query}'")
        logger.info(f"Hybrid search with offset={offset}, limit={limit}")
        
        # Build filter expression from filters dict
        filter_expr = self._build_filter_expression(filters)
        if filter_expr:
            logger.info(f"Applied filter expression: {filter_expr}")
        
        query_embedding = self.milvus_service.generate_embeddings([query])
        dense_vec = query_embedding.get("dense")[0]
        sparse_vec = query_embedding.get("sparse")[0]

        # Normalize sparse row to dict {index: value} if needed
        if hasattr(sparse_vec, "tocoo"):
            coo = sparse_vec.tocoo()
            sparse_vec = {int(j): float(v) for j, v in zip(coo.col, coo.data)}

        dense_req = AnnSearchRequest(
            data=[dense_vec],
            anns_field="dense_vector",
            param={"metric_type": "COSINE"},
            limit=limit,
            expr=filter_expr,
        )
        
        sparse_req = AnnSearchRequest(
            data=[sparse_vec],
            anns_field="sparse_vector",
            param={"metric_type": "IP"},
            limit=limit,
            expr=filter_expr,
        )
        results = self.milvus_service.jobs_collection.hybrid_search(
            reqs=[dense_req, sparse_req],
            rerank=WeightedRanker(float(0.4), float(0.6)),
            offset=offset * limit,
            limit=limit+1,
            output_fields=["id"],
        )
        job_ids = []
        for hits in results:
            for hit in hits:
                if hit.score is None or hit.score < float(threshold):
                    continue
                job_ids.append(hit.entity.get("id"))
                logger.info(
                    f"Hit: id={hit.entity.get('id')}, score={hit.score}, threshold={threshold}"
                )
                
        has_next = len(job_ids) > limit
        job_ids = job_ids[:limit]
        # Build pagination info
        pagination = PaginationInfo(
            limit=limit,
            offset=offset,
            count=len(job_ids),
            has_next=has_next,
            has_prev=offset > 0,
        )

        return job_ids, pagination

    def _build_filter_expression(self, filters: Optional[Dict[str, Any]]) -> Optional[str]:
        if not filters:
            return "status == 'PUBLISHED'"
        
        conditions = []
        
        # Status filter: default to PUBLISHED if not specified
        status = filters.get("status", "PUBLISHED")
        if status:
            status_escaped = str(status).replace("'", "\\'")
            conditions.append(f"status == '{status_escaped}'")
        
        string_fields = {
            "company": "company",
            "jobRole": "job_role",
            "seniority": "seniority",
            "workMode": "work_mode",
            "currency": "currency",
        }
        
        for filter_key, field_name in string_fields.items():
            if filter_key in filters and filters[filter_key]:
                value = str(filters[filter_key]).replace("'", "\\'")
                conditions.append(f"{field_name} == '{value}'")
        
        # Location filter: use LIKE for partial match (contains)
        if "location" in filters and filters["location"]:
            location_value = str(filters["location"]).replace("'", "\\'")
            conditions.append(f"location like '%{location_value}%'")
        
        # salaryMin: greater than or equal
        if "salaryMin" in filters:
            salary_min = filters["salaryMin"]
            if salary_min is not None:
                conditions.append(f"salary_min >= {int(salary_min)}")
        
        # salaryMax: less than or equal
        if "salaryMax" in filters:
            salary_max = filters["salaryMax"]
            if salary_max is not None:
                conditions.append(f"salary_max <= {int(salary_max)}")
        
        # datePosted: can be exact match, range, or relative (e.g., "last_7_days")
        if "datePosted" in filters:
            date_posted = filters["datePosted"]
            if isinstance(date_posted, (list, tuple)) and len(date_posted) == 2:
                # Range: [start_timestamp_ms, end_timestamp_ms]
                conditions.append(f"date_posted >= {int(date_posted[0])}")
                conditions.append(f"date_posted <= {int(date_posted[1])}")
            elif isinstance(date_posted, str):
                # Relative time (e.g., "last_7_days", "last_30_days")
                now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
                if date_posted == "last_7_days":
                    days_ago_ms = now_ms - (7 * 24 * 60 * 60 * 1000)
                    conditions.append(f"date_posted >= {days_ago_ms}")
                elif date_posted == "last_30_days":
                    days_ago_ms = now_ms - (30 * 24 * 60 * 60 * 1000)
                    conditions.append(f"date_posted >= {days_ago_ms}")
            else:
                # Exact match (timestamp in milliseconds)
                conditions.append(f"date_posted == {int(date_posted)}")
        
        # dateExpires: filter for non-expired jobs (default behavior)
        if filters.get("excludeExpired", True):
            now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            conditions.append(f"(date_expires == 0 || date_expires > {now_ms})")
        elif "dateExpires" in filters:
            date_expires = filters["dateExpires"]
            if isinstance(date_expires, (list, tuple)) and len(date_expires) == 2:
                # Range
                conditions.append(f"date_expires >= {int(date_expires[0])}")
                conditions.append(f"date_expires <= {int(date_expires[1])}")
            else:
                # Exact match
                conditions.append(f"date_expires == {int(date_expires)}")
        
        # Combine all conditions with AND
        if conditions:
            return " && ".join(conditions)
        
        # Fallback: if no conditions were built, default to PUBLISHED
        return "status == 'PUBLISHED'"
