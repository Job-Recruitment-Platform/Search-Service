from typing import Dict, Any, List, Optional
import math
import json
from datetime import datetime, timezone
from collections import Counter
import redis
import numpy as np

from services.milvus_service import MilvusService
from app.config import INTERACTION_WEIGHTS, Config


class RecommendationService:
    def __init__(self, milvus_service: MilvusService):
        self.milvus_service = milvus_service
        # Initialize Redis client for short-term vector caching
        try:
            self.redis_client = redis.Redis(
                host=Config.REDIS_HOST,
                port=Config.REDIS_PORT,
                db=Config.REDIS_DB,
                decode_responses=False  # We'll handle JSON encoding/decoding manually
            )
            # Test connection
            self.redis_client.ping()
        except Exception as e:
            print(f"Warning: Failed to connect to Redis: {e}")
            self.redis_client = None

    def recommend(
        self, 
        user_id: int, 
        top_k: int = 20, 
        filters: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Main recommendation endpoint - MVP version
        
        Args:
            user_id: User ID to get recommendations for
            top_k: Number of recommendations to return
            filters: Optional filters (e.g., location, job_type, salary_range)
        
        Returns:
            List of job recommendations with scores
        """
        try:
            # 1. Get user data
            user_profile = self._get_user_profile(user_id)
            user_interactions = self._get_user_interactions(user_id)
            
            if not user_profile:
                # Cold start: return popular jobs
                return self._get_popular_jobs(top_k, filters)
            
            # 2. Calculate user vector (combines profile + interactions)
            user_vector = self._calculate_user_vector(user_profile, user_interactions)
            
            if not user_vector:
                return self._get_popular_jobs(top_k, filters)
            
            # 3. Search Milvus for similar jobs
            search_limit = top_k * 3  # Get more candidates for re-ranking
            
            search_results = self.milvus_service.search(
                collection_name="jobs",
                query_vectors=[user_vector],
                limit=search_limit,
                output_fields=["job_id", "title", "company", "location", "salary_range"],
                # Apply filters if provided
                expr=self._build_filter_expr(filters) if filters else None
            )
            
            if not search_results or not search_results[0]:
                return self._get_popular_jobs(top_k, filters)
            
            # 4. Format results
            recommendations = []
            for hit in search_results[0][:top_k]:  # Take top_k after search
                recommendations.append({
                    "job_id": hit.get("job_id") or hit.id,
                    "title": hit.get("title", ""),
                    "company": hit.get("company", ""),
                    "location": hit.get("location", ""),
                    "salary_range": hit.get("salary_range", ""),
                    "score": float(hit.score) if hasattr(hit, 'score') else 0.0,
                    "source": "content_based"
                })
            
            return recommendations
            
        except Exception as e:
            print(f"Error in recommend(): {e}")
            # Fallback to popular jobs
            return self._get_popular_jobs(top_k, filters)

    def _calculate_long_term_user_vector(
        self, 
        user_profile: Dict[str, Any]
    ) -> List[float]:
        """Calculate long-term user vector from profile data only.
        
        This represents the user's stable preferences based on their profile
        (skills, education, location, preferences). This vector is saved to Milvus
        as it represents long-term user characteristics.
        
        Args:
            user_profile: User profile dictionary with skills, education, location, etc.
            
        Returns:
            Normalized dense vector representing long-term user preferences
        """
        # Build profile text (without interaction insights for long-term vector)
        profile_text = self._build_profile_text(user_profile, user_interactions=None)
        profile_dense = self._embed_text_to_dense(profile_text)
        
        if not profile_dense:
            return []
        
        # Normalize to unit vector using numpy
        profile_dense = self._normalize_vector(profile_dense)
        
        # Save to Milvus (long-term storage)
        user_id = user_profile.get("id")
        if user_id is not None:
            try:
                self.milvus_service.upsert_user_vector(int(user_id), profile_dense)
            except Exception as e:
                print(f"Warning: Failed to upsert long-term user vector for user {user_id}: {e}")
        
        return profile_dense

    def _calculate_short_term_user_vector(
        self, 
        user_id: int,
        user_interactions: Dict[str, Any],
        cache_ttl: int = 3600
    ) -> List[float]:
        """Calculate short-term user vector from recent interactions.
        
        This represents the user's current interests based on their recent behavior
        (clicks, saves, applies). This vector is cached in Redis as it changes
        frequently and needs to be updated in real-time.
        
        Args:
            user_id: User ID
            user_interactions: Dictionary of user interactions with timestamps
            cache_ttl: Time-to-live for Redis cache in seconds (default: 1 hour)
            
        Returns:
            Normalized dense vector representing short-term user interests
        """
        # Check Redis cache first
        cache_key = f"user_vector:short_term:{user_id}"
        if self.redis_client:
            try:
                cached = self.redis_client.get(cache_key)
                if cached:
                    return json.loads(cached)
            except Exception as e:
                print(f"Warning: Failed to read from Redis cache: {e}")
        
        # Compute behavior vector from interactions
        behavior_dense = self._compute_behavior_dense(
            user_interactions, 
            self.milvus_service.dense_dim
        )
        
        if not behavior_dense:
            return []
        
        # Normalize to unit vector using numpy
        behavior_dense = self._normalize_vector(behavior_dense)
        
        # Cache in Redis (short-term storage)
        if self.redis_client:
            try:
                self.redis_client.setex(
                    cache_key,
                    cache_ttl,
                    json.dumps(behavior_dense)
                )
            except Exception as e:
                print(f"Warning: Failed to cache short-term user vector in Redis: {e}")
        
        return behavior_dense

    def invalidate_short_term_cache(self, user_id: int) -> None:
        """Invalidate the short-term user vector cache in Redis.
        
        Call this method when new user interactions are recorded to ensure
        the short-term vector is recalculated on the next request.
        
        Args:
            user_id: User ID whose cache should be invalidated
        """
        if self.redis_client:
            try:
                cache_key = f"user_vector:short_term:{user_id}"
                self.redis_client.delete(cache_key)
            except Exception as e:
                print(f"Warning: Failed to invalidate short-term cache for user {user_id}: {e}")

    def _calculate_user_vector(
        self, 
        user_profile: Dict[str, Any], 
        user_interactions: Dict[str, Any]
    ) -> List[float]:
        """Calculate combined user dense vector from profile and interactions.
        
        This method combines long-term (profile) and short-term (interactions) vectors
        using adaptive weighting based on interaction count:
        - Cold start (<5): 90% profile, 10% behavior
        - Growing (5-20): 60% profile, 40% behavior  
        - Mature (>20): 30% profile, 70% behavior
        
        Note: This method uses the separate long-term and short-term calculation methods
        internally. For better performance, consider using those methods directly.
        """
        # Get long-term vector (from profile, saved in Milvus)
        long_term_vector = self._calculate_long_term_user_vector(user_profile)
        
        # Get short-term vector (from interactions, cached in Redis)
        user_id = user_profile.get("id")
        if user_id is None:
            return long_term_vector
        
        short_term_vector = self._calculate_short_term_user_vector(
            int(user_id),
            user_interactions
        )
        
        # If no interactions, return long-term vector only
        if not short_term_vector:
            return long_term_vector
        
        # Count interactions for adaptive weights
        interaction_count = self._count_total_interactions(user_interactions)
        
        # Adaptive weights
        if interaction_count < 5:
            alpha, beta = 0.9, 0.1  # Cold start
        elif interaction_count < 20:
            alpha, beta = 0.6, 0.4  # Growing
        else:
            alpha, beta = 0.3, 0.7  # Mature
        
        # Combine long-term and short-term vectors
        final_dense = self._combine_vectors(long_term_vector, short_term_vector, alpha, beta)
        
        # Normalize to unit vector using numpy
        final_dense = self._normalize_vector(final_dense)
        
        return final_dense

    def _count_total_interactions(self, interactions: Dict[str, Any]) -> int:
        """Count total number of interactions across all types"""
        count = 0
        if not isinstance(interactions, dict):
            return 0
        
        for entries in interactions.values():
            if isinstance(entries, dict):
                count += len(entries)
            elif isinstance(entries, (list, tuple, set)):
                count += len(entries)
        return count

    def _to_text(self, value: Any) -> str:
        """Convert any value to text representation"""
        if value is None:
            return ""
        if isinstance(value, (list, tuple, set)):
            return ", ".join(str(v) for v in value if v is not None)
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, bool):
            return "yes" if value else "no"
        return str(value)

    def _build_profile_text(
        self, 
        user_profile: Dict[str, Any],
        user_interactions: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build text representation of user profile"""
        parts: List[str] = []
        
        # Skills
        skills = user_profile.get("skills")
        if skills:
            parts.append(f"Skills: {self._to_text(skills)}")
        
        # Education
        education = user_profile.get("education")
        if education:
            parts.append(f"Education: {self._to_text(education)}")
        
        # Location
        location = user_profile.get("location")
        if location:
            parts.append(f"Location: {self._to_text(location)}")
        
        # Preferences
        preferences = user_profile.get("preferences") or {}
        if isinstance(preferences, dict):
            if "remote" in preferences:
                parts.append(f"Prefers remote: {self._to_text(preferences.get('remote'))}")
            if "relocation" in preferences:
                parts.append(f"Open to relocation: {self._to_text(preferences.get('relocation'))}")
        
        # Add interaction insights if available
        if user_interactions:
            insights = self._extract_interaction_insights(user_interactions)
            if insights.get('preferred_skills'):
                parts.append(f"Interested in: {', '.join(insights['preferred_skills'][:5])}")
        
        text = "\n".join(parts).strip()
        return text or "No profile data"

    def _extract_interaction_insights(self, interactions: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extract insights from positive interactions"""
        # Collect positive interaction job IDs
        positive_job_ids = []
        positive_types = {"APPLY", "SAVE", "CLICK_FROM_SEARCH"}
        
        for key, entries in interactions.items():
            if str(key).upper() not in positive_types:
                continue
            
            if isinstance(entries, dict):
                positive_job_ids.extend(str(jid) for jid in entries.keys())
            elif isinstance(entries, (list, tuple, set)):
                positive_job_ids.extend(str(jid) for jid in entries)
        
        if not positive_job_ids:
            return {}
        
        # Get job metadata
        try:
            jobs_metadata = self._get_jobs_metadata(positive_job_ids)
        except Exception:
            return {}
        
        # Extract patterns
        skills = []
        for job in jobs_metadata:
            if job.get('required_skills'):
                skills.extend(job['required_skills'])
        
        return {
            'preferred_skills': [s for s, _ in Counter(skills).most_common(10)]
        }

    def _get_jobs_metadata(self, job_ids: List[str]) -> List[Dict]:
        """Get job metadata - implement based on your architecture"""
        # TODO: Query from database or job service
        # Should return list of dicts with 'required_skills', 'industry', etc.
        return []

    def _embed_text_to_dense(self, text: str) -> List[float]:
        """Embed text to dense vector using BGE-M3"""
        embeddings = self.milvus_service.generate_embeddings([text])
        dense = embeddings["dense"] if embeddings else [[]]
        return dense if dense else []

    def _compute_behavior_dense(
        self, 
        interactions: Dict[str, Any], 
        dimension: int
    ) -> List[float]:
        """Compute behavior vector from interactions with time decay"""
        
        allowed_keys = {
            "APPLY", "SAVE", "CLICK",
            "CLICK_FROM_SIMILAR", "CLICK_FROM_RECOMMENDED", "CLICK_FROM_SEARCH",
            "SKIP_FROM_SIMILAR", "SKIP_FROM_RECOMMENDED", "SKIP_FROM_SEARCH",
        }
        
        half_life_days = getattr(Config, "INTERACTION_HALF_LIFE_DAYS", 30)
        now_ts = datetime.now(timezone.utc).timestamp()
        
        # Use numpy array for efficient accumulation
        acc = np.zeros(dimension, dtype=np.float32)
        weight_sum: float = 0.0
        
        if not isinstance(interactions, dict):
            return [0.0] * dimension
        
        for raw_key, entries in interactions.items():
            key_upper = str(raw_key).upper()
            if key_upper not in allowed_keys or key_upper not in INTERACTION_WEIGHTS:
                continue
            
            base_w = float(INTERACTION_WEIGHTS[key_upper])
            
            if isinstance(entries, dict):
                items = entries.items()
            elif isinstance(entries, (list, tuple, set)):
                items = [(jid, None) for jid in entries]
            else:
                continue
            
            for job_id, ts in items:
                try:
                    j_id = int(job_id)
                except Exception:
                    continue
                
                job_vec = self.milvus_service.get_job_dense_vector(j_id)
                if not job_vec or len(job_vec) != dimension:
                    continue
                
                decay = self._exp_time_decay(ts, now_ts, half_life_days)
                w = base_w * decay
                
                # Use numpy for vector addition
                acc += w * np.array(job_vec, dtype=np.float32)
                weight_sum += abs(w)
        
        # Normalize using numpy
        if weight_sum > 1e-8:
            acc = acc / weight_sum
            return acc.tolist()
        
        return [0.0] * dimension

    def _normalize_vector(self, vector: List[float]) -> List[float]:
        """Normalize a vector to unit length using numpy.
        
        Args:
            vector: Input vector as list of floats
            
        Returns:
            Normalized vector as list of floats
        """
        if not vector:
            return []
        
        vec = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(vec)
        
        if norm > 1e-8:
            vec = vec / norm
        
        return vec.tolist()

    def _combine_vectors(
        self, 
        a: List[float], 
        b: List[float], 
        wa: float, 
        wb: float
    ) -> List[float]:
        """Combine two vectors with weights using numpy.
        
        Args:
            a: First vector
            b: Second vector
            wa: Weight for first vector
            wb: Weight for second vector
            
        Returns:
            Combined weighted vector
        """
        if not a and not b:
            return []
        if not a:
            return b
        if not b:
            return a
        
        # Convert to numpy arrays
        vec_a = np.array(a, dtype=np.float32)
        vec_b = np.array(b, dtype=np.float32)
        
        # Pad shorter vector with zeros if needed
        max_dim = max(len(vec_a), len(vec_b))
        if len(vec_a) < max_dim:
            vec_a = np.pad(vec_a, (0, max_dim - len(vec_a)), mode='constant')
        if len(vec_b) < max_dim:
            vec_b = np.pad(vec_b, (0, max_dim - len(vec_b)), mode='constant')
        
        # Weighted combination
        result = wa * vec_a + wb * vec_b
        return result.tolist()

    def _exp_time_decay(
        self, 
        ts: Any, 
        now_ts: float, 
        half_life_days: float
    ) -> float:
        """Exponential time decay with half-life in days"""
        if ts is None:
            return 1.0
        
        try:
            ts_float = float(ts)
        except Exception:
            return 1.0
        
        try:
            delta_days = max(0.0, (now_ts - ts_float) / 86400.0)
            return math.exp(-math.log(2) * (delta_days / float(half_life_days)))
        except Exception:
            return 1.0