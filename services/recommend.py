from typing import Dict, Any, List, Optional
import math
from datetime import datetime, timezone
from collections import Counter

from services.milvus_service import MilvusService
from app.config import INTERACTION_WEIGHTS, Config


class RecommendationService:
    def __init__(self, milvus_service: MilvusService):
        self.milvus_service = milvus_service

    def recommend(self, user_id: int, top_k: int = 20, filters: Optional[Dict] = None):
        """Main recommendation endpoint"""
        # TODO: Implement full recommendation logic
        # 1. Get user vector
        # 2. Search jobs collection
        # 3. Add CF scores (if available)
        # 4. Hybrid scoring
        # 5. Re-rank and return
        return None

    def _calculate_user_vector(
        self, 
        user_profile: Dict[str, Any], 
        user_interactions: Dict[str, Any]
    ) -> List[float]:
        """Calculate user dense vector from profile and interactions.
        
        Uses adaptive weighting based on interaction count:
        - Cold start (<5): 90% profile, 10% behavior
        - Growing (5-20): 60% profile, 40% behavior  
        - Mature (>20): 30% profile, 70% behavior
        """
        
        # Build profile text (with interaction insights if available)
        profile_text = self._build_profile_text(user_profile, user_interactions)
        profile_dense = self._embed_text_to_dense(profile_text)
        
        # Count interactions for adaptive weights
        interaction_count = self._count_total_interactions(user_interactions)
        
        # Adaptive weights
        if interaction_count < 5:
            alpha, beta = 0.9, 0.1  # Cold start
        elif interaction_count < 20:
            alpha, beta = 0.6, 0.4  # Growing
        else:
            alpha, beta = 0.3, 0.7  # Mature
        
        # Compute behavior vector using the model's dense dimension
        behavior_dense = self._compute_behavior_dense(user_interactions, self.milvus_service.dense_dim)
        
        # Combine
        final_dense = self._combine_vectors(profile_dense, behavior_dense, alpha, beta)
        
        # Normalize to unit vector
        norm = math.sqrt(sum(x * x for x in final_dense))
        if norm > 1e-8:
            final_dense = [x / norm for x in final_dense]
        
        # Upsert to Milvus
        user_id = user_profile.get("id")
        if user_id is not None:
            try:
                self.milvus_service.upsert_user_vector(int(user_id), final_dense)
            except Exception as e:
                # Log error but don't fail
                print(f"Warning: Failed to upsert user vector for user {user_id}: {e}")
        
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
        
        acc: List[float] = [0.0] * dimension
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
                
                for i in range(dimension):
                    acc[i] += w * float(job_vec[i])
                weight_sum += abs(w)
        
        # Normalize
        if weight_sum > 1e-8:
            return [v / weight_sum for v in acc]
        
        return [0.0] * dimension

    def _combine_vectors(
        self, 
        a: List[float], 
        b: List[float], 
        wa: float, 
        wb: float
    ) -> List[float]:
        """Combine two vectors with weights"""
        final_dim = max(len(a), len(b))
        if final_dim == 0:
            return []
        
        if len(a) < final_dim:
            a = a + [0.0] * (final_dim - len(a))
        if len(b) < final_dim:
            b = b + [0.0] * (final_dim - len(b))
        
        return [wa * a[i] + wb * b[i] for i in range(final_dim)]

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