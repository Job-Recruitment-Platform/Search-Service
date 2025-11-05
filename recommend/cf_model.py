from typing import Dict, List, Tuple, Optional, Literal
import numpy as np
from scipy.sparse import csr_matrix
import implicit
from implicit.als import AlternatingLeastSquares


class CFModel:
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self.model: Optional[AlternatingLeastSquares] = None
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None

    def train(
        self,
        user_item_matrix: csr_matrix,
        factors: int = 64,
        regularization: float = 0.15,
        iterations: int = 20,
    ) -> "CFModel":
        """Train an implicit ALS model with exploration-friendly parameters."""
        if not isinstance(user_item_matrix, csr_matrix):
            raise TypeError("user_item_matrix must be a scipy.sparse.csr_matrix")

        item_user = user_item_matrix.T.tocsr()

        model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            use_gpu=self.use_gpu,
            calculate_training_loss=False,
            num_threads=0,
        )
        model.fit(item_user)

        self.model = model
        self.user_factors = model.user_factors
        self.item_factors = model.item_factors
        return self

    def score(
        self,
        user_id: int,
        job_id: int,
        user_id_to_index: Dict[int, int],
        item_id_to_index: Dict[int, int],
    ) -> float:
        """Compute CF score for a specific user and job via dot product."""
        if self.user_factors is None or self.item_factors is None:
            return 0.0
        uidx = user_id_to_index.get(int(user_id))
        iidx = item_id_to_index.get(int(job_id))
        if uidx is None or iidx is None:
            return 0.0
        uf = self.user_factors[uidx]
        vf = self.item_factors[iidx]
        return float(np.dot(uf, vf))

    def recommend(
        self,
        user_id: int,
        user_id_to_index: Dict[int, int],
        index_to_item_id: Dict[int, int],
        user_item_matrix: csr_matrix,
        k: int = 10,
        filter_seen: bool = True,
        exploration_strategy: Literal["none", "epsilon_greedy", "thompson", "diversity"] = "none",
        exploration_epsilon: float = 0.15,
        exploration_noise_scale: float = 0.1,
        diversity_weight: float = 0.3,
        random_state: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """Get top-k recommendations with optional exploration strategies."""
        if self.model is None:
            return []

        uidx = user_id_to_index.get(int(user_id))
        if uidx is None:
            return []

        rng = np.random.default_rng(random_state)

        if exploration_strategy == "none":
            return self._top_k_exploit(uidx, user_item_matrix, index_to_item_id, k, filter_seen)
        if exploration_strategy == "epsilon_greedy":
            return self._top_k_epsilon_greedy(uidx, user_item_matrix, index_to_item_id, k, filter_seen, exploration_epsilon, rng)
        if exploration_strategy == "thompson":
            return self._top_k_thompson(uidx, user_item_matrix, index_to_item_id, k, filter_seen, exploration_noise_scale, rng)
        if exploration_strategy == "diversity":
            return self._top_k_with_diversity(uidx, user_item_matrix, index_to_item_id, k, filter_seen, diversity_weight)
        return self._top_k_exploit(uidx, user_item_matrix, index_to_item_id, k, filter_seen)

    def recommend_with_adaptive_exploration(
        self,
        user_id: int,
        user_id_to_index: Dict[int, int],
        index_to_item_id: Dict[int, int],
        user_item_matrix: csr_matrix,
        k: int = 10,
        filter_seen: bool = True,
    ) -> List[Tuple[int, float]]:
        """Adaptive exploration based on user maturity (interaction count)."""
        if self.model is None:
            return []
        uidx = user_id_to_index.get(int(user_id))
        if uidx is None:
            return []
        user_row = user_item_matrix[uidx].toarray().flatten()
        interaction_count = int(np.sum(user_row > 0))
        if interaction_count < 5:
            strategy = "epsilon_greedy"
            epsilon = 0.30
            return self.recommend(user_id, user_id_to_index, index_to_item_id, user_item_matrix, k, filter_seen, strategy, exploration_epsilon=epsilon)
        if interaction_count < 20:
            strategy = "thompson"
            return self.recommend(user_id, user_id_to_index, index_to_item_id, user_item_matrix, k, filter_seen, strategy, exploration_noise_scale=0.1)
        strategy = "diversity"
        return self.recommend(user_id, user_id_to_index, index_to_item_id, user_item_matrix, k, filter_seen, strategy, diversity_weight=0.3)

    def print_top_k_recommendations(
        self,
        user_id: int,
        user_id_to_index: Dict[int, int],
        index_to_item_id: Dict[int, int],
        user_item_matrix: csr_matrix,
        k: int = 10,
        exploration_strategy: str = "none",
    ) -> None:
        recs = self.recommend(
            user_id=user_id,
            user_id_to_index=user_id_to_index,
            index_to_item_id=index_to_item_id,
            user_item_matrix=user_item_matrix,
            k=k,
            exploration_strategy=exploration_strategy,
        )
        if not recs:
            print(f"No recommendations for user {user_id}.")
            return
        print(f"Top-{k} recommendations for user {user_id} (strategy: {exploration_strategy}):")
        for rank, (job_id, score) in enumerate(recs, start=1):
            print(f"{rank:2d}. job_id={job_id}  score={score:.6f}")

    # ---------- Private helpers ----------

    def _top_k_exploit(
        self,
        user_idx: int,
        user_item_matrix: csr_matrix,
        index_to_item_id: Dict[int, int],
        k: int,
        filter_seen: bool,
    ) -> List[Tuple[int, float]]:
        user_items = user_item_matrix.tocsr()
        recs = self.model.recommend(  # type: ignore[union-attr]
            userid=user_idx,
            user_items=user_items,
            N=k,
            filter_already_liked_items=filter_seen,
            recalculate_user=True,
        )
        out: List[Tuple[int, float]] = []
        for item_index, score in recs:
            job_id = index_to_item_id.get(int(item_index))
            if job_id is not None:
                out.append((job_id, float(score)))
        return out

    def _top_k_epsilon_greedy(
        self,
        user_idx: int,
        user_item_matrix: csr_matrix,
        index_to_item_id: Dict[int, int],
        k: int,
        filter_seen: bool,
        epsilon: float,
        rng: np.random.Generator,
    ) -> List[Tuple[int, float]]:
        explore_count = int(k * epsilon)
        exploit_count = k - explore_count
        exploit_recs = self._top_k_exploit(user_idx, user_item_matrix, index_to_item_id, exploit_count, filter_seen)
        if explore_count <= 0:
            return exploit_recs
        seen_items = set()
        if filter_seen:
            user_row = user_item_matrix[user_idx].toarray().flatten()
            seen_indices = np.where(user_row > 0)[0]
            seen_items = {index_to_item_id.get(int(idx)) for idx in seen_indices}
            seen_items.discard(None)
        exploited_items = {job_id for job_id, _ in exploit_recs}
        all_items = set(index_to_item_id.values())
        available = all_items - seen_items - exploited_items
        if not available:
            return exploit_recs
        explore_count_actual = min(explore_count, len(available))
        explored_items = rng.choice(list(available), size=explore_count_actual, replace=False)
        explore_recs: List[Tuple[int, float]] = []
        for job_id in explored_items:
            item_idx = next((idx for idx, jid in index_to_item_id.items() if jid == job_id), None)
            if item_idx is None:
                continue
            score = float(np.dot(self.model.user_factors[user_idx], self.model.item_factors[item_idx]))  # type: ignore[union-attr]
            explore_recs.append((job_id, score))
        all_recs = exploit_recs + explore_recs
        rng.shuffle(all_recs)
        return all_recs[:k]

    def _top_k_thompson(
        self,
        user_idx: int,
        user_item_matrix: csr_matrix,
        index_to_item_id: Dict[int, int],
        k: int,
        filter_seen: bool,
        noise_scale: float,
        rng: np.random.Generator,
    ) -> List[Tuple[int, float]]:
        user_factor = self.model.user_factors[user_idx].copy()  # type: ignore[union-attr]
        noise = rng.normal(0, noise_scale, size=user_factor.shape)
        user_factor_noisy = user_factor + noise
        scores = self.model.item_factors @ user_factor_noisy  # type: ignore[union-attr]
        if filter_seen:
            user_row = user_item_matrix[user_idx].toarray().flatten()
            seen_mask = user_row > 0
            scores[seen_mask] = -np.inf
        if k >= len(scores):
            top_indices = np.argsort(-scores)
        else:
            top_indices = np.argpartition(-scores, k)[:k]
            top_indices = top_indices[np.argsort(-scores[top_indices])]
        out: List[Tuple[int, float]] = []
        for idx in top_indices:
            job_id = index_to_item_id.get(int(idx))
            if job_id is None:
                continue
            original_score = float(np.dot(user_factor, self.model.item_factors[int(idx)]))  # type: ignore[union-attr]
            out.append((job_id, original_score))
        return out[:k]

    def _top_k_with_diversity(
        self,
        user_idx: int,
        user_item_matrix: csr_matrix,
        index_to_item_id: Dict[int, int],
        k: int,
        filter_seen: bool,
        diversity_weight: float,
    ) -> List[Tuple[int, float]]:
        candidates = self._top_k_exploit(user_idx, user_item_matrix, index_to_item_id, k * 3, filter_seen)
        if len(candidates) <= k:
            return candidates
        item_id_to_index = {jid: idx for idx, jid in index_to_item_id.items()}
        selected: List[Tuple[int, float]] = []
        remaining = list(candidates)
        while len(selected) < k and remaining:
            best_score = -np.inf
            best_idx = 0
            for idx, (job_id, relevance) in enumerate(remaining):
                item_idx = item_id_to_index.get(job_id)
                if item_idx is None:
                    continue
                if selected:
                    selected_indices = [item_id_to_index[sel_job_id] for sel_job_id, _ in selected if sel_job_id in item_id_to_index]
                    if selected_indices:
                        item_vec = self.model.item_factors[item_idx]  # type: ignore[union-attr]
                        selected_vecs = self.model.item_factors[selected_indices]  # type: ignore[union-attr]
                        similarities = selected_vecs @ item_vec
                        avg_similarity = float(similarities.mean())
                        diversity = 1.0 - avg_similarity
                    else:
                        diversity = 1.0
                else:
                    diversity = 1.0
                mmr_score = (1 - diversity_weight) * relevance + diversity_weight * diversity
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            selected.append(remaining.pop(best_idx))
        return selected[:k]