"""Embeddings model"""
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class Embeddings:
    """Embeddings result from BGEM3"""
    dense: List[List[float]]  # List of dense vectors
    sparse: List[Any]  # List of sparse vectors (can be dict or scipy sparse)

    @classmethod
    def from_dict(cls, data: Dict[str, List]) -> "Embeddings":
        """Create Embeddings from dict returned by BGEM3"""
        return cls(
            dense=data.get("dense", []),
            sparse=data.get("sparse", []),
        )

    def get_dense_vector(self, index: int = 0) -> List[float]:
        """Get dense vector at index"""
        if self.dense and index < len(self.dense):
            return self.dense[index]
        return []

    def get_sparse_vector(self, index: int = 0) -> Any:
        """Get sparse vector at index"""
        # Handle case where sparse is a list of vectors
        if isinstance(self.sparse, list):
            if len(self.sparse) > 0 and index < len(self.sparse):
                return self.sparse[index]
            return {}
        # Handle case where sparse might be a single scipy sparse matrix or other format
        # (shouldn't normally happen, but handle it)
        if index == 0 and self.sparse is not None:
            return self.sparse
        return {}

