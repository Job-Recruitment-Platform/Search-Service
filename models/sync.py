"""Sync operation result models"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class SyncResult:
    """Result of sync operation"""
    processed: int = 0
    inserted: int = 0
    deleted: int = 0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dict"""
        result = {
            "processed": self.processed,
            "inserted": self.inserted,
            "deleted": self.deleted,
        }
        if self.error:
            result["error"] = self.error
        return result

