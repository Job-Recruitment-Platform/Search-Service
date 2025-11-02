"""Search result models"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SearchWeights:
    """Search weights for hybrid search"""
    dense: float = 1.0
    sparse: float = 1.0

    @classmethod
    def from_dict(cls, data: dict) -> "SearchWeights":
        """Create from dict"""
        return cls(
            dense=float(data.get("dense", 1.0)),
            sparse=float(data.get("sparse", 1.0)),
        )

    def to_dict(self) -> dict:
        """Convert to dict"""
        return {"dense": self.dense, "sparse": self.sparse}


@dataclass
class SearchResult:
    """Search result item"""
    id: int
    score: float
    title: str
    company: str
    job_role: str
    seniority: str
    min_experience_years: int
    work_mode: str
    salary_min: int
    salary_max: int
    currency: str
    status: str
    date_posted: int
    date_expires: int
    skills: List[str] = field(default_factory=list)
    location: str = ""

    @classmethod
    def from_milvus_hit(cls, hit) -> "SearchResult":
        """Create SearchResult from Milvus search hit"""
        skills = hit.get("skills", [])
        if skills:
            skills = list(skills) if isinstance(skills, (list, tuple)) else []

        return cls(
            id=int(hit["id"]),
            score=float(hit.score),
            title=str(hit.get("title", "")),
            company=str(hit.get("company", "")),
            job_role=str(hit.get("job_role", "")),
            seniority=str(hit.get("seniority", "")),
            min_experience_years=int(hit.get("min_experience_years", 0)),
            work_mode=str(hit.get("work_mode", "")),
            salary_min=int(hit.get("salary_min", 0)),
            salary_max=int(hit.get("salary_max", 0)),
            currency=str(hit.get("currency", "")),
            status=str(hit.get("status", "")),
            date_posted=int(hit.get("date_posted", 0)),
            date_expires=int(hit.get("date_expires", 0)),
            skills=skills,
            location=str(hit.get("location", "")),
        )

    def to_dict(self) -> dict:
        """Convert to dict"""
        return {
            "id": self.id,
            "score": self.score,
            "title": self.title,
            "company": self.company,
            "job_role": self.job_role,
            "seniority": self.seniority,
            "min_experience_years": self.min_experience_years,
            "work_mode": self.work_mode,
            "salary_min": self.salary_min,
            "salary_max": self.salary_max,
            "currency": self.currency,
            "status": self.status,
            "date_posted": self.date_posted,
            "date_expires": self.date_expires,
            "skills": self.skills,
            "location": self.location,
        }

