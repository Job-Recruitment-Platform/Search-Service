"""Search result models"""
import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


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
        # Extract skills - handle various formats from Milvus
        # Milvus ARRAY field returns as list/tuple, but might be empty or None
        skills_raw = hit.get("skills")
        skills = []
        
        # Debug logging
        logger.debug(f"Raw skills from Milvus: type={type(skills_raw)}, value={skills_raw}, job_id={hit.get('id')}")
        
        # Handle None or empty cases
        if not skills_raw:
            logger.debug(f"Skills is None or empty for job_id={hit.get('id')}")
        elif isinstance(skills_raw, (list, tuple)):
            # Milvus ARRAY field should return as list/tuple
            if len(skills_raw) > 0:
                # Convert to list of strings
                for skill in skills_raw:
                    if skill is not None:
                        # Handle both string and object formats
                        if isinstance(skill, str):
                            skill_str = skill.strip()
                            if skill_str:
                                skills.append(skill_str)
                        elif isinstance(skill, dict) and "name" in skill:
                            name = skill.get("name")
                            if name:
                                skills.append(str(name).strip())
                        else:
                            # Try to convert to string
                            skill_str = str(skill).strip()
                            if skill_str and skill_str != "None":
                                skills.append(skill_str)
                logger.debug(f"Extracted {len(skills)} skills: {skills}")
            else:
                logger.debug(f"Skills array is empty for job_id={hit.get('id')}")
        else:
            # Try to convert other types (shouldn't normally happen)
            logger.warning(f"Unexpected skills type: {type(skills_raw)} for job_id={hit.get('id')}")
            try:
                skills_list = list(skills_raw)
                skills = [str(s).strip() for s in skills_list if s and str(s).strip()]
            except (TypeError, ValueError) as e:
                logger.warning(f"Failed to convert skills: {e}, type={type(skills_raw)}")
                skills = []
        
        # Final check - remove any empty strings
        skills = [s for s in skills if s]
        
        # Log if skills is still empty
        if not skills:
            logger.debug(f"No skills extracted for job_id={hit.get('id')}, raw_skills={skills_raw}, raw_type={type(skills_raw)}")

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

