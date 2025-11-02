"""Data processing utilities"""
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class DataProcessor:
    """Data processing utilities"""

    @staticmethod
    def clean_text(text: str) -> str:
        """Basic text cleaning"""
        if not text:
            return ""
        return " ".join(text.strip().split())

    @staticmethod
    def extract_skill_names(skills: Any) -> List[str]:
        """
        Extract skill names from skills array.
        Handles both formats:
        - Array of strings: ["Python", "Java"]
        - Array of objects: [{"id": 1, "name": "Python"}, {"id": 2, "name": "Java"}]
        """
        if not skills or not isinstance(skills, list):
            return []
        
        # Check if first item is a dict (object) or string
        if isinstance(skills[0], dict):
            # Array of objects: extract "name" field
            skill_names = []
            for skill in skills:
                if isinstance(skill, dict):
                    name = skill.get("name")  # Get "name" field from skill object
                    if name:
                        skill_names.append(str(name))
            return skill_names
        else:
            # Array of strings: return as is
            return [str(skill) for skill in skills if skill]

    @staticmethod
    def normalize_job_fields(job: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize job fields from camelCase (API format) to snake_case (internal format)
        Also handles field name variations
        """
        normalized = {}
        
        # Map camelCase to snake_case field names
        field_mapping = {
            "jobRole": "job_role",
            "job_role": "job_role",  # Already correct
            "workMode": "work_mode",
            "work_mode": "work_mode",  # Already correct
            "datePosted": "date_posted",
            "date_posted": "date_posted",  # Already correct
            "dateExpires": "date_expires",
            "date_expires": "date_expires",  # Already correct
            "minExperienceYears": "min_experience_years",
            "min_experience_years": "min_experience_years",  # Already correct
            "salaryMin": "salary_min",
            "salary_min": "salary_min",  # Already correct
            "salaryMax": "salary_max",
            "salary_max": "salary_max",  # Already correct
        }
        
        # Copy all fields first
        for key, value in job.items():
            # Map field names
            if key in field_mapping:
                normalized[field_mapping[key]] = value
            else:
                normalized[key] = value
        
        # Extract and normalize skills from original job data
        # Handle both "skills" key variations
        original_skills = job.get("skills", [])
        if original_skills:
            normalized["skills"] = DataProcessor.extract_skill_names(original_skills)
        else:
            normalized["skills"] = []
        
        return normalized

    @staticmethod
    def combine_job_text(job: Dict[str, Any]) -> str:
        """Combine relevant job fields into a single text for embedding"""
        # Normalize job fields first
        normalized_job = DataProcessor.normalize_job_fields(job)
        
        # Extract skill names
        skill_names = DataProcessor.extract_skill_names(normalized_job.get("skills", []))
        
        parts = [
            normalized_job.get("title", ""),
            " ".join(skill_names),
            normalized_job.get("company", ""),
            normalized_job.get("job_role", ""),
            normalized_job.get("seniority", ""),
            normalized_job.get("location", ""),
            normalized_job.get("work_mode", ""),
        ]
        combined = " | ".join([DataProcessor.clean_text(p) for p in parts if p])
        return combined

    @staticmethod
    def build_entities(
        jobs: List[Dict[str, Any]], embeddings: Dict = None, dense_dim: int = 0
    ) -> List[Dict[str, Any]]:
        """Build entities for Milvus insertion"""
        ids, titles, skills, companies, job_roles, seniorities = [], [], [], [], [], []
        min_experience_years, work_modes, salary_min, salary_max = [], [], [], []
        currencies, statuses, max_candidates = [], [], []
        date_posted, date_expires, locations = [], [], []
        for job in jobs:
            (
                title,
                skill_list,
                company,
                job_role,
                seniority,
                min_exp,
                work_mode,
                sal_min,
                sal_max,
                currency,
                status,
                max_cand,
                date_post,
                date_exp,
                location,
            ) = DataProcessor.extract_fields(job)

            ids.append(job["id"])
            titles.append(title)
            skills.append(skill_list)
            companies.append(company)
            job_roles.append(job_role)
            seniorities.append(seniority)
            min_experience_years.append(min_exp)
            work_modes.append(work_mode)
            salary_min.append(sal_min)
            salary_max.append(sal_max)
            currencies.append(currency)
            statuses.append(status)
            max_candidates.append(max_cand)
            date_posted.append(date_post)
            date_expires.append(date_exp)
            locations.append(location)

            if embeddings:
                sparse_vectors = embeddings["sparse"]
                dense_vectors = embeddings["dense"]
            else:
                sparse_vectors = [{} for _ in jobs]  # empty sparse vectors
                dense_vectors = [
                    [0.0] * dense_dim for _ in jobs
                ]  # placeholder

        return [
            ids,                    # 1
            titles,                 # 2
            skills,                 # 3
            companies,              # 4
            job_roles,              # 5
            seniorities,            # 6
            min_experience_years,   # 7
            work_modes,             # 8
            salary_min,             # 9
            salary_max,             # 10
            currencies,             # 11
            statuses,               # 12
            max_candidates,         # 13
            date_posted,            # 14
            date_expires,           # 15
            locations,              # 16
            sparse_vectors,        # 17
            dense_vectors,          # 18
        ]

    @staticmethod
    def extract_fields(job: Dict) -> tuple:
        """Extract specified fields from job dict (handles both camelCase and snake_case)"""
        # Normalize job fields first to handle camelCase
        normalized_job = DataProcessor.normalize_job_fields(job)
        
        title = normalized_job.get("title", "")
        
        # Extract skill names (array of strings)
        skill_list = DataProcessor.extract_skill_names(normalized_job.get("skills", []))
        
        company = normalized_job.get("company", "")
        job_role = normalized_job.get("job_role", "")
        seniority = normalized_job.get("seniority", "")
        
        # Handle min_experience_years (can be minExperienceYears or min_experience_years)
        min_exp = normalized_job.get("min_experience_years") or normalized_job.get("minExperienceYears", 0)
        min_experience_years = int(min_exp) if min_exp else 0
        
        work_mode = normalized_job.get("work_mode", "")
        
        # Handle salary fields
        sal_min = normalized_job.get("salary_min") or normalized_job.get("salaryMin", 0)
        salary_min = int(sal_min) if sal_min else 0
        
        sal_max = normalized_job.get("salary_max") or normalized_job.get("salaryMax", 0)
        salary_max = int(sal_max) if sal_max else 0
        
        currency = normalized_job.get("currency", "")
        status = normalized_job.get("status", "")
        
        # Handle max_candidates
        max_cand = normalized_job.get("max_candidates") or normalized_job.get("maxCandidates", 0)
        max_candidates = int(max_cand) if max_cand else 0
        
        # Handle date fields - convert ISO string to timestamp if needed
        # Get from normalized_job (already mapped) or original job as fallback
        date_posted_raw = normalized_job.get("date_posted") or normalized_job.get("datePosted") or job.get("datePosted")
        
        # Debug: log if date_posted_raw is None or empty
        if not date_posted_raw:
            logger.warning(f"date_posted is missing in job data. Available keys: {list(job.keys())}")
        
        if date_posted_raw and isinstance(date_posted_raw, str):
            # Try to parse ISO date string to timestamp
            try:
                from datetime import datetime
                # Handle different ISO formats
                date_str = date_posted_raw.strip()
                
                # Replace Z with +00:00
                if date_str.endswith('Z'):
                    date_str = date_str[:-1] + '+00:00'
                
                # Fix microseconds if present (datetime.fromisoformat supports max 6 digits)
                # Format: 2025-11-02T18:51:50.1635356+07:00 -> 2025-11-02T18:51:50.163535+07:00
                if '.' in date_str and ('+' in date_str or '-' in date_str[-6:]):
                    # Find the dot position
                    dot_idx = date_str.index('.')
                    # Find timezone separator (+ or -)
                    tz_idx = len(date_str)
                    for i in range(len(date_str) - 1, dot_idx, -1):
                        if date_str[i] in '+-' and i > dot_idx + 1:
                            tz_idx = i
                            break
                    
                    if tz_idx < len(date_str):
                        # Extract parts
                        before_dot = date_str[:dot_idx]
                        after_dot = date_str[dot_idx+1:tz_idx]
                        timezone = date_str[tz_idx:]
                        
                        # Truncate microseconds to 6 digits if longer
                        if len(after_dot) > 6:
                            after_dot = after_dot[:6]
                        
                        date_str = f"{before_dot}.{after_dot}{timezone}"
                
                dt = datetime.fromisoformat(date_str)
                date_posted = int(dt.timestamp() * 1000)  # Convert to milliseconds
                logger.debug(f"Parsed date_posted: '{date_posted_raw}' -> timestamp: {date_posted}")
            except Exception as e:
                logger.warning(f"Failed to parse date_posted '{date_posted_raw}': {e}")
                date_posted = 0
        elif date_posted_raw:
            # Already a number (timestamp)
            date_posted = int(date_posted_raw) if date_posted_raw else 0
        else:
            date_posted = 0
        
        date_expires_raw = normalized_job.get("date_expires") or normalized_job.get("dateExpires") or job.get("dateExpires")
        
        if date_expires_raw and isinstance(date_expires_raw, str):
            try:
                from datetime import datetime
                date_str = date_expires_raw.strip()
                
                # Replace Z with +00:00
                if date_str.endswith('Z'):
                    date_str = date_str[:-1] + '+00:00'
                
                # Fix microseconds if present
                if '.' in date_str and ('+' in date_str or '-' in date_str[-6:]):
                    dot_idx = date_str.index('.')
                    tz_idx = len(date_str)
                    for i in range(len(date_str) - 1, dot_idx, -1):
                        if date_str[i] in '+-' and i > dot_idx + 1:
                            tz_idx = i
                            break
                    
                    if tz_idx < len(date_str):
                        before_dot = date_str[:dot_idx]
                        after_dot = date_str[dot_idx+1:tz_idx]
                        timezone = date_str[tz_idx:]
                        
                        if len(after_dot) > 6:
                            after_dot = after_dot[:6]
                        
                        date_str = f"{before_dot}.{after_dot}{timezone}"
                
                dt = datetime.fromisoformat(date_str)
                date_expires = int(dt.timestamp() * 1000)
                logger.debug(f"Parsed date_expires: '{date_expires_raw}' -> timestamp: {date_expires}")
            except Exception as e:
                logger.warning(f"Failed to parse date_expires '{date_expires_raw}': {e}")
                date_expires = 0
        elif date_expires_raw:
            date_expires = int(date_expires_raw) if date_expires_raw else 0
        else:
            date_expires = 0
        
        location = normalized_job.get("location", "") or ""
        # Keep location as-is (schema max_length is now 100)
        if location:
            location = str(location).strip()
        else:
            location = ""
        
        return (
            title,
            skill_list,  # Array of skill name strings
            company,
            job_role,
            seniority,
            min_experience_years,
            work_mode,
            salary_min,
            salary_max,
            currency,
            status,
            max_candidates,  # max_candidates field
            date_posted,
            date_expires,
            location,
        )

