"""Job management for tracking training jobs."""

import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .job import Job
from .models import JobInfo, JobStatus, TrainingConfig


class JobManager:
    """Manages training job state and tracking."""
    
    def __init__(self, jobs_dir: Optional[str] = None):
        """Initialize job manager.
        
        Args:
            jobs_dir: Directory to store job data (default: server/train_ply/jobs/)
        """
        if jobs_dir is None:
            # Default to server/train_ply/jobs/ relative to this file
            script_dir = Path(__file__).parent
            jobs_dir = str(script_dir / "jobs")
        
        self.jobs_dir = Path(jobs_dir)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory job storage: job_id -> Job
        self._jobs: Dict[str, Job] = {}
    
    def create_job(self, job_id: Optional[str] = None) -> str:
        """Create a new job and return its ID.
        
        Args:
            job_id: Optional custom job ID (default: generate UUID)
            
        Returns:
            Job ID string
        """
        if job_id is None:
            job_id = str(uuid.uuid4())
        
        now = datetime.now()
        job_info = JobInfo(
            job_id=job_id,
            status=JobStatus.PENDING,
            created_at=now,
            updated_at=now,
        )
        
        # Create job directory
        job_dir = self.jobs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "input").mkdir(exist_ok=True)
        (job_dir / "output").mkdir(exist_ok=True)
        (job_dir / "logs").mkdir(exist_ok=True)
        
        # Create and store Job instance
        job = Job(job_info, job_dir)
        self._jobs[job_id] = job
        
        return job_id
    
    def get_job(self, job_id: str) -> Optional[JobInfo]:
        """Get job information.
        
        Args:
            job_id: Job identifier
            
        Returns:
            JobInfo if found, None otherwise
        """
        job = self._jobs.get(job_id)
        if job is None:
            return None
        return job.to_info()
    
    def _get_job_object(self, job_id: str) -> Optional[Job]:
        """Get Job object directly (internal helper).
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job object if found, None otherwise
        """
        return self._jobs.get(job_id)
    
    def list_jobs(self, status: Optional[JobStatus] = None) -> list[JobInfo]:
        """List all jobs, optionally filtered by status.
        
        Args:
            status: Optional status filter
            
        Returns:
            List of JobInfo objects
        """
        jobs = list(self._jobs.values())
        if status is not None:
            jobs = [j for j in jobs if j._info.status == status]
        return sorted([j.to_info() for j in jobs], key=lambda j: j.created_at, reverse=True)
    
    def get_job_dir(self, job_id: str) -> Optional[Path]:
        """Get the directory path for a job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Path to job directory, or None if job doesn't exist
        """
        job = self._jobs.get(job_id)
        if job is None:
            return None
        return job.get_dir()
    
    def get_active_jobs_count(self) -> int:
        """Get the number of active (pending or running) jobs.
        
        Returns:
            Number of active jobs
        """
        return len([
            j for j in self._jobs.values()
            if j._info.status in [JobStatus.PENDING, JobStatus.RUNNING]
        ])
    
    def create_upload_job(self, job_id: Optional[str] = None) -> str:
        """Create a new upload job in UPLOADING state.
        
        Args:
            job_id: Optional custom job ID (default: generate UUID)
            
        Returns:
            Job ID string
        """
        if job_id is None:
            job_id = str(uuid.uuid4())
        
        now = datetime.now()
        job_info = JobInfo(
            job_id=job_id,
            status=JobStatus.UPLOADING,
            created_at=now,
            updated_at=now,
        )
        
        # Create job directory
        job_dir = self.jobs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "input").mkdir(exist_ok=True)
        (job_dir / "input" / "images").mkdir(exist_ok=True)
        (job_dir / "output").mkdir(exist_ok=True)
        (job_dir / "logs").mkdir(exist_ok=True)
        
        # Create and store Job instance
        job = Job(job_info, job_dir)
        self._jobs[job_id] = job
        
        return job_id
    
    def get_upload_status(self, job_id: str) -> Optional[dict]:
        """Get upload status for a job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Dict with upload status, or None if job not found
            Note: This does NOT run validation automatically. Use validate_job_ready() separately.
        """
        job = self._jobs.get(job_id)
        if job is None:
            return None
        
        # Return current status without auto-validation
        # is_ready reflects the current job status (READY if previously validated successfully)
        is_ready = job._info.status == JobStatus.READY
        
        return {
            "job_id": job_id,
            "status": job._info.status,
            "ply_uploaded": job._info.ply_uploaded,
            "cameras_uploaded": job._info.cameras_uploaded,
            "images_uploaded": job._info.images_uploaded.copy(),
            "config_uploaded": job._info.config_uploaded,
            "validation_errors": job._info.validation_errors.copy(),
            "is_ready": is_ready,
        }
    
    def cleanup_upload(self, job_id: str) -> bool:
        """Clean up an incomplete upload job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            True if successful, False if job not found or already started
        """
        job = self._jobs.get(job_id)
        if job is None:
            return False
        
        # Only allow cleanup if still in UPLOADING or READY state
        if job._info.status not in [JobStatus.UPLOADING, JobStatus.READY]:
            return False
        
        # Delete job directory
        if job._job_dir.exists():
            shutil.rmtree(job._job_dir)
        
        # Remove from jobs dict
        del self._jobs[job_id]
        
        return True