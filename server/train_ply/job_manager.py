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
    
    def create_job(self, job_id: Optional[str] = None) -> str:
        """Create a new job in UPLOADING state.
        
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
    
    def cleanup_job(self, job_id: str) -> bool:
        """Clean up a job based on its current state.
        
        - UPLOADING/READY: Full cleanup (delete directory and remove from memory)
        - PENDING/RUNNING: Update status to CANCELLED (keep directory/memory for logs/results)
        - COMPLETED/FAILED/CANCELLED: Cannot cleanup (returns False)
        
        Args:
            job_id: Job identifier
            
        Returns:
            True if successful, False if job not found or cannot be cleaned up
        """
        job = self._jobs.get(job_id)
        if job is None:
            return False
        
        status = job._info.status
        
        # Full cleanup for incomplete uploads
        if status in [JobStatus.UPLOADING, JobStatus.READY]:
            # Delete job directory
            if job._job_dir.exists():
                shutil.rmtree(job._job_dir)
            
            # Remove from jobs dict
            del self._jobs[job_id]
            return True
        
        # Cancel active training jobs (keep directory/memory for logs/results)
        elif status in [JobStatus.PENDING, JobStatus.RUNNING]:
            job.cancel()
            return True
        
        # Cannot cleanup completed/failed/cancelled jobs
        else:
            return False