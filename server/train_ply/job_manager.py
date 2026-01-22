"""Job management for tracking training jobs."""

import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from .models import JobInfo, JobStatus


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
        
        # In-memory job storage: job_id -> JobInfo
        self._jobs: Dict[str, JobInfo] = {}
    
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
        
        self._jobs[job_id] = job_info
        
        # Create job directory
        job_dir = self.jobs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "input").mkdir(exist_ok=True)
        (job_dir / "output").mkdir(exist_ok=True)
        (job_dir / "logs").mkdir(exist_ok=True)
        
        return job_id
    
    def get_job(self, job_id: str) -> Optional[JobInfo]:
        """Get job information.
        
        Args:
            job_id: Job identifier
            
        Returns:
            JobInfo if found, None otherwise
        """
        return self._jobs.get(job_id)
    
    def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        current_step: Optional[int] = None,
        max_steps: Optional[int] = None,
        error_message: Optional[str] = None,
        error_traceback: Optional[str] = None,
    ) -> bool:
        """Update job status and progress.
        
        Args:
            job_id: Job identifier
            status: New status
            current_step: Current training step (optional)
            max_steps: Total training steps (optional)
            error_message: Error message if failed (optional)
            error_traceback: Error traceback if failed (optional)
            
        Returns:
            True if job was updated, False if job not found
        """
        job = self._jobs.get(job_id)
        if job is None:
            return False
        
        job.status = status
        job.updated_at = datetime.now()
        
        if current_step is not None:
            job.current_step = current_step
        if max_steps is not None:
            job.max_steps = max_steps
        if error_message is not None:
            job.error_message = error_message
        if error_traceback is not None:
            job.error_traceback = error_traceback
        
        # Calculate progress
        if job.current_step is not None and job.max_steps is not None:
            job.progress = min(job.current_step / job.max_steps, 1.0)
        elif status == JobStatus.COMPLETED:
            job.progress = 1.0
        elif status == JobStatus.FAILED or status == JobStatus.CANCELLED:
            job.progress = job.progress or 0.0
        
        return True
    
    def update_job_results(
        self,
        job_id: str,
        result_dir: Optional[str] = None,
        ply_files: Optional[list] = None,
        checkpoint_files: Optional[list] = None,
    ) -> bool:
        """Update job result paths.
        
        Args:
            job_id: Job identifier
            result_dir: Directory containing results
            ply_files: List of PLY file paths
            checkpoint_files: List of checkpoint file paths
            
        Returns:
            True if job was updated, False if job not found
        """
        job = self._jobs.get(job_id)
        if job is None:
            return False
        
        if result_dir is not None:
            job.result_dir = result_dir
        if ply_files is not None:
            job.ply_files = ply_files
        if checkpoint_files is not None:
            job.checkpoint_files = checkpoint_files
        
        job.updated_at = datetime.now()
        return True
    
    def list_jobs(self, status: Optional[JobStatus] = None) -> list[JobInfo]:
        """List all jobs, optionally filtered by status.
        
        Args:
            status: Optional status filter
            
        Returns:
            List of JobInfo objects
        """
        jobs = list(self._jobs.values())
        if status is not None:
            jobs = [j for j in jobs if j.status == status]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)
    
    def get_job_dir(self, job_id: str) -> Optional[Path]:
        """Get the directory path for a job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Path to job directory, or None if job doesn't exist
        """
        if job_id not in self._jobs:
            return None
        return self.jobs_dir / job_id
    
    def get_active_jobs_count(self) -> int:
        """Get the number of active (pending or running) jobs.
        
        Returns:
            Number of active jobs
        """
        return len([
            j for j in self._jobs.values()
            if j.status in [JobStatus.PENDING, JobStatus.RUNNING]
        ])
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job (if it's pending or running).
        
        Args:
            job_id: Job identifier
            
        Returns:
            True if job was cancelled, False otherwise
        """
        job = self._jobs.get(job_id)
        if job is None:
            return False
        
        if job.status in [JobStatus.PENDING, JobStatus.RUNNING]:
            return self.update_job_status(job_id, JobStatus.CANCELLED)
        
        return False
