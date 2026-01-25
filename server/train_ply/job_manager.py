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
        
        job.update_status(status, current_step, max_steps, error_message, error_traceback)
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
        
        job.update_results(result_dir, ply_files, checkpoint_files)
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
        
        return job.cancel()
    
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
    
    def upload_ply(self, job_id: str, file_path: Path) -> bool:
        """Upload PLY file to job.
        
        Args:
            job_id: Job identifier
            file_path: Path to PLY file to save
            
        Returns:
            True if successful, False if job not found
        """
        job = self._jobs.get(job_id)
        if job is None:
            return False
        
        return job.upload_ply(file_path)
    
    def upload_image(self, job_id: str, filename: str, file_path: Path) -> bool:
        """Upload image file to job.
        
        Args:
            job_id: Job identifier
            filename: Name of the image file
            file_path: Path to image file to save
            
        Returns:
            True if successful, False if job not found
        """
        job = self._jobs.get(job_id)
        if job is None:
            return False
        
        return job.upload_image(filename, file_path)
    
    def upload_cameras(self, job_id: str, file_path: Path) -> bool:
        """Upload camera.json file to job.
        
        Args:
            job_id: Job identifier
            file_path: Path to camera.json file to save
            
        Returns:
            True if successful, False if job not found or invalid JSON
        """
        job = self._jobs.get(job_id)
        if job is None:
            return False
        
        return job.upload_cameras(file_path)
    
    def upload_config(self, job_id: str, config: TrainingConfig) -> bool:
        """Upload training config to job.
        
        Args:
            job_id: Job identifier
            config: TrainingConfig object
            
        Returns:
            True if successful, False if job not found
        """
        job = self._jobs.get(job_id)
        if job is None:
            return False
        
        job.upload_config(config)
        return True
    
    def validate_job_ready(self, job_id: str) -> tuple[bool, List[str]]:
        """Validate that job has all required files and is ready to start.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Tuple of (is_ready: bool, errors: List[str])
        """
        job = self._jobs.get(job_id)
        if job is None:
            return False, [f"Job {job_id} not found"]
        
        return job.validate_ready()
    
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
    
    def start_training_from_upload(self, job_id: str) -> bool:
        """Start training from an uploaded job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            True if successful, False if validation fails
        """
        job = self._jobs.get(job_id)
        if job is None:
            return False
        
        return job.start_training()
    
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