"""Job management for tracking training jobs."""

import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

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
        
        self._jobs[job_id] = job_info
        
        # Create job directory
        job_dir = self.jobs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "input").mkdir(exist_ok=True)
        (job_dir / "input" / "images").mkdir(exist_ok=True)
        (job_dir / "output").mkdir(exist_ok=True)
        (job_dir / "logs").mkdir(exist_ok=True)
        
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
        
        job_dir = self.get_job_dir(job_id)
        if job_dir is None:
            return False
        
        # Copy file to job directory
        target_path = job_dir / "input" / "model.ply"
        shutil.copy2(file_path, target_path)
        
        job.ply_uploaded = True
        job.updated_at = datetime.now()
        
        return True
    
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
        
        job_dir = self.get_job_dir(job_id)
        if job_dir is None:
            return False
        
        # Check for duplicates
        if filename in job.images_uploaded:
            return False  # Duplicate upload
        
        # Copy file to job directory
        target_path = job_dir / "input" / "images" / filename
        shutil.copy2(file_path, target_path)
        
        job.images_uploaded.append(filename)
        job.updated_at = datetime.now()
        
        return True
    
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
        
        job_dir = self.get_job_dir(job_id)
        if job_dir is None:
            return False
        
        # Validate JSON format
        try:
            with open(file_path, "r") as f:
                json.load(f)
        except (json.JSONDecodeError, IOError):
            return False
        
        # Copy file to job directory
        target_path = job_dir / "input" / "cameras.json"
        shutil.copy2(file_path, target_path)
        
        job.cameras_uploaded = True
        job.updated_at = datetime.now()
        
        return True
    
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
        
        job.config = config
        job.config_uploaded = True
        job.updated_at = datetime.now()
        
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
        
        errors = []
        
        # Check PLY file
        if not job.ply_uploaded:
            errors.append("PLY file not uploaded")
        else:
            job_dir = self.get_job_dir(job_id)
            if job_dir:
                ply_path = job_dir / "input" / "model.ply"
                if not ply_path.exists():
                    errors.append("PLY file missing from disk")
        
        # Check camera.json
        if not job.cameras_uploaded:
            errors.append("Camera JSON not uploaded")
        else:
            job_dir = self.get_job_dir(job_id)
            if job_dir:
                cameras_path = job_dir / "input" / "cameras.json"
                if not cameras_path.exists():
                    errors.append("Camera JSON missing from disk")
                else:
                    # Validate camera JSON and check image matching
                    try:
                        with open(cameras_path, "r") as f:
                            camera_data = json.load(f)
                        
                        if not isinstance(camera_data, dict):
                            errors.append("Camera JSON is not a valid object")
                        else:
                            camera_image_names = set(camera_data.keys())
                            uploaded_image_names = set(job.images_uploaded)
                            
                            # Check if all camera keys have corresponding images
                            missing_images = camera_image_names - uploaded_image_names
                            if missing_images:
                                errors.append(f"Missing images referenced in camera JSON: {missing_images}")
                            
                            # Check if we have at least one image
                            if not job.images_uploaded:
                                errors.append("No images uploaded")
                            else:
                                # Verify all uploaded images exist on disk
                                images_dir = job_dir / "input" / "images"
                                missing_files = []
                                for img_name in job.images_uploaded:
                                    img_path = images_dir / img_name
                                    if not img_path.exists():
                                        missing_files.append(img_name)
                                if missing_files:
                                    errors.append(f"Image files missing from disk: {missing_files}")
                    except (json.JSONDecodeError, IOError) as e:
                        errors.append(f"Invalid camera JSON: {str(e)}")
        
        # Update validation errors in job
        job.validation_errors = errors
        
        is_ready = len(errors) == 0
        if is_ready:
            # Set status to READY if validation passes (only if currently UPLOADING or READY)
            if job.status in [JobStatus.UPLOADING, JobStatus.READY]:
                job.status = JobStatus.READY
        else:
            # Reset status to UPLOADING if validation fails (allows re-uploading)
            if job.status == JobStatus.READY:
                job.status = JobStatus.UPLOADING
        
        return is_ready, errors
    
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
        is_ready = job.status == JobStatus.READY
        
        return {
            "job_id": job_id,
            "status": job.status,
            "ply_uploaded": job.ply_uploaded,
            "cameras_uploaded": job.cameras_uploaded,
            "images_uploaded": job.images_uploaded.copy(),
            "config_uploaded": job.config_uploaded,
            "validation_errors": job.validation_errors.copy(),
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
        
        # Validate job is ready
        is_ready, errors = self.validate_job_ready(job_id)
        if not is_ready:
            return False
        
        # Change status from READY to PENDING
        if job.status == JobStatus.READY:
            job.status = JobStatus.PENDING
            job.updated_at = datetime.now()
            return True
        
        return False
    
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
        if job.status not in [JobStatus.UPLOADING, JobStatus.READY]:
            return False
        
        # Delete job directory
        job_dir = self.get_job_dir(job_id)
        if job_dir and job_dir.exists():
            shutil.rmtree(job_dir)
        
        # Remove from jobs dict
        del self._jobs[job_id]
        
        return True
