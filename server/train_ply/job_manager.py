"""Job management for tracking training jobs."""

import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .models import JobInfo, JobStatus, TrainingConfig


class Job:
    """Represents a single training job with its state and operations."""
    
    def __init__(self, job_info: JobInfo, job_dir: Path):
        """Initialize a job.
        
        Args:
            job_info: JobInfo Pydantic model containing job state
            job_dir: Path to the job's directory
        """
        self._info = job_info
        self._job_dir = job_dir
    
    def to_info(self) -> JobInfo:
        """Convert Job to JobInfo for API responses.
        
        Returns:
            JobInfo instance (same reference, not a copy)
        """
        return self._info
    
    def update_status(
        self,
        status: JobStatus,
        current_step: Optional[int] = None,
        max_steps: Optional[int] = None,
        error_message: Optional[str] = None,
        error_traceback: Optional[str] = None,
    ) -> None:
        """Update job status and progress.
        
        Args:
            status: New status
            current_step: Current training step (optional)
            max_steps: Total training steps (optional)
            error_message: Error message if failed (optional)
            error_traceback: Error traceback if failed (optional)
        """
        self._info.status = status
        self._info.updated_at = datetime.now()
        
        if current_step is not None:
            self._info.current_step = current_step
        if max_steps is not None:
            self._info.max_steps = max_steps
        if error_message is not None:
            self._info.error_message = error_message
        if error_traceback is not None:
            self._info.error_traceback = error_traceback
        
        # Calculate progress
        if self._info.current_step is not None and self._info.max_steps is not None:
            self._info.progress = min(self._info.current_step / self._info.max_steps, 1.0)
        elif status == JobStatus.COMPLETED:
            self._info.progress = 1.0
        elif status == JobStatus.FAILED or status == JobStatus.CANCELLED:
            self._info.progress = self._info.progress or 0.0
    
    def update_results(
        self,
        result_dir: Optional[str] = None,
        ply_files: Optional[list] = None,
        checkpoint_files: Optional[list] = None,
    ) -> None:
        """Update job result paths.
        
        Args:
            result_dir: Directory containing results
            ply_files: List of PLY file paths
            checkpoint_files: List of checkpoint file paths
        """
        if result_dir is not None:
            self._info.result_dir = result_dir
        if ply_files is not None:
            self._info.ply_files = ply_files
        if checkpoint_files is not None:
            self._info.checkpoint_files = checkpoint_files
        
        self._info.updated_at = datetime.now()
    
    def upload_ply(self, file_path: Path) -> bool:
        """Upload PLY file to job.
        
        Args:
            file_path: Path to PLY file to save
            
        Returns:
            True if successful, False if job directory doesn't exist
        """
        if not self._job_dir.exists():
            return False
        
        # Copy file to job directory
        target_path = self._job_dir / "input" / "model.ply"
        shutil.copy2(file_path, target_path)
        
        self._info.ply_uploaded = True
        self._info.updated_at = datetime.now()
        
        return True
    
    def upload_image(self, filename: str, file_path: Path) -> bool:
        """Upload image file to job.
        
        Args:
            filename: Name of the image file
            file_path: Path to image file to save
            
        Returns:
            True if successful, False if duplicate or job directory doesn't exist
        """
        if not self._job_dir.exists():
            return False
        
        # Check for duplicates
        if filename in self._info.images_uploaded:
            return False  # Duplicate upload
        
        # Copy file to job directory
        target_path = self._job_dir / "input" / "images" / filename
        shutil.copy2(file_path, target_path)
        
        self._info.images_uploaded.append(filename)
        self._info.updated_at = datetime.now()
        
        return True
    
    def upload_cameras(self, file_path: Path) -> bool:
        """Upload camera.json file to job.
        
        Args:
            file_path: Path to camera.json file to save
            
        Returns:
            True if successful, False if invalid JSON or job directory doesn't exist
        """
        if not self._job_dir.exists():
            return False
        
        # Validate JSON format
        try:
            with open(file_path, "r") as f:
                json.load(f)
        except (json.JSONDecodeError, IOError):
            return False
        
        # Copy file to job directory
        target_path = self._job_dir / "input" / "cameras.json"
        shutil.copy2(file_path, target_path)
        
        self._info.cameras_uploaded = True
        self._info.updated_at = datetime.now()
        
        return True
    
    def upload_config(self, config: TrainingConfig) -> None:
        """Upload training config to job.
        
        Args:
            config: TrainingConfig object
        """
        self._info.config = config
        self._info.config_uploaded = True
        self._info.updated_at = datetime.now()
    
    def validate_ready(self) -> tuple[bool, List[str]]:
        """Validate that job has all required files and is ready to start.
        
        Returns:
            Tuple of (is_ready: bool, errors: List[str])
        """
        errors = []
        
        # Check PLY file
        if not self._info.ply_uploaded:
            errors.append("PLY file not uploaded")
        else:
            ply_path = self._job_dir / "input" / "model.ply"
            if not ply_path.exists():
                errors.append("PLY file missing from disk")
        
        # Check camera.json
        if not self._info.cameras_uploaded:
            errors.append("Camera JSON not uploaded")
        else:
            cameras_path = self._job_dir / "input" / "cameras.json"
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
                        uploaded_image_names = set(self._info.images_uploaded)
                        
                        # Check if all camera keys have corresponding images
                        missing_images = camera_image_names - uploaded_image_names
                        if missing_images:
                            errors.append(f"Missing images referenced in camera JSON: {missing_images}")
                        
                        # Check if we have at least one image
                        if not self._info.images_uploaded:
                            errors.append("No images uploaded")
                        else:
                            # Verify all uploaded images exist on disk
                            images_dir = self._job_dir / "input" / "images"
                            missing_files = []
                            for img_name in self._info.images_uploaded:
                                img_path = images_dir / img_name
                                if not img_path.exists():
                                    missing_files.append(img_name)
                            if missing_files:
                                errors.append(f"Image files missing from disk: {missing_files}")
                except (json.JSONDecodeError, IOError) as e:
                    errors.append(f"Invalid camera JSON: {str(e)}")
        
        # Update validation errors in job
        self._info.validation_errors = errors
        
        is_ready = len(errors) == 0
        if is_ready:
            # Set status to READY if validation passes (only if currently UPLOADING or READY)
            if self._info.status in [JobStatus.UPLOADING, JobStatus.READY]:
                self._info.status = JobStatus.READY
        else:
            # Reset status to UPLOADING if validation fails (allows re-uploading)
            if self._info.status == JobStatus.READY:
                self._info.status = JobStatus.UPLOADING
        
        return is_ready, errors
    
    def cancel(self) -> bool:
        """Cancel a job (if it's pending or running).
        
        Returns:
            True if job was cancelled, False otherwise
        """
        if self._info.status in [JobStatus.PENDING, JobStatus.RUNNING]:
            self.update_status(JobStatus.CANCELLED)
            return True
        return False
    
    def start_training(self) -> bool:
        """Start training from an uploaded job (change status from READY to PENDING).
        
        Returns:
            True if successful, False if validation fails or status is not READY
        """
        # Validate job is ready
        is_ready, _ = self.validate_ready()
        if not is_ready:
            return False
        
        # Change status from READY to PENDING
        if self._info.status == JobStatus.READY:
            self._info.status = JobStatus.PENDING
            self._info.updated_at = datetime.now()
            return True
        
        return False
    
    def get_dir(self) -> Path:
        """Get the job directory path.
        
        Returns:
            Path to job directory
        """
        return self._job_dir


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