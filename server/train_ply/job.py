"""Job class for representing a single training job."""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional

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
    
    def upload_camera(self, filename: str, file_path: Path) -> bool:
        """Upload a camera file to job.
        
        Args:
            filename: Name of the camera file
            file_path: Path to camera file to save
            
        Returns:
            True if successful, False if duplicate, invalid JSON, or job directory doesn't exist
        """
        if not self._job_dir.exists():
            return False
        
        # Check for duplicates
        if filename in self._info.cameras_uploaded:
            return False  # Duplicate upload
        
        # Validate JSON format
        try:
            with open(file_path, "r") as f:
                camera_data = json.load(f)
                # Each .cam.json file should contain a single camera entry
                if not isinstance(camera_data, dict) or len(camera_data) == 0:
                    return False
        except (json.JSONDecodeError, IOError):
            return False
        
        # Create cameras directory if it doesn't exist
        cameras_dir = self._job_dir / "input" / "cameras"
        cameras_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy file to job directory
        target_path = cameras_dir / filename
        shutil.copy2(file_path, target_path)
        
        self._info.cameras_uploaded.append(filename)
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
        
        # Check cameras directory
        if not self._info.cameras_uploaded:
            errors.append("No camera files uploaded")
        else:
            cameras_dir = self._job_dir / "input" / "cameras"
            if not cameras_dir.exists() or not cameras_dir.is_dir():
                errors.append("Cameras directory missing from disk")
            else:
                # Load and merge all camera files
                camera_data = {}
                missing_files = []
                invalid_files = []
                
                for camera_filename in self._info.cameras_uploaded:
                    camera_path = cameras_dir / camera_filename
                    if not camera_path.exists():
                        missing_files.append(camera_filename)
                    else:
                        try:
                            with open(camera_path, "r") as f:
                                file_data = json.load(f)
                                if isinstance(file_data, dict):
                                    camera_data.update(file_data)
                                else:
                                    invalid_files.append(f"{camera_filename} (not a valid object)")
                        except (json.JSONDecodeError, IOError) as e:
                            invalid_files.append(f"{camera_filename} ({str(e)})")
                
                if missing_files:
                    errors.append(f"Camera files missing from disk: {missing_files}")
                
                if invalid_files:
                    errors.append(f"Invalid camera files: {invalid_files}")
                
                if camera_data:
                    camera_image_names = set(camera_data.keys())
                    uploaded_image_names = set(self._info.images_uploaded)
                    
                    # Check if all camera keys have corresponding images
                    missing_images = camera_image_names - uploaded_image_names
                    if missing_images:
                        errors.append(f"Missing images referenced in camera files: {missing_images}")
                
                # Check if we have at least one image
                if not self._info.images_uploaded:
                    errors.append("No images uploaded")
                else:
                    # Verify all uploaded images exist on disk
                    images_dir = self._job_dir / "input" / "images"
                    missing_image_files = []
                    for img_name in self._info.images_uploaded:
                        img_path = images_dir / img_name
                        if not img_path.exists():
                            missing_image_files.append(img_name)
                    if missing_image_files:
                        errors.append(f"Image files missing from disk: {missing_image_files}")
        
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
