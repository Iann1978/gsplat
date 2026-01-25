"""Pydantic models for request/response validation."""

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Job status enumeration."""
    UPLOADING = "uploading"
    READY = "ready"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingConfig(BaseModel):
    """Training configuration parameters."""
    
    # Training parameters
    max_steps: int = Field(default=30000, ge=1, description="Number of training steps")
    batch_size: int = Field(default=1, ge=1, description="Batch size for training")
    steps_scaler: float = Field(default=1.0, gt=0, description="Global factor to scale training steps")
    
    # Spherical harmonics
    sh_degree: int = Field(default=3, ge=0, le=3, description="Degree of spherical harmonics")
    sh_degree_interval: int = Field(default=1000, ge=1, description="Steps between SH degree increases")
    
    # Loss weights
    ssim_lambda: float = Field(default=0.2, ge=0, le=1, description="Weight for SSIM loss")
    
    # Learning rates
    means_lr: float = Field(default=1.6e-4, gt=0, description="Learning rate for positions")
    scales_lr: float = Field(default=5e-3, gt=0, description="Learning rate for scales")
    opacities_lr: float = Field(default=5e-2, gt=0, description="Learning rate for opacities")
    quats_lr: float = Field(default=1e-3, gt=0, description="Learning rate for quaternions")
    sh0_lr: float = Field(default=2.5e-3, gt=0, description="Learning rate for SH DC")
    shN_lr: Optional[float] = Field(default=None, gt=0, description="Learning rate for SH rest (defaults to sh0_lr / 20)")
    
    # Regularization
    opacity_reg: float = Field(default=0.0, ge=0, description="Opacity regularization weight")
    scale_reg: float = Field(default=0.0, ge=0, description="Scale regularization weight")
    
    # Data splitting
    test_every: int = Field(default=8, ge=1, description="Every N images is a test image")
    patch_size: Optional[int] = Field(default=None, ge=1, description="Random crop size for training")
    
    # Scene settings
    normalize_world_space: bool = Field(default=False, description="Normalize scene to unit sphere")
    global_scale: float = Field(default=1.0, gt=0, description="Global scale factor")
    camera_model: str = Field(default="pinhole", pattern="^(pinhole|ortho|fisheye)$", description="Camera model type")
    
    # Rendering settings
    near_plane: float = Field(default=0.01, gt=0, description="Near plane clipping distance")
    far_plane: float = Field(default=1e10, gt=0, description="Far plane clipping distance")
    packed: bool = Field(default=False, description="Use packed mode for rasterization")
    antialiased: bool = Field(default=False, description="Enable anti-aliasing in rasterization")
    random_bkgd: bool = Field(default=False, description="Use random background for training")
    
    # Strategy
    strategy: str = Field(default="default", pattern="^(default|mcmc)$", description="Densification strategy")
    
    # Advanced options
    sparse_grad: bool = Field(default=False, description="Use sparse gradients")
    visible_adam: bool = Field(default=False, description="Use visible adam optimizer")
    pose_opt: bool = Field(default=False, description="Enable camera pose optimization")
    pose_opt_lr: float = Field(default=1e-5, gt=0, description="Learning rate for pose optimization")
    pose_opt_reg: float = Field(default=1e-6, ge=0, description="Regularization for pose optimization")
    app_opt: bool = Field(default=False, description="Enable appearance optimization")
    app_opt_lr: float = Field(default=1e-3, gt=0, description="Learning rate for appearance optimization")
    app_opt_reg: float = Field(default=1e-6, ge=0, description="Regularization for appearance optimization")
    depth_loss: bool = Field(default=False, description="Enable depth loss")
    depth_lambda: float = Field(default=1e-2, ge=0, description="Weight for depth loss")
    
    # Output settings
    save_ply: bool = Field(default=True, description="Save PLY files during training")
    eval_steps: List[int] = Field(default_factory=lambda: [7000, 30000], description="Steps at which to evaluate")
    save_steps: List[int] = Field(default_factory=lambda: [7000, 30000], description="Steps at which to save checkpoints")
    ply_steps: List[int] = Field(default_factory=lambda: [7000, 30000], description="Steps at which to save PLY files")
    
    # TensorBoard
    tb_every: int = Field(default=100, ge=1, description="Steps between tensorboard logs")
    tb_save_image: bool = Field(default=False, description="Save training images to tensorboard")
    
    # 3DGUT
    with_ut: bool = Field(default=False, description="Enable unscented transform")
    with_eval3d: bool = Field(default=False, description="Enable 3D evaluation")
    
    # LPIPS
    lpips_net: str = Field(default="alex", pattern="^(vgg|alex)$", description="LPIPS network type")


class TrainResponse(BaseModel):
    """Response when starting a training job."""
    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Initial job status")
    message: str = Field(default="Training job created successfully", description="Status message")


class JobInfo(BaseModel):
    """Detailed job information."""
    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    created_at: datetime = Field(..., description="Job creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    
    # Progress information
    current_step: Optional[int] = Field(default=None, ge=0, description="Current training step")
    max_steps: Optional[int] = Field(default=None, ge=1, description="Total training steps")
    progress: Optional[float] = Field(default=None, ge=0, le=1, description="Training progress (0-1)")
    
    # Result paths
    result_dir: Optional[str] = Field(default=None, description="Directory containing results")
    ply_files: List[str] = Field(default_factory=list, description="List of generated PLY files")
    checkpoint_files: List[str] = Field(default_factory=list, description="List of checkpoint files")
    
    # Error information
    error_message: Optional[str] = Field(default=None, description="Error message if job failed")
    error_traceback: Optional[str] = Field(default=None, description="Error traceback if job failed")
    
    # Training configuration
    config: Optional[TrainingConfig] = Field(default=None, description="Training configuration used")
    
    # Upload tracking
    ply_uploaded: bool = Field(default=False, description="Whether PLY file is uploaded")
    cameras_uploaded: bool = Field(default=False, description="Whether camera.json is uploaded")
    images_uploaded: List[str] = Field(default_factory=list, description="List of uploaded image filenames")
    config_uploaded: bool = Field(default=False, description="Whether training config is uploaded")
    validation_errors: List[str] = Field(default_factory=list, description="List of validation errors")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(default="healthy", description="Service status")
    active_jobs: int = Field(default=0, ge=0, description="Number of active training jobs")
    total_jobs: int = Field(default=0, ge=0, description="Total number of jobs")


class UploadStatusResponse(BaseModel):
    """Response for upload status check."""
    job_id: str = Field(..., description="Job identifier")
    status: JobStatus = Field(..., description="Current job status")
    ply_uploaded: bool = Field(..., description="Whether PLY file is uploaded")
    cameras_uploaded: bool = Field(..., description="Whether camera.json is uploaded")
    images_uploaded: List[str] = Field(..., description="List of uploaded image filenames")
    config_uploaded: bool = Field(..., description="Whether training config is uploaded")
    validation_errors: List[str] = Field(..., description="List of validation errors")
    is_ready: bool = Field(..., description="Whether job is ready to start training")


class ValidateResponse(BaseModel):
    """Response for manual validation."""
    job_id: str = Field(..., description="Job identifier")
    status: JobStatus = Field(..., description="Current job status after validation")
    is_ready: bool = Field(..., description="Whether job is ready to start training")
    validation_errors: List[str] = Field(..., description="List of validation errors")
