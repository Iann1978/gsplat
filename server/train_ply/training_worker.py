"""Background training worker for executing training jobs."""

import asyncio
import json
import os
import shutil
import sys
import traceback
from pathlib import Path
from typing import Optional

import torch
import tqdm

# Add parent directory to path for imports
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent
sys.path.insert(0, str(_project_root / "examples"))
sys.path.insert(0, str(_project_root))

from server.train_from_ply import PLYRunner, Config
from gsplat.strategy import DefaultStrategy, MCMCStrategy

from .job_manager import JobManager
from .models import JobStatus, TrainingConfig


class ProgressTrackingIterator:
    """Iterator wrapper that tracks progress and updates job status."""
    
    def __init__(self, iterable, job_id: str, job_manager: JobManager, 
                 max_steps: int, update_interval: int):
        """Initialize the iterator wrapper.
        
        Args:
            iterable: The iterable to wrap (e.g., range object)
            job_id: Job identifier
            job_manager: JobManager instance
            max_steps: Total training steps
            update_interval: Steps between progress updates
        """
        self._iterable = iter(iterable)
        self._job_id = job_id
        self._job_manager = job_manager
        self._max_steps = max_steps
        self._update_interval = update_interval
        self._last_update_step = -1
    
    def __iter__(self):
        """Return self as the iterator."""
        return self
    
    def __next__(self):
        """Get next value and update progress if needed."""
        value = next(self._iterable)
        
        # The value from the range iterator is the step number
        # For range(init_step, max_steps), value is the current step
        current_step = value
        
        # Check if this is the last step (or close to it)
        is_last_step = (current_step >= self._max_steps - 1)
        
        # Update job status if interval reached, this is the first step, or last step
        if (self._last_update_step < 0) or (current_step - self._last_update_step >= self._update_interval) or is_last_step:
            self._last_update_step = current_step
            self._update_job_progress(current_step)
        
        return value
    
    def _update_job_progress(self, step: int):
        """Update job progress status.
        
        Args:
            step: Current training step
        """
        try:
            if self._max_steps > 0:
                job_obj = self._job_manager._get_job_object(self._job_id)
                if job_obj:
                    job_obj.update_status(
                        JobStatus.RUNNING, 
                        current_step=step, 
                        max_steps=self._max_steps
                    )
        except Exception:
            # Silently ignore errors to prevent training failures
            # Progress updates are non-critical
            pass


class ProgressTrackingTqdm:
    """Wrapper around tqdm.tqdm that updates job progress periodically."""
    
    def __init__(self, original_tqdm_class, job_id: str, job_manager: JobManager, max_steps: int):
        """Initialize the wrapper.
        
        Args:
            original_tqdm_class: The original tqdm.tqdm class to wrap
            job_id: Job identifier
            job_manager: JobManager instance
            max_steps: Total training steps
        """
        self._original_tqdm = original_tqdm_class
        self._job_id = job_id
        self._job_manager = job_manager
        self._max_steps = max_steps
        # Calculate update interval: every 50 steps or 1% of max_steps, whichever is smaller
        self._update_interval = min(50, max(1, max_steps // 100))
    
    def __call__(self, *args, **kwargs):
        """Create a tqdm instance with progress tracking."""
        # Wrap the first argument (iterable) with progress tracking
        if args:
            iterable = args[0]
            wrapped_iterable = ProgressTrackingIterator(
                iterable,
                self._job_id,
                self._job_manager,
                self._max_steps,
                self._update_interval
            )
            # Replace first argument with wrapped iterable
            args = (wrapped_iterable,) + args[1:]
        
        # Create tqdm instance with wrapped iterable
        return self._original_tqdm(*args, **kwargs)


async def run_training_job(
    job_id: str,
    job_manager: JobManager,
    ply_file_path: str,
    camera_dir: str,
    data_dir: str,
    training_config: Optional[TrainingConfig] = None,
    result_base_dir: Optional[str] = None,
) -> None:
    """Run a training job asynchronously.
    
    Args:
        job_id: Job identifier
        job_manager: JobManager instance
        ply_file_path: Path to PLY file
        camera_dir: Path to cameras directory containing .cam.json files
        data_dir: Directory containing images folder (parent of images/)
        training_config: Optional training configuration
        result_base_dir: Base directory for results
    """
    try:
        # Update status to running
        job_obj = job_manager._get_job_object(job_id)
        if job_obj:
            job_obj.update_status(JobStatus.RUNNING)
        
        # Get job directory
        job_dir = job_manager.get_job_dir(job_id)
        if job_dir is None:
            raise ValueError(f"Job {job_id} not found")
        
        # Set up result directory
        if result_base_dir is None:
            result_base_dir = str(_project_root / "results" / "train_ply")
        
        result_dir = Path(result_base_dir) / job_id
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # Create training config from request
        cfg = _create_config_from_request(
            ply_path=ply_file_path,
            camera_dir=camera_dir,
            data_dir=data_dir,
            result_dir=str(result_dir),
            training_config=training_config,
        )
        
        # Update job with config
        job = job_manager.get_job(job_id)
        if job:
            job.config = training_config
            job.max_steps = cfg.max_steps
        
        # Run training in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            _run_training_sync,
            job_id,
            job_manager,
            cfg,
        )
        
        # Training completed successfully
        job_obj = job_manager._get_job_object(job_id)
        if job_obj:
            job_obj.update_status(JobStatus.COMPLETED)
        
        # Update result paths
        _update_job_results(job_id, job_manager, result_dir)
        
    except Exception as e:
        # Training failed
        error_msg = str(e)
        error_tb = traceback.format_exc()
        
        job_obj = job_manager._get_job_object(job_id)
        if job_obj:
            job_obj.update_status(JobStatus.FAILED, error_message=error_msg, error_traceback=error_tb)
        
        # Save error log
        job_dir = job_manager.get_job_dir(job_id)
        if job_dir:
            error_log_path = job_dir / "logs" / "error.log"
            with open(error_log_path, "w") as f:
                f.write(f"Error: {error_msg}\n\n")
                f.write(error_tb)


def _run_training_sync(
    job_id: str,
    job_manager: JobManager,
    cfg: Config,
) -> None:
    """Run training synchronously (called from executor).
    
    Args:
        job_id: Job identifier
        job_manager: JobManager instance
        cfg: Training configuration
    """
    # Redirect stdout/stderr to log file
    job_dir = job_manager.get_job_dir(job_id)
    if job_dir:
        log_file = job_dir / "logs" / "training.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Open log file
        log_fd = open(log_file, "w")
        
        # Save original stdout/stderr
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        # Redirect to log file
        sys.stdout = log_fd
        sys.stderr = log_fd
        
        # Save original tqdm.tqdm and patch it with progress tracking wrapper
        original_tqdm = tqdm.tqdm
        progress_tracker = ProgressTrackingTqdm(
            original_tqdm_class=original_tqdm,
            job_id=job_id,
            job_manager=job_manager,
            max_steps=cfg.max_steps
        )
        
        try:
            # Patch tqdm.tqdm with the wrapper
            tqdm.tqdm = progress_tracker
            
            # Create runner and train
            # Use single GPU (rank 0)
            runner = PLYRunner(
                local_rank=0,
                world_rank=0,
                world_size=1,
                cfg=cfg,
            )
            
            # Override runner's eval method to update job status
            original_eval = runner.eval
            
            def eval_with_progress(step: int, stage: str = "val"):
                # Update job progress
                if cfg.max_steps > 0:
                    progress = step / cfg.max_steps
                    job_obj = job_manager._get_job_object(job_id)
                    if job_obj:
                        job_obj.update_status(JobStatus.RUNNING, current_step=step, max_steps=cfg.max_steps)
                
                # Call original eval
                return original_eval(step, stage)
            
            runner.eval = eval_with_progress
            
            # Run training
            runner.train()
            
        finally:
            # Restore original tqdm.tqdm
            tqdm.tqdm = original_tqdm
            
            # Restore stdout/stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_fd.close()


def _create_config_from_request(
    ply_path: str,
    camera_dir: str,
    data_dir: str,
    result_dir: str,
    training_config: Optional[TrainingConfig] = None,
) -> Config:
    """Create Config from request parameters.
    
    Args:
        ply_path: Path to PLY file
        camera_dir: Path to cameras directory containing .cam.json files
        data_dir: Directory containing images
        result_dir: Directory for results
        training_config: Optional training configuration
        
    Returns:
        Config object
    """
    # Start with default config
    cfg = Config(
        ply_path=ply_path,
        camera_dir=camera_dir,
        data_dir=data_dir,
        result_dir=result_dir,
        disable_viewer=True,
    )
    
    # Apply training config if provided
    if training_config:
        cfg.max_steps = training_config.max_steps
        cfg.batch_size = training_config.batch_size
        cfg.steps_scaler = training_config.steps_scaler
        cfg.sh_degree = training_config.sh_degree
        cfg.sh_degree_interval = training_config.sh_degree_interval
        cfg.ssim_lambda = training_config.ssim_lambda
        cfg.means_lr = training_config.means_lr
        cfg.scales_lr = training_config.scales_lr
        cfg.opacities_lr = training_config.opacities_lr
        cfg.quats_lr = training_config.quats_lr
        cfg.sh0_lr = training_config.sh0_lr
        cfg.shN_lr = training_config.shN_lr or (training_config.sh0_lr / 20)
        cfg.opacity_reg = training_config.opacity_reg
        cfg.scale_reg = training_config.scale_reg
        cfg.test_every = training_config.test_every
        cfg.patch_size = training_config.patch_size
        cfg.normalize_world_space = training_config.normalize_world_space
        cfg.global_scale = training_config.global_scale
        cfg.camera_model = training_config.camera_model
        cfg.near_plane = training_config.near_plane
        cfg.far_plane = training_config.far_plane
        cfg.packed = training_config.packed
        cfg.antialiased = training_config.antialiased
        cfg.random_bkgd = training_config.random_bkgd
        cfg.sparse_grad = training_config.sparse_grad
        cfg.visible_adam = training_config.visible_adam
        cfg.pose_opt = training_config.pose_opt
        cfg.pose_opt_lr = training_config.pose_opt_lr
        cfg.pose_opt_reg = training_config.pose_opt_reg
        cfg.app_opt = training_config.app_opt
        cfg.app_opt_lr = training_config.app_opt_lr
        cfg.app_opt_reg = training_config.app_opt_reg
        cfg.depth_loss = training_config.depth_loss
        cfg.depth_lambda = training_config.depth_lambda
        cfg.save_ply = training_config.save_ply
        cfg.eval_steps = training_config.eval_steps
        cfg.save_steps = training_config.save_steps
        cfg.ply_steps = training_config.ply_steps
        cfg.tb_every = training_config.tb_every
        cfg.tb_save_image = training_config.tb_save_image
        cfg.with_ut = training_config.with_ut
        cfg.with_eval3d = training_config.with_eval3d
        cfg.lpips_net = training_config.lpips_net
        
        # Set strategy
        if training_config.strategy == "mcmc":
            cfg.strategy = MCMCStrategy(verbose=True)
        else:
            cfg.strategy = DefaultStrategy(verbose=True)
    
    # Ensure test_every doesn't result in empty training set
    # Count .cam.json files in cameras directory and adjust test_every if needed
    try:
        import glob
        camera_files = glob.glob(os.path.join(camera_dir, "*.cam.json"))
        num_images = len(camera_files)
        
        # If test_every >= num_images, all images would go to validation
        # Set test_every to num_images + 1 to ensure at least one image goes to training
        if cfg.test_every >= num_images:
            cfg.test_every = num_images + 1
    except Exception:
        # If we can't count camera files, let it fail later with a clearer error
        pass
    
    # Apply steps scaler
    cfg.adjust_steps(cfg.steps_scaler)
    
    return cfg


def _update_job_results(
    job_id: str,
    job_manager: JobManager,
    result_dir: Path,
) -> None:
    """Update job with result file paths.
    
    Args:
        job_id: Job identifier
        job_manager: JobManager instance
        result_dir: Result directory path
    """
    ply_files = []
    checkpoint_files = []
    
    # Find PLY files
    ply_dir = result_dir / "ply"
    if ply_dir.exists():
        ply_files = sorted([str(f) for f in ply_dir.glob("*.ply")])
    
    # Find checkpoint files
    ckpt_dir = result_dir / "ckpts"
    if ckpt_dir.exists():
        checkpoint_files = sorted([str(f) for f in ckpt_dir.glob("*.pt")])
    
    job_obj = job_manager._get_job_object(job_id)
    if job_obj:
        job_obj.update_results(result_dir=str(result_dir), ply_files=ply_files, checkpoint_files=checkpoint_files)
