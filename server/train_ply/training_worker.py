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

# Add parent directory to path for imports
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent
sys.path.insert(0, str(_project_root / "examples"))
sys.path.insert(0, str(_project_root))

from server.train_from_ply import PLYRunner, Config
from gsplat.strategy import DefaultStrategy, MCMCStrategy

from .job_manager import JobManager
from .models import JobStatus, TrainingConfig


async def run_training_job(
    job_id: str,
    job_manager: JobManager,
    ply_file_path: str,
    camera_json_path: str,
    data_dir: str,
    training_config: Optional[TrainingConfig] = None,
    result_base_dir: Optional[str] = None,
) -> None:
    """Run a training job asynchronously.
    
    Args:
        job_id: Job identifier
        job_manager: JobManager instance
        ply_file_path: Path to PLY file
        camera_json_path: Path to camera JSON file
        data_dir: Directory containing images folder (parent of images/)
        training_config: Optional training configuration
        result_base_dir: Base directory for results
    """
    try:
        # Update status to running
        job_manager.update_job_status(job_id, JobStatus.RUNNING)
        
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
            camera_json=camera_json_path,
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
        job_manager.update_job_status(job_id, JobStatus.COMPLETED)
        
        # Update result paths
        _update_job_results(job_id, job_manager, result_dir)
        
    except Exception as e:
        # Training failed
        error_msg = str(e)
        error_tb = traceback.format_exc()
        
        job_manager.update_job_status(
            job_id,
            JobStatus.FAILED,
            error_message=error_msg,
            error_traceback=error_tb,
        )
        
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
        
        try:
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
                    job_manager.update_job_status(
                        job_id,
                        JobStatus.RUNNING,
                        current_step=step,
                        max_steps=cfg.max_steps,
                    )
                
                # Call original eval
                return original_eval(step, stage)
            
            runner.eval = eval_with_progress
            
            # Run training
            runner.train()
            
        finally:
            # Restore stdout/stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_fd.close()


def _create_config_from_request(
    ply_path: str,
    camera_json: str,
    data_dir: str,
    result_dir: str,
    training_config: Optional[TrainingConfig] = None,
) -> Config:
    """Create Config from request parameters.
    
    Args:
        ply_path: Path to PLY file
        camera_json: Path to camera JSON file
        data_dir: Directory containing images
        result_dir: Directory for results
        training_config: Optional training configuration
        
    Returns:
        Config object
    """
    # Start with default config
    cfg = Config(
        ply_path=ply_path,
        camera_json=camera_json,
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
    
    job_manager.update_job_results(
        job_id,
        result_dir=str(result_dir),
        ply_files=ply_files,
        checkpoint_files=checkpoint_files,
    )
