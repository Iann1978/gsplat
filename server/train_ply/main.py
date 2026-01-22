"""FastAPI application for PLY-based Gaussian Splatting training service."""

import asyncio
import io
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import List, Optional

import aiofiles
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from .job_manager import JobManager
from .models import HealthResponse, JobInfo, JobStatus, TrainResponse, TrainingConfig
from .training_worker import run_training_job

# Configuration from environment variables
JOBS_DIR = os.getenv("TRAIN_PLY_JOBS_DIR", None)
RESULT_BASE_DIR = os.getenv("TRAIN_PLY_RESULT_BASE_DIR", None)
MAX_CONCURRENT_JOBS = int(os.getenv("TRAIN_PLY_MAX_CONCURRENT_JOBS", "1"))

# Initialize FastAPI app
app = FastAPI(
    title="Train PLY Service",
    description="REST API for training Gaussian Splatting models from PLY files",
    version="1.0.0",
)

# Initialize job manager
job_manager = JobManager(jobs_dir=JOBS_DIR)

# Track active training tasks
_active_tasks: dict[str, asyncio.Task] = {}


@app.post("/train", response_model=TrainResponse)
async def start_training(
    background_tasks: BackgroundTasks,
    ply_file: UploadFile = File(..., description="PLY file"),
    camera_json: UploadFile = File(..., description="Camera JSON file"),
    images: List[UploadFile] = File(..., description="Image files"),
    config_json: Optional[str] = Form(None, description="Training configuration as JSON"),
) -> TrainResponse:
    """Start a new training job.
    
    Args:
        background_tasks: FastAPI background tasks
        ply_file: PLY file upload
        camera_json: Camera JSON file upload
        images: List of image file uploads
        config_json: Optional training configuration JSON string
        
    Returns:
        TrainResponse with job_id and status
    """
    # Check concurrent job limit
    active_count = job_manager.get_active_jobs_count()
    if active_count >= MAX_CONCURRENT_JOBS:
        raise HTTPException(
            status_code=503,
            detail=f"Maximum concurrent jobs ({MAX_CONCURRENT_JOBS}) reached. Please wait for a job to complete.",
        )
    
    # Create job
    job_id = job_manager.create_job()
    job_dir = job_manager.get_job_dir(job_id)
    
    if job_dir is None:
        raise HTTPException(status_code=500, detail="Failed to create job directory")
    
    try:
        # Save uploaded files
        input_dir = job_dir / "input"
        images_dir = input_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Save PLY file
        ply_path = input_dir / "model.ply"
        async with aiofiles.open(ply_path, "wb") as f:
            content = await ply_file.read()
            await f.write(content)
        
        # Save camera JSON
        camera_json_path = input_dir / "cameras.json"
        async with aiofiles.open(camera_json_path, "wb") as f:
            content = await camera_json.read()
            await f.write(content)
        
        # Validate camera JSON
        try:
            async with aiofiles.open(camera_json_path, "r") as f:
                content = await f.read()
                camera_data = json.loads(content)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid camera JSON: {str(e)}")
        
        # Save images
        image_names = []
        for img_file in images:
            if img_file.filename:
                image_path = images_dir / img_file.filename
                async with aiofiles.open(image_path, "wb") as f:
                    content = await img_file.read()
                    await f.write(content)
                image_names.append(img_file.filename)
        
        # Validate that images match camera JSON
        camera_image_names = set(camera_data.keys())
        uploaded_image_names = set(image_names)
        
        if not camera_image_names.issubset(uploaded_image_names):
            missing = camera_image_names - uploaded_image_names
            raise HTTPException(
                status_code=400,
                detail=f"Missing images referenced in camera JSON: {missing}",
            )
        
        # Parse training config
        training_config = None
        if config_json:
            try:
                config_dict = json.loads(config_json)
                training_config = TrainingConfig(**config_dict)
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid training config JSON: {str(e)}",
                )
        
        # Start training task
        task = asyncio.create_task(
            run_training_job(
                job_id=job_id,
                job_manager=job_manager,
                ply_file_path=str(ply_path),
                camera_json_path=str(camera_json_path),
                data_dir=str(input_dir),
                training_config=training_config,
                result_base_dir=RESULT_BASE_DIR,
            )
        )
        _active_tasks[job_id] = task
        
        # Clean up task when done
        def cleanup_task(job_id: str):
            if job_id in _active_tasks:
                del _active_tasks[job_id]
        
        task.add_done_callback(lambda _: cleanup_task(job_id))
        
        return TrainResponse(
            job_id=job_id,
            status=JobStatus.PENDING,
            message="Training job created successfully",
        )
    
    except HTTPException:
        raise
    except Exception as e:
        # Clean up on error
        job_manager.update_job_status(
            job_id,
            JobStatus.FAILED,
            error_message=str(e),
        )
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")


@app.get("/train/{job_id}/status", response_model=JobInfo)
async def get_job_status(job_id: str) -> JobInfo:
    """Get the status of a training job.
    
    Args:
        job_id: Job identifier
        
    Returns:
        JobInfo with current status and progress
    """
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    return job


@app.get("/train/{job_id}/results")
async def get_job_results(job_id: str, file_type: Optional[str] = None) -> FileResponse:
    """Download training results.
    
    Args:
        job_id: Job identifier
        file_type: Optional file type filter (ply, checkpoint, render, stats, all)
        
    Returns:
        Zip file with results or individual file
    """
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job {job_id} is not completed (status: {job.status})",
        )
    
    if job.result_dir is None:
        raise HTTPException(status_code=404, detail="Results not found")
    
    result_dir = Path(job.result_dir)
    if not result_dir.exists():
        raise HTTPException(status_code=404, detail="Result directory does not exist")
    
    # If file_type is specified, return individual file or directory
    if file_type:
        if file_type == "ply":
            ply_dir = result_dir / "ply"
            if ply_dir.exists() and any(ply_dir.glob("*.ply")):
                # Return first PLY file or create zip of all
                ply_files = list(ply_dir.glob("*.ply"))
                if len(ply_files) == 1:
                    return FileResponse(ply_files[0], media_type="application/octet-stream")
                else:
                    # Create zip of all PLY files
                    return _create_zip_response(ply_files, f"{job_id}_ply.zip")
            raise HTTPException(status_code=404, detail="No PLY files found")
        
        elif file_type == "checkpoint":
            ckpt_dir = result_dir / "ckpts"
            if ckpt_dir.exists() and any(ckpt_dir.glob("*.pt")):
                ckpt_files = list(ckpt_dir.glob("*.pt"))
                if len(ckpt_files) == 1:
                    return FileResponse(ckpt_files[0], media_type="application/octet-stream")
                else:
                    return _create_zip_response(ckpt_files, f"{job_id}_checkpoints.zip")
            raise HTTPException(status_code=404, detail="No checkpoint files found")
        
        elif file_type == "render":
            render_dir = result_dir / "renders"
            if render_dir.exists():
                render_files = list(render_dir.glob("*.png"))
                if render_files:
                    return _create_zip_response(render_files, f"{job_id}_renders.zip")
            raise HTTPException(status_code=404, detail="No render files found")
        
        elif file_type == "stats":
            stats_dir = result_dir / "stats"
            if stats_dir.exists():
                stats_files = list(stats_dir.glob("*.json"))
                if stats_files:
                    return _create_zip_response(stats_files, f"{job_id}_stats.zip")
            raise HTTPException(status_code=404, detail="No stats files found")
    
    # Default: create zip of all results
    return _create_results_zip(job_id, result_dir)


@app.get("/train/{job_id}/logs")
async def get_job_logs(job_id: str) -> StreamingResponse:
    """Get training logs for a job.
    
    Args:
        job_id: Job identifier
        
    Returns:
        Text log content
    """
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    job_dir = job_manager.get_job_dir(job_id)
    if job_dir is None:
        raise HTTPException(status_code=404, detail="Job directory not found")
    
    log_file = job_dir / "logs" / "training.log"
    if not log_file.exists():
        # Return error log if available
        error_log = job_dir / "logs" / "error.log"
        if error_log.exists():
            async def read_error_log():
                async with aiofiles.open(error_log, "r") as f:
                    content = await f.read()
                    yield content
            return StreamingResponse(read_error_log(), media_type="text/plain")
        raise HTTPException(status_code=404, detail="Log file not found")
    
    async def read_log():
        async with aiofiles.open(log_file, "r") as f:
            async for line in f:
                yield line
    
    return StreamingResponse(read_log(), media_type="text/plain")


@app.delete("/train/{job_id}")
async def cancel_job(job_id: str) -> JSONResponse:
    """Cancel a running training job.
    
    Args:
        job_id: Job identifier
        
    Returns:
        JSON response with cancellation status
    """
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    if job.status not in [JobStatus.PENDING, JobStatus.RUNNING]:
        raise HTTPException(
            status_code=400,
            detail=f"Job {job_id} cannot be cancelled (status: {job.status})",
        )
    
    # Cancel the task if it exists
    if job_id in _active_tasks:
        task = _active_tasks[job_id]
        task.cancel()
        del _active_tasks[job_id]
    
    # Update job status
    success = job_manager.cancel_job(job_id)
    
    if success:
        return JSONResponse(
            content={"message": f"Job {job_id} cancelled successfully"},
            status_code=200,
        )
    else:
        raise HTTPException(status_code=500, detail="Failed to cancel job")


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint.
    
    Returns:
        HealthResponse with service status and active job count
    """
    active_jobs = job_manager.get_active_jobs_count()
    total_jobs = len(job_manager.list_jobs())
    
    return HealthResponse(
        status="healthy",
        active_jobs=active_jobs,
        total_jobs=total_jobs,
    )


def _create_zip_response(files: List[Path], zip_name: str) -> StreamingResponse:
    """Create a zip file response from a list of files.
    
    Args:
        files: List of file paths
        zip_name: Name for the zip file
        
    Returns:
        StreamingResponse with zip file
    """
    def generate_zip():
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for file_path in files:
                if file_path.exists():
                    zip_file.write(file_path, file_path.name)
        zip_buffer.seek(0)
        yield zip_buffer.read()
    
    return StreamingResponse(
        generate_zip(),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={zip_name}"},
    )


def _create_results_zip(job_id: str, result_dir: Path) -> StreamingResponse:
    """Create a zip file containing all results.
    
    Args:
        job_id: Job identifier
        result_dir: Result directory path
        
    Returns:
        StreamingResponse with zip file
    """
    def generate_zip():
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            # Add all files from result directory
            for root, dirs, files in os.walk(result_dir):
                for file in files:
                    file_path = Path(root) / file
                    if file_path.exists():
                        arc_name = file_path.relative_to(result_dir)
                        zip_file.write(file_path, arc_name)
        zip_buffer.seek(0)
        yield zip_buffer.read()
    
    return StreamingResponse(
        generate_zip(),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={job_id}_results.zip"},
    )


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(app, host=host, port=port)
