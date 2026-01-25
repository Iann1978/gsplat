"""FastAPI application for PLY-based Gaussian Splatting training service."""

import asyncio
import io
import json
import logging
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import List, Optional

import aiofiles
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from .job_manager import JobManager
from .models import HealthResponse, JobInfo, JobStatus, TrainResponse, TrainingConfig
from .training_worker import run_training_job

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create logs directory
_log_dir = Path(__file__).parent / "logs"
_log_dir.mkdir(parents=True, exist_ok=True)

# Configure file handler for API logs
_log_file = _log_dir / "api.log"
_file_handler = logging.FileHandler(_log_file, mode='a', encoding='utf-8')
_file_handler.setLevel(logging.DEBUG)

# Configure log format
_log_format = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
_file_handler.setFormatter(_log_format)

# Add file handler to logger
logger.addHandler(_file_handler)

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


@app.post("/jobs", response_model=TrainResponse, status_code=201)
async def create_job() -> TrainResponse:
    """Create a new training job.
    
    Returns:
        TrainResponse with job_id and UPLOADING status
    """
    logger.info("POST /jobs - Creating new job")
    job_id = job_manager.create_job()
    logger.info(f"POST /jobs - Created job: {job_id}")
    
    return TrainResponse(
        job_id=job_id,
        status=JobStatus.UPLOADING,
        message="Job created successfully",
    )


@app.get("/jobs/{job_id}", response_model=JobInfo)
async def get_job_status(job_id: str) -> JobInfo:
    """Get comprehensive job status and information.
    
    Automatically validates job if status is UPLOADING or READY.
    
    Args:
        job_id: Job identifier
        
    Returns:
        JobInfo with current status, progress, upload status, and validation errors
    """
    logger.info(f"GET /jobs/{job_id} - Job status requested")
    job = job_manager.get_job(job_id)
    if job is None:
        logger.error(f"GET /jobs/{job_id} - Job not found: {job_id}")
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    # Auto-validate if job is in UPLOADING or READY state
    if job.status in [JobStatus.UPLOADING, JobStatus.READY]:
        logger.debug(f"GET /jobs/{job_id} - Auto-validating job")
        job_obj = job_manager._get_job_object(job_id)
        if job_obj:
            job_obj.validate_ready()
        job = job_manager.get_job(job_id)  # Refresh to get updated validation_errors
    
    logger.info(f"GET /jobs/{job_id} - Returning status: {job.status}")
    return job


@app.get("/jobs/{job_id}/results")
async def get_job_results(job_id: str, file_type: Optional[str] = None) -> FileResponse:
    """Download training results.
    
    Args:
        job_id: Job identifier
        file_type: Optional file type filter (ply, checkpoint, render, stats, all)
        
    Returns:
        Zip file with results or individual file
    """
    logger.info(f"GET /jobs/{job_id}/results - Results requested (file_type: {file_type or 'all'})")
    job = job_manager.get_job(job_id)
    if job is None:
        logger.error(f"GET /jobs/{job_id}/results - Job not found: {job_id}")
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    if job.status != JobStatus.COMPLETED:
        logger.warning(f"GET /jobs/{job_id}/results - Job not completed (status: {job.status})")
        raise HTTPException(
            status_code=400,
            detail=f"Job {job_id} is not completed (status: {job.status})",
        )
    
    if job.result_dir is None:
        logger.error(f"GET /jobs/{job_id}/results - Results not found for job: {job_id}")
        raise HTTPException(status_code=404, detail="Results not found")
    
    result_dir = Path(job.result_dir)
    if not result_dir.exists():
        logger.error(f"GET /jobs/{job_id}/results - Result directory does not exist: {result_dir}")
        raise HTTPException(status_code=404, detail="Result directory does not exist")
    
    # If file_type is specified, return individual file or directory
    if file_type:
        if file_type == "ply":
            ply_dir = result_dir / "ply"
            if ply_dir.exists() and any(ply_dir.glob("*.ply")):
                # Return first PLY file or create zip of all
                ply_files = list(ply_dir.glob("*.ply"))
                logger.info(f"GET /jobs/{job_id}/results - Returning {len(ply_files)} PLY file(s)")
                if len(ply_files) == 1:
                    return FileResponse(ply_files[0], media_type="application/octet-stream")
                else:
                    # Create zip of all PLY files
                    return _create_zip_response(ply_files, f"{job_id}_ply.zip")
            logger.error(f"GET /jobs/{job_id}/results - No PLY files found")
            raise HTTPException(status_code=404, detail="No PLY files found")
        
        elif file_type == "checkpoint":
            ckpt_dir = result_dir / "ckpts"
            if ckpt_dir.exists() and any(ckpt_dir.glob("*.pt")):
                ckpt_files = list(ckpt_dir.glob("*.pt"))
                logger.info(f"GET /jobs/{job_id}/results - Returning {len(ckpt_files)} checkpoint file(s)")
                if len(ckpt_files) == 1:
                    return FileResponse(ckpt_files[0], media_type="application/octet-stream")
                else:
                    return _create_zip_response(ckpt_files, f"{job_id}_checkpoints.zip")
            logger.error(f"GET /jobs/{job_id}/results - No checkpoint files found")
            raise HTTPException(status_code=404, detail="No checkpoint files found")
        
        elif file_type == "render":
            render_dir = result_dir / "renders"
            if render_dir.exists():
                render_files = list(render_dir.glob("*.png"))
                if render_files:
                    logger.info(f"GET /jobs/{job_id}/results - Returning {len(render_files)} render file(s)")
                    return _create_zip_response(render_files, f"{job_id}_renders.zip")
            logger.error(f"GET /jobs/{job_id}/results - No render files found")
            raise HTTPException(status_code=404, detail="No render files found")
        
        elif file_type == "stats":
            stats_dir = result_dir / "stats"
            if stats_dir.exists():
                stats_files = list(stats_dir.glob("*.json"))
                if stats_files:
                    logger.info(f"GET /jobs/{job_id}/results - Returning {len(stats_files)} stats file(s)")
                    return _create_zip_response(stats_files, f"{job_id}_stats.zip")
            logger.error(f"GET /jobs/{job_id}/results - No stats files found")
            raise HTTPException(status_code=404, detail="No stats files found")
    
    # Default: create zip of all results
    logger.info(f"GET /jobs/{job_id}/results - Returning all results as zip")
    return _create_results_zip(job_id, result_dir)


@app.get("/jobs/{job_id}/logs")
async def get_job_logs(job_id: str) -> StreamingResponse:
    """Get training logs for a job.
    
    Args:
        job_id: Job identifier
        
    Returns:
        Text log content
    """
    logger.info(f"GET /jobs/{job_id}/logs - Logs requested")
    job = job_manager.get_job(job_id)
    if job is None:
        logger.error(f"GET /jobs/{job_id}/logs - Job not found: {job_id}")
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    job_dir = job_manager.get_job_dir(job_id)
    if job_dir is None:
        logger.error(f"GET /jobs/{job_id}/logs - Job directory not found: {job_id}")
        raise HTTPException(status_code=404, detail="Job directory not found")
    
    log_file = job_dir / "logs" / "training.log"
    if not log_file.exists():
        # Return error log if available
        error_log = job_dir / "logs" / "error.log"
        if error_log.exists():
            logger.info(f"GET /jobs/{job_id}/logs - Returning error log")
            async def read_error_log():
                async with aiofiles.open(error_log, "r") as f:
                    content = await f.read()
                    yield content
            return StreamingResponse(read_error_log(), media_type="text/plain")
        logger.error(f"GET /jobs/{job_id}/logs - Log file not found")
        raise HTTPException(status_code=404, detail="Log file not found")
    
    logger.info(f"GET /jobs/{job_id}/logs - Returning training log")
    async def read_log():
        async with aiofiles.open(log_file, "r") as f:
            async for line in f:
                yield line
    
    return StreamingResponse(read_log(), media_type="text/plain")


@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str) -> JSONResponse:
    """Cancel or delete a job.
    
    Handles both active training jobs (PENDING/RUNNING) and incomplete uploads (UPLOADING/READY).
    
    Args:
        job_id: Job identifier
        
    Returns:
        JSON response with cancellation status
    """
    logger.info(f"DELETE /jobs/{job_id} - Cancellation requested")
    job = job_manager.get_job(job_id)
    if job is None:
        logger.error(f"DELETE /jobs/{job_id} - Job not found: {job_id}")
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    # Handle different job states
    if job.status in [JobStatus.UPLOADING, JobStatus.READY]:
        # Cleanup upload job (full cleanup: delete directory and remove from memory)
        logger.info(f"DELETE /jobs/{job_id} - Cleaning up upload job")
        success = job_manager.cleanup_job(job_id)
        if success:
            logger.info(f"DELETE /jobs/{job_id} - Upload job cancelled successfully")
            return JSONResponse(
                content={"message": f"Job {job_id} cancelled successfully"},
                status_code=200,
            )
        else:
            logger.error(f"DELETE /jobs/{job_id} - Failed to cancel upload job")
            raise HTTPException(status_code=500, detail="Failed to cancel upload job")
    
    elif job.status in [JobStatus.PENDING, JobStatus.RUNNING]:
        # Cancel training job
        # Cancel the async task if it exists
        if job_id in _active_tasks:
            task = _active_tasks[job_id]
            task.cancel()
            del _active_tasks[job_id]
            logger.info(f"DELETE /jobs/{job_id} - Cancelled active task")
        
        # Update job status via cleanup_job (keeps directory/memory for logs/results)
        success = job_manager.cleanup_job(job_id)
        
        if success:
            logger.info(f"DELETE /jobs/{job_id} - Training job cancelled successfully")
            return JSONResponse(
                content={"message": f"Job {job_id} cancelled successfully"},
                status_code=200,
            )
        else:
            logger.error(f"DELETE /jobs/{job_id} - Failed to cancel job")
            raise HTTPException(status_code=500, detail="Failed to cancel job")
    else:
        # Cannot cancel completed/failed/cancelled jobs
        logger.warning(f"DELETE /jobs/{job_id} - Job cannot be cancelled (status: {job.status})")
        raise HTTPException(
            status_code=400,
            detail=f"Job {job_id} cannot be cancelled (status: {job.status})",
        )


@app.post("/jobs/{job_id}/ply")
async def upload_ply(job_id: str, ply_file: UploadFile = File(..., description="PLY file")) -> JSONResponse:
    """Upload PLY file to a job.
    
    Args:
        job_id: Job identifier
        ply_file: PLY file upload
        
    Returns:
        JSON response with upload status
    """
    logger.info(f"POST /jobs/{job_id}/ply - PLY upload requested")
    job = job_manager.get_job(job_id)
    if job is None:
        logger.error(f"POST /jobs/{job_id}/ply - Job not found: {job_id}")
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    if job.status != JobStatus.UPLOADING:
        logger.warning(f"POST /jobs/{job_id}/ply - Job not in UPLOADING state (current: {job.status})")
        raise HTTPException(
            status_code=400,
            detail=f"Job {job_id} is not in UPLOADING state (current: {job.status})",
        )
    
    job_dir = job_manager.get_job_dir(job_id)
    if job_dir is None:
        logger.error(f"POST /jobs/{job_id}/ply - Job directory not found")
        raise HTTPException(status_code=500, detail="Job directory not found")
    
    # Save PLY file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ply") as tmp_file:
        content = await ply_file.read()
        tmp_file.write(content)
        tmp_path = Path(tmp_file.name)
    
    try:
        job_obj = job_manager._get_job_object(job_id)
        success = job_obj.upload_ply(tmp_path) if job_obj else False
        if not success:
            logger.error(f"POST /jobs/{job_id}/ply - Failed to upload PLY file")
            raise HTTPException(status_code=500, detail="Failed to upload PLY file")
        
        # Refresh job to get updated status
        job = job_manager.get_job(job_id)
        logger.info(f"POST /jobs/{job_id}/ply - PLY file uploaded successfully")
        
        return JSONResponse(
            content={
                "job_id": job_id,
                "status": job.status,
                "ply_uploaded": True,
            },
            status_code=200,
        )
    finally:
        # Clean up temp file
        if tmp_path.exists():
            tmp_path.unlink()


@app.post("/jobs/{job_id}/images")
async def upload_images(job_id: str, images: List[UploadFile] = File(..., description="Image files")) -> JSONResponse:
    """Upload one or more image files to a job.
    
    Args:
        job_id: Job identifier
        images: List of image file uploads
        
    Returns:
        JSON response with upload status
    """
    logger.info(f"POST /jobs/{job_id}/images - Image upload requested ({len(images)} images)")
    job = job_manager.get_job(job_id)
    if job is None:
        logger.error(f"POST /jobs/{job_id}/images - Job not found: {job_id}")
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    if job.status != JobStatus.UPLOADING:
        logger.warning(f"POST /jobs/{job_id}/images - Job not in UPLOADING state (current: {job.status})")
        raise HTTPException(
            status_code=400,
            detail=f"Job {job_id} is not in UPLOADING state (current: {job.status})",
        )
    
    job_dir = job_manager.get_job_dir(job_id)
    if job_dir is None:
        logger.error(f"POST /jobs/{job_id}/images - Job directory not found")
        raise HTTPException(status_code=500, detail="Job directory not found")
    
    uploaded_images = []
    failed_images = []
    
    # Process each image
    for image in images:
        if not image.filename:
            logger.warning(f"POST /jobs/{job_id}/images - Skipping image without filename")
            continue
        
        # Check for duplicates
        if image.filename in job.images_uploaded:
            logger.warning(f"POST /jobs/{job_id}/images - Image {image.filename} already uploaded")
            failed_images.append(f"{image.filename} (already uploaded)")
            continue
        
        # Save image file temporarily
        ext = Path(image.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            content = await image.read()
            tmp_file.write(content)
            tmp_path = Path(tmp_file.name)
        
        try:
            job_obj = job_manager._get_job_object(job_id)
            success = job_obj.upload_image(image.filename, tmp_path) if job_obj else False
            if success:
                uploaded_images.append(image.filename)
            else:
                failed_images.append(f"{image.filename} (upload failed)")
        finally:
            # Clean up temp file
            if tmp_path.exists():
                tmp_path.unlink()
    
    if failed_images and not uploaded_images:
        # All uploads failed
        logger.error(f"POST /jobs/{job_id}/images - All image uploads failed")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to upload images: {', '.join(failed_images)}",
        )
    
    # Refresh job to get updated images list
    job = job_manager.get_job(job_id)
    
    response_content = {
        "job_id": job_id,
        "status": job.status,
        "images_uploaded": job.images_uploaded.copy(),
    }
    
    if failed_images:
        response_content["failed_images"] = failed_images
        logger.warning(f"POST /jobs/{job_id}/images - Some images failed to upload: {failed_images}")
    
    logger.info(f"POST /jobs/{job_id}/images - Uploaded {len(uploaded_images)} image(s) successfully")
    
    return JSONResponse(
        content=response_content,
        status_code=200,
    )


@app.post("/jobs/{job_id}/cameras")
async def upload_cameras(job_id: str, cameras_json: UploadFile = File(..., description="Camera JSON file")) -> JSONResponse:
    """Upload camera.json file to a job.
    
    Args:
        job_id: Job identifier
        cameras_json: Camera JSON file upload
        
    Returns:
        JSON response with upload status
    """
    logger.info(f"POST /jobs/{job_id}/cameras - Camera JSON upload requested")
    job = job_manager.get_job(job_id)
    if job is None:
        logger.error(f"POST /jobs/{job_id}/cameras - Job not found: {job_id}")
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    if job.status != JobStatus.UPLOADING:
        logger.warning(f"POST /jobs/{job_id}/cameras - Job not in UPLOADING state (current: {job.status})")
        raise HTTPException(
            status_code=400,
            detail=f"Job {job_id} is not in UPLOADING state (current: {job.status})",
        )
    
    job_dir = job_manager.get_job_dir(job_id)
    if job_dir is None:
        logger.error(f"POST /jobs/{job_id}/cameras - Job directory not found")
        raise HTTPException(status_code=500, detail="Job directory not found")
    
    # Save camera JSON file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="wb") as tmp_file:
        content = await cameras_json.read()
        tmp_file.write(content)
        tmp_path = Path(tmp_file.name)
    
    try:
        job_obj = job_manager._get_job_object(job_id)
        success = job_obj.upload_cameras(tmp_path) if job_obj else False
        if not success:
            logger.error(f"POST /jobs/{job_id}/cameras - Invalid camera JSON format")
            raise HTTPException(status_code=400, detail="Invalid camera JSON format")
        
        # Refresh job to get updated status
        job = job_manager.get_job(job_id)
        logger.info(f"POST /jobs/{job_id}/cameras - Camera JSON uploaded successfully")
        
        return JSONResponse(
            content={
                "job_id": job_id,
                "status": job.status,
                "cameras_uploaded": True,
            },
            status_code=200,
        )
    finally:
        # Clean up temp file
        if tmp_path.exists():
            tmp_path.unlink()


@app.post("/jobs/{job_id}/config")
async def upload_config(job_id: str, config_json: str = Form(..., description="Training configuration as JSON")) -> JSONResponse:
    """Upload training configuration to a job.
    
    Args:
        job_id: Job identifier
        config_json: Training configuration as JSON string
        
    Returns:
        JSON response with upload status
    """
    logger.info(f"POST /jobs/{job_id}/config - Config upload requested")
    job = job_manager.get_job(job_id)
    if job is None:
        logger.error(f"POST /jobs/{job_id}/config - Job not found: {job_id}")
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    if job.status != JobStatus.UPLOADING:
        logger.warning(f"POST /jobs/{job_id}/config - Job not in UPLOADING state (current: {job.status})")
        raise HTTPException(
            status_code=400,
            detail=f"Job {job_id} is not in UPLOADING state (current: {job.status})",
        )
    
    # Parse training config
    try:
        config_dict = json.loads(config_json)
        training_config = TrainingConfig(**config_dict)
    except Exception as e:
        logger.error(f"POST /jobs/{job_id}/config - Invalid training config JSON: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid training config JSON: {str(e)}",
        )
    
    job_obj = job_manager._get_job_object(job_id)
    if job_obj:
        job_obj.upload_config(training_config)
        success = True
    else:
        success = False
    if not success:
        logger.error(f"POST /jobs/{job_id}/config - Failed to upload config")
        raise HTTPException(status_code=500, detail="Failed to upload config")
    
    job = job_manager.get_job(job_id)
    logger.info(f"POST /jobs/{job_id}/config - Config uploaded successfully")
    
    return JSONResponse(
        content={
            "job_id": job_id,
            "status": job.status,
            "config_uploaded": True,
        },
        status_code=200,
    )


@app.post("/jobs/{job_id}/start", response_model=TrainResponse)
async def start_training_from_upload(job_id: str) -> TrainResponse:
    """Start training from an uploaded job.
    
    Automatically validates the job before starting. If validation fails, returns error details.
    
    Args:
        job_id: Job identifier
        
    Returns:
        TrainResponse with job_id and PENDING status
    """
    logger.info(f"POST /jobs/{job_id}/start - Start training requested")
    job = job_manager.get_job(job_id)
    if job is None:
        logger.error(f"POST /jobs/{job_id}/start - Job not found: {job_id}")
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    # Check concurrent job limit
    active_count = job_manager.get_active_jobs_count()
    if active_count >= MAX_CONCURRENT_JOBS:
        logger.warning(f"POST /jobs/{job_id}/start - Maximum concurrent jobs ({MAX_CONCURRENT_JOBS}) reached")
        raise HTTPException(
            status_code=503,
            detail=f"Maximum concurrent jobs ({MAX_CONCURRENT_JOBS}) reached. Please wait for a job to complete.",
        )
    
    # Validate and start (auto-validation)
    logger.debug(f"POST /jobs/{job_id}/start - Validating job before starting")
    job_obj = job_manager._get_job_object(job_id)
    if job_obj:
        is_ready, errors = job_obj.validate_ready()
    else:
        is_ready, errors = False, [f"Job {job_id} not found"]
    if not is_ready:
        logger.warning(f"POST /jobs/{job_id}/start - Job validation failed: {errors}")
        raise HTTPException(
            status_code=400,
            detail=f"Job {job_id} is not ready to start. Errors: {errors}",
        )
    
    job_obj = job_manager._get_job_object(job_id)
    success = job_obj.start_training() if job_obj else False
    if not success:
        logger.error(f"POST /jobs/{job_id}/start - Failed to start training")
        raise HTTPException(
            status_code=400,
            detail=f"Job {job_id} cannot be started (status: {job.status})",
        )
    
    job_dir = job_manager.get_job_dir(job_id)
    if job_dir is None:
        logger.error(f"POST /jobs/{job_id}/start - Job directory not found")
        raise HTTPException(status_code=500, detail="Job directory not found")
    
    # Get paths
    ply_path = job_dir / "input" / "model.ply"
    camera_json_path = job_dir / "input" / "cameras.json"
    input_dir = job_dir / "input"
    
    # Get training config
    job = job_manager.get_job(job_id)
    training_config = job.config if job else None
    
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
    
    logger.info(f"POST /jobs/{job_id}/start - Training started successfully")
    return TrainResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        message="Training started successfully",
    )


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint.
    
    Returns:
        HealthResponse with service status and active job count
    """
    active_jobs = job_manager.get_active_jobs_count()
    total_jobs = len(job_manager.list_jobs())
    
    logger.info(f"GET /health - Health check: {active_jobs} active jobs, {total_jobs} total jobs")
    
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
