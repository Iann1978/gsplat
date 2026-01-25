# Train PLY Service API

A FastAPI-based REST API server for training Gaussian Splatting models from PLY files using custom camera data.

## Features

- **PLY-based training**: Submit PLY files with camera data and images to train Gaussian Splatting models
- **Async job management**: Long-running training jobs are executed asynchronously with status tracking
- **FastAPI framework**: Modern, fast, with automatic API documentation
- **Request validation**: Automatic validation of inputs using Pydantic
- **Error handling**: Comprehensive error handling with clear error messages
- **Progress tracking**: Real-time job status and progress updates
- **Result download**: Download trained models, checkpoints, renders, and statistics

## Installation

1. Install the server dependencies:
```bash
pip install -r server/train_ply/requirements.txt
```

2. Ensure gsplat and its dependencies are installed (as per main repository instructions)

3. Ensure you have CUDA-capable GPU for training

## Configuration

The server can be configured using environment variables:

- `TRAIN_PLY_JOBS_DIR`: Directory for job data (default: `server/train_ply/jobs/`)
- `TRAIN_PLY_RESULT_BASE_DIR`: Base directory for results (default: `results/train_ply/`)
- `TRAIN_PLY_MAX_CONCURRENT_JOBS`: Maximum parallel jobs (default: `1`)
- `TRAIN_PLY_JOB_RETENTION_DAYS`: Days to keep completed jobs (default: `7`)
- `CUDA_VISIBLE_DEVICES`: GPU selection (e.g., `0` or `0,1,2,3`)
- `PORT`: Server port (default: `8000`)
- `HOST`: Server host (default: `0.0.0.0`)

## Running the Server

### Basic usage:
```bash
python -m server.train_ply.main
```

Or using uvicorn directly:
```bash
uvicorn server.train_ply.main:app --host 0.0.0.0 --port 8000
```

### With custom configuration:
```bash
TRAIN_PLY_MAX_CONCURRENT_JOBS=2 CUDA_VISIBLE_DEVICES=0 uvicorn server.train_ply.main:app --host 0.0.0.0 --port 8000
```

The server will start on `http://localhost:8000`

## API Documentation

Once the server is running, visit:
- **Interactive API docs**: http://localhost:8000/docs
- **Alternative docs**: http://localhost:8000/redoc

## API Endpoints

### POST /train

Start a new training job.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Parameters:
  - `ply_file`: PLY file (required)
  - `camera_json`: Camera JSON file (required)
  - `images`: Image files (required, multiple files)
  - `config_json`: Training configuration as JSON string (optional)

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/train" \
  -F "ply_file=@path/to/model.ply" \
  -F "camera_json=@path/to/cameras.json" \
  -F "images=@path/to/image1.jpg" \
  -F "images=@path/to/image2.jpg" \
  -F "images=@path/to/image3.jpg"
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "Training job created successfully"
}
```

### GET /train/{job_id}/status

Get the status of a training job.

**Request:**
- Method: `GET`
- Path: `/train/{job_id}/status`

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running",
  "created_at": "2024-01-15T14:30:45.123456",
  "updated_at": "2024-01-15T14:35:12.789012",
  "current_step": 15000,
  "max_steps": 30000,
  "progress": 0.5,
  "result_dir": "results/train_ply/550e8400-e29b-41d4-a716-446655440000",
  "ply_files": ["results/train_ply/.../ply/point_cloud_7000.ply"],
  "checkpoint_files": ["results/train_ply/.../ckpts/ckpt_7000_rank0.pt"],
  "error_message": null,
  "error_traceback": null,
  "config": { ... }
}
```

**Job Status Values:**
- `uploading`: Job created, files being uploaded (incremental upload mode)
- `ready`: All required files uploaded, ready to start training (incremental upload mode)
- `pending`: Job created but not started
- `running`: Training in progress
- `completed`: Training completed successfully
- `failed`: Training failed with error
- `cancelled`: Job was cancelled

### GET /train/{job_id}/results

Download training results.

**Request:**
- Method: `GET`
- Path: `/train/{job_id}/results`
- Query parameters:
  - `file_type` (optional): Filter by type (`ply`, `checkpoint`, `render`, `stats`, or omit for all)

**Response:**
- Returns a zip file containing results
- If `file_type` is specified, returns filtered results

**Example:**
```bash
# Download all results
curl -O "http://localhost:8000/train/{job_id}/results"

# Download only PLY files
curl -O "http://localhost:8000/train/{job_id}/results?file_type=ply"

# Download only checkpoints
curl -O "http://localhost:8000/train/{job_id}/results?file_type=checkpoint"
```

### GET /train/{job_id}/logs

Get training logs for a job.

**Request:**
- Method: `GET`
- Path: `/train/{job_id}/logs`

**Response:**
- Returns plain text log content

**Example:**
```bash
curl "http://localhost:8000/train/{job_id}/logs" > training.log
```

### DELETE /train/{job_id}

Cancel a running training job.

**Request:**
- Method: `DELETE`
- Path: `/train/{job_id}`

**Response:**
```json
{
  "message": "Job {job_id} cancelled successfully"
}
```

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "active_jobs": 2,
  "total_jobs": 10
}
```

## Incremental Upload API

The API supports incremental uploads, allowing you to upload files one-by-one before starting training. This is useful for large files or when files are generated dynamically.

### POST /train/create

Create a new upload job for incremental file uploads.

**Request:**
- Method: `POST`
- Path: `/train/create`

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "uploading",
  "message": "Upload job created successfully"
}
```

### POST /train/{job_id}/ply

Upload PLY file to a job.

**Request:**
- Method: `POST`
- Path: `/train/{job_id}/ply`
- Content-Type: `multipart/form-data`
- Parameters:
  - `ply_file`: PLY file (required)

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "uploading",
  "ply_uploaded": true
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/train/{job_id}/ply" \
  -F "ply_file=@path/to/model.ply"
```

### POST /train/{job_id}/image

Upload a single image file to a job.

**Request:**
- Method: `POST`
- Path: `/train/{job_id}/image`
- Content-Type: `multipart/form-data`
- Parameters:
  - `image`: Image file (required)

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "uploading",
  "images_uploaded": ["image1.jpg", "image2.jpg"]
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/train/{job_id}/image" \
  -F "image=@path/to/image1.jpg"

curl -X POST "http://localhost:8000/train/{job_id}/image" \
  -F "image=@path/to/image2.jpg"
```

**Note:** Duplicate image uploads will return a 409 Conflict error.

### POST /train/{job_id}/cameras

Upload camera.json file to a job.

**Request:**
- Method: `POST`
- Path: `/train/{job_id}/cameras`
- Content-Type: `multipart/form-data`
- Parameters:
  - `cameras_json`: Camera JSON file (required)

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "uploading",
  "cameras_uploaded": true
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/train/{job_id}/cameras" \
  -F "cameras_json=@path/to/cameras.json"
```

**Note:** Validation is not performed automatically. Use the `/validate` endpoint to check if the job is ready after uploading all files.

### POST /train/{job_id}/config

Upload training configuration to a job.

**Request:**
- Method: `POST`
- Path: `/train/{job_id}/config`
- Content-Type: `application/x-www-form-urlencoded`
- Parameters:
  - `config_json`: Training configuration as JSON string (required)

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "uploading",
  "config_uploaded": true
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/train/{job_id}/config" \
  -F "config_json={\"max_steps\": 30000, \"batch_size\": 1}"
```

### GET /train/{job_id}/upload-status

Get upload status for a job. Returns the current status without running validation.

**Request:**
- Method: `GET`
- Path: `/train/{job_id}/upload-status`

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "uploading",
  "ply_uploaded": true,
  "cameras_uploaded": true,
  "images_uploaded": ["image1.jpg", "image2.jpg"],
  "config_uploaded": false,
  "validation_errors": [],
  "is_ready": false
}
```

**Example:**
```bash
curl "http://localhost:8000/train/{job_id}/upload-status"
```

**Note:** This endpoint does NOT run validation automatically. The `is_ready` field reflects the current job status (true if status is "ready", false otherwise). Use the `/validate` endpoint to manually validate the job.

### POST /train/{job_id}/validate

Manually validate a job to check if it's ready to start training. This endpoint validates that all required files are present and match correctly.

**Request:**
- Method: `POST`
- Path: `/train/{job_id}/validate`

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "ready",
  "is_ready": true,
  "validation_errors": []
}
```

**Error Response (if validation fails):**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "uploading",
  "is_ready": false,
  "validation_errors": [
    "Missing images referenced in camera JSON: {'image3.jpg'}"
  ]
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/train/{job_id}/validate"
```

**Note:** 
- Only jobs in `uploading` or `ready` state can be validated
- If validation passes, the job status changes to `ready`
- If validation fails, the job status remains `uploading` and errors are returned
- You can call this endpoint multiple times to re-validate after uploading additional files

### POST /train/{job_id}/start

Start training from an uploaded job. Validates that all required files are present before starting.

**Request:**
- Method: `POST`
- Path: `/train/{job_id}/start`

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "Training started successfully"
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/train/{job_id}/start"
```

**Error Response (if not ready):**
```json
{
  "detail": "Job {job_id} is not ready to start. Errors: ['Missing images referenced in camera JSON: {'image3.jpg'}']"
}
```

### DELETE /train/{job_id}/upload

Cancel an incomplete upload job and delete uploaded files.

**Request:**
- Method: `DELETE`
- Path: `/train/{job_id}/upload`

**Response:**
```json
{
  "message": "Upload {job_id} cancelled successfully"
}
```

**Example:**
```bash
curl -X DELETE "http://localhost:8000/train/{job_id}/upload"
```

**Note:** Only jobs in `uploading` or `ready` status can be cancelled this way.

## Incremental Upload Workflow

Here's a complete example of using the incremental upload API:

```python
import requests
import json
import time

BASE_URL = "http://localhost:8000"

# Step 1: Create upload job
response = requests.post(f"{BASE_URL}/train/create")
job_data = response.json()
job_id = job_data["job_id"]
print(f"Created job: {job_id}")

# Step 2: Upload PLY file
with open("path/to/model.ply", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/train/{job_id}/ply",
        files={"ply_file": f}
    )
print("PLY uploaded")

# Step 3: Upload images one by one
for img_path in ["image1.jpg", "image2.jpg", "image3.jpg"]:
    with open(f"path/to/{img_path}", "rb") as f:
        response = requests.post(
            f"{BASE_URL}/train/{job_id}/image",
            files={"image": (img_path, f)}
        )
    print(f"Uploaded {img_path}")

# Step 4: Upload camera JSON
with open("path/to/cameras.json", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/train/{job_id}/cameras",
        files={"cameras_json": f}
    )
print("Cameras uploaded")

# Step 5: Manually validate the job
response = requests.post(f"{BASE_URL}/train/{job_id}/validate")
validation = response.json()
print(f"Validation: Ready={validation['is_ready']}, Status={validation['status']}")
if validation['validation_errors']:
    print(f"Validation errors: {validation['validation_errors']}")
    # If there are errors, you may need to upload more images or fix the camera JSON
    # Then call validate again

# Step 6: Check upload status (optional, for checking current state)
response = requests.get(f"{BASE_URL}/train/{job_id}/upload-status")
status = response.json()
print(f"Current status: {status['status']}")

# Step 7: Upload config (optional)
config = {
    "max_steps": 30000,
    "batch_size": 1,
    "sh_degree": 3,
}
response = requests.post(
    f"{BASE_URL}/train/{job_id}/config",
    data={"config_json": json.dumps(config)}
)
print("Config uploaded")

# Step 8: Start training (validates one more time before starting)
response = requests.post(f"{BASE_URL}/train/{job_id}/start")
print("Training started!")

# Step 9: Monitor training (same as bulk upload)
while True:
    response = requests.get(f"{BASE_URL}/train/{job_id}/status")
    status = response.json()
    
    print(f"Status: {status['status']}, Progress: {status.get('progress', 0):.2%}")
    
    if status["status"] == "completed":
        break
    elif status["status"] == "failed":
        print(f"Failed: {status.get('error_message')}")
        break
    
    time.sleep(5)
```

## Choosing Between Bulk and Incremental Upload

**Use Bulk Upload (`POST /train`) when:**
- All files are available at once
- Files are small to medium size
- You want a simpler workflow

**Use Incremental Upload (`POST /train/create` + individual uploads) when:**
- Files are large and may timeout
- Files are generated dynamically
- You want to upload files one-by-one
- You want to validate files before starting training
- You want more control over the upload process

## Training Configuration

Training parameters can be provided via the `config_json` form field. All parameters are optional and will use defaults if not specified.

**Example configuration:**
```json
{
  "max_steps": 30000,
  "batch_size": 1,
  "sh_degree": 3,
  "ssim_lambda": 0.2,
  "means_lr": 1.6e-4,
  "scales_lr": 5e-3,
  "opacities_lr": 5e-2,
  "quats_lr": 1e-3,
  "sh0_lr": 2.5e-3,
  "test_every": 8,
  "save_ply": true,
  "strategy": "default"
}
```

**Available Parameters:**

- `max_steps`: Number of training steps (default: 30000)
- `batch_size`: Batch size (default: 1)
- `sh_degree`: Spherical harmonics degree (default: 3)
- `ssim_lambda`: Weight for SSIM loss (default: 0.2)
- `means_lr`: Learning rate for positions (default: 1.6e-4)
- `scales_lr`: Learning rate for scales (default: 5e-3)
- `opacities_lr`: Learning rate for opacities (default: 5e-2)
- `quats_lr`: Learning rate for quaternions (default: 1e-3)
- `sh0_lr`: Learning rate for SH DC (default: 2.5e-3)
- `shN_lr`: Learning rate for SH rest (default: sh0_lr / 20)
- `test_every`: Every N images is a test image (default: 8)
- `save_ply`: Save PLY files during training (default: true)
- `strategy`: Densification strategy - `"default"` or `"mcmc"` (default: "default")
- `normalize_world_space`: Normalize scene to unit sphere (default: false)
- `camera_model`: Camera model type - `"pinhole"`, `"ortho"`, or `"fisheye"` (default: "pinhole")

See `models.py` for the complete list of available parameters.

## Data Format

### PLY File Format

The PLY file should be in binary little-endian format with the following properties:
- `x, y, z` - Gaussian positions (means)
- `f_dc_0, f_dc_1, f_dc_2` - Spherical harmonics DC coefficients
- `f_rest_*` - Higher-order spherical harmonics coefficients
- `opacity` - Gaussian opacity (in sigmoid space)
- `scale_0, scale_1, scale_2` - Gaussian scales (in log space)
- `rot_0, rot_1, rot_2, rot_3` - Gaussian quaternions

### Camera JSON Format

Create a JSON file with camera data in the following format:

```json
{
  "image_000.jpg": {
    "camtoworld": [
      [r11, r12, r13, tx],
      [r21, r22, r23, ty],
      [r31, r32, r33, tz],
      [0,   0,   0,   1 ]
    ],
    "K": [
      [fx,  0, cx],
      [0,  fy, cy],
      [0,   0,  1]
    ],
    "width": 1920,
    "height": 1080
  },
  "image_001.jpg": {
    "camtoworld": [...],
    "K": [...],
    "width": 1920,
    "height": 1080
  }
}
```

**Alternative field names:**
- `camtoworld` can also be `c2w`
- `K` can also be `intrinsic`
- `width` and `height` are optional if images can be loaded to determine size

## Example Python Client

```python
import requests
import json
import time

# Server URL
BASE_URL = "http://localhost:8000"

# Prepare files
files = {
    "ply_file": open("path/to/model.ply", "rb"),
    "camera_json": open("path/to/cameras.json", "rb"),
}

# Add images
image_files = [
    ("images", open("path/to/image1.jpg", "rb")),
    ("images", open("path/to/image2.jpg", "rb")),
    ("images", open("path/to/image3.jpg", "rb")),
]

# Optional training configuration
config = {
    "max_steps": 30000,
    "batch_size": 1,
    "sh_degree": 3,
    "strategy": "default",
}
files["config_json"] = json.dumps(config)

# Start training job
response = requests.post(f"{BASE_URL}/train", files=files)
response.raise_for_status()
job_data = response.json()
job_id = job_data["job_id"]
print(f"Job created: {job_id}")

# Poll for status
while True:
    response = requests.get(f"{BASE_URL}/train/{job_id}/status")
    response.raise_for_status()
    status = response.json()
    
    print(f"Status: {status['status']}, Progress: {status.get('progress', 0):.2%}")
    
    if status["status"] == "completed":
        print("Training completed!")
        break
    elif status["status"] == "failed":
        print(f"Training failed: {status.get('error_message')}")
        break
    
    time.sleep(5)

# Download results
response = requests.get(f"{BASE_URL}/train/{job_id}/results")
with open(f"{job_id}_results.zip", "wb") as f:
    f.write(response.content)
print("Results downloaded!")

# Download logs
response = requests.get(f"{BASE_URL}/train/{job_id}/logs")
with open(f"{job_id}_logs.txt", "w") as f:
    f.write(response.text)
print("Logs downloaded!")
```

## Error Handling

The API returns appropriate HTTP status codes:

- `200`: Success
- `400`: Bad Request (invalid file format, missing files, invalid configuration)
- `404`: Not Found (job not found, results not found)
- `500`: Internal Server Error (training errors, server errors)
- `503`: Service Unavailable (maximum concurrent jobs reached)

Error responses include a `detail` field with error information:

```json
{
  "detail": "Invalid camera JSON: Expecting value: line 1 column 1 (char 0)"
}
```

## Troubleshooting

1. **Job creation fails**: 
   - Check that all required files (PLY, camera JSON, images) are provided
   - Verify camera JSON format is valid
   - Ensure image filenames match camera JSON keys

2. **Training fails**:
   - Check job logs: `GET /train/{job_id}/logs`
   - Verify PLY file format is correct
   - Check GPU memory availability
   - Review error message in job status

3. **CUDA errors**:
   - Check GPU availability: `nvidia-smi`
   - Verify CUDA installation
   - Set `CUDA_VISIBLE_DEVICES` environment variable

4. **Memory errors**:
   - Reduce batch size in training config
   - Reduce number of training steps
   - Use `packed: true` in training config

5. **Import errors**:
   - Ensure gsplat is installed in the same Python environment
   - Check that all dependencies from `examples/requirements.txt` are installed

6. **Job stuck in pending**:
   - Check `TRAIN_PLY_MAX_CONCURRENT_JOBS` limit
   - Verify server logs for errors
   - Check GPU availability

## Output Structure

Training results are saved in the following structure:

```
result_dir/
├── ckpts/              # Model checkpoints (.pt files)
├── ply/                 # Exported PLY files
├── renders/             # Rendered validation images
├── stats/               # Training statistics (JSON)
├── tb/                  # TensorBoard logs
└── cfg.yml              # Training configuration
```

## Notes

- The model is trained asynchronously - jobs are queued and executed in the background
- Training can take a long time depending on the number of steps and data size
- GPU is required for training
- Job data is stored in `server/train_ply/jobs/` directory
- Results are stored in `results/train_ply/` directory (or custom `TRAIN_PLY_RESULT_BASE_DIR`)
- Old jobs are automatically cleaned up after the retention period
- Maximum concurrent jobs can be configured to prevent GPU memory issues

## License

This code follows the same license as the gsplat project.
