# PLY-based Gaussian Splatting Training

This module provides training capabilities for refining Gaussian Splatting models from PLY files using custom camera data. It includes both a command-line interface and a REST API service.

## Overview

The training system allows you to:
- Load an existing Gaussian Splatting model from a PLY file
- Train/refine the model using images with known camera poses and intrinsics
- Export refined models back to PLY format
- Use either CLI tools or REST API for programmatic access

## Prerequisites

- Python 3.8+
- PyTorch with CUDA support
- gsplat library installed
- Required dependencies from `examples/requirements.txt`
- CUDA-capable GPU for training

## REST API Service

A FastAPI-based REST API server for training Gaussian Splatting models from PLY files using custom camera data.

### Features

- **PLY-based training**: Submit PLY files with camera data and images to train Gaussian Splatting models
- **Async job management**: Long-running training jobs are executed asynchronously with status tracking
- **FastAPI framework**: Modern, fast, with automatic API documentation
- **Request validation**: Automatic validation of inputs using Pydantic
- **Error handling**: Comprehensive error handling with clear error messages
- **Progress tracking**: Real-time job status and progress updates
- **Result download**: Download trained models, checkpoints, renders, and statistics

### Installation

1. Install the server dependencies:
```bash
pip install -r server/train_ply/requirements.txt
```

2. Ensure gsplat and its dependencies are installed (as per main repository instructions)

3. Ensure you have CUDA-capable GPU for training

### Configuration

The server can be configured using environment variables:

- `TRAIN_PLY_JOBS_DIR`: Directory for job data (default: `server/train_ply/jobs/`)
- `TRAIN_PLY_RESULT_BASE_DIR`: Base directory for results (default: `results/train_ply/`)
- `TRAIN_PLY_MAX_CONCURRENT_JOBS`: Maximum parallel jobs (default: `1`)
- `TRAIN_PLY_JOB_RETENTION_DAYS`: Days to keep completed jobs (default: `7`)
- `CUDA_VISIBLE_DEVICES`: GPU selection (e.g., `0` or `0,1,2,3`)
- `PORT`: Server port (default: `8000`)
- `HOST`: Server host (default: `0.0.0.0`)

### Running the Server

**Basic usage:**
```bash
python -m server.train_ply.main
```

**Or using uvicorn directly:**
```bash
uvicorn server.train_ply.main:app --host 0.0.0.0 --port 8000
```

**With custom configuration:**
```bash
TRAIN_PLY_MAX_CONCURRENT_JOBS=2 CUDA_VISIBLE_DEVICES=0 uvicorn server.train_ply.main:app --host 0.0.0.0 --port 8000
```

The server will start on `http://localhost:8000`

### API Documentation

Once the server is running, visit:
- **Interactive API docs**: http://localhost:8000/docs
- **Alternative docs**: http://localhost:8000/redoc

### API Endpoints

#### POST `/jobs`

Create a new training job.

**Request:**
- Method: `POST`
- Body: None

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "uploading",
  "message": "Job created successfully"
}
```

**Status Codes:**
- `201 Created` - Job created successfully
- `503 Service Unavailable` - Maximum concurrent jobs reached

**Notes:**
- Creates a new job in `UPLOADING` status
- Returns a unique `job_id` for subsequent operations

---

#### POST `/jobs/{job_id}/ply`

Upload PLY file to a job.

**Request:**
- Method: `POST`
- Path Parameters:
  - `job_id` (string) - Job identifier
- Body: `multipart/form-data`
  - `ply_file` (file, required) - PLY file

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "uploading",
  "ply_uploaded": true
}
```

**Status Codes:**
- `200 OK` - PLY file uploaded successfully
- `404 Not Found` - Job not found
- `400 Bad Request` - Job is not in UPLOADING state

**Example:**
```bash
curl -X POST "http://localhost:8000/jobs/{job_id}/ply" \
  -F "ply_file=@path/to/model.ply"
```

---

#### POST `/jobs/{job_id}/images`

Upload one or more image files to a job.

**Request:**
- Method: `POST`
- Path Parameters:
  - `job_id` (string) - Job identifier
- Body: `multipart/form-data`
  - `images` (file[], required) - One or more image files

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "uploading",
  "images_uploaded": ["image1.jpg", "image2.jpg", ...]
}
```

**Status Codes:**
- `200 OK` - Image(s) uploaded successfully
- `404 Not Found` - Job not found
- `400 Bad Request` - Job is not in UPLOADING state or filename missing
- `409 Conflict` - Image already uploaded (duplicate filename)

**Notes:**
- Can be called multiple times to upload additional images
- Duplicate filenames are rejected
- Image filenames must match keys in the camera files

**Example:**
```bash
curl -X POST "http://localhost:8000/jobs/{job_id}/images" \
  -F "images=@path/to/image1.jpg" \
  -F "images=@path/to/image2.jpg"
```

---

#### POST `/jobs/{job_id}/cameras`

Upload one or more camera files to a job.

**Request:**
- Method: `POST`
- Path Parameters:
  - `job_id` (string) - Job identifier
- Body: `multipart/form-data`
  - `cameras` (file[], required) - One or more camera files (.cam.json files)

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "uploading",
  "cameras_uploaded": ["DSC00153.cam.json", "DSC00154.cam.json", ...]
}
```

**Status Codes:**
- `200 OK` - Camera file(s) uploaded successfully
- `404 Not Found` - Job not found
- `400 Bad Request` - Job is not in UPLOADING state, invalid JSON format, or all uploads failed

**Notes:**
- Can be called multiple times to upload additional camera files
- Duplicate filenames are rejected
- Validates JSON format on upload for each file
- Each `.cam.json` file should contain a single camera entry: `{"IMAGE_NAME.JPG": {...}}`
- Camera file keys should match uploaded image filenames

**Example:**
```bash
curl -X POST "http://localhost:8000/jobs/{job_id}/cameras" \
  -F "cameras=@path/to/DSC00153.cam.json" \
  -F "cameras=@path/to/DSC00154.cam.json"
```

---

#### POST `/jobs/{job_id}/config`

Upload training configuration to a job.

**Request:**
- Method: `POST`
- Path Parameters:
  - `job_id` (string) - Job identifier
- Body: `application/x-www-form-urlencoded` or `multipart/form-data`
  - `config_json` (string, required) - Training configuration as JSON string

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "uploading",
  "config_uploaded": true
}
```

**Status Codes:**
- `200 OK` - Config uploaded successfully
- `404 Not Found` - Job not found
- `400 Bad Request` - Job is not in UPLOADING state or invalid config JSON

**Notes:**
- Config is optional (defaults will be used if not provided)
- See Training Configuration section for available parameters

**Example:**
```bash
curl -X POST "http://localhost:8000/jobs/{job_id}/config" \
  -F "config_json={\"max_steps\": 30000, \"batch_size\": 1}"
```

---

#### GET `/jobs/{job_id}`

Get comprehensive job status and information.

**Request:**
- Method: `GET`
- Path Parameters:
  - `job_id` (string) - Job identifier

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "ready",
  "created_at": "2026-01-25T10:00:00",
  "updated_at": "2026-01-25T10:05:00",
  "current_step": null,
  "max_steps": null,
  "progress": null,
  "result_dir": null,
  "ply_files": [],
  "checkpoint_files": [],
  "error_message": null,
  "error_traceback": null,
  "config": {
    "max_steps": 30000,
    "batch_size": 1,
    ...
  },
  "ply_uploaded": true,
  "cameras_uploaded": ["DSC00153.cam.json", "DSC00154.cam.json"],
  "images_uploaded": ["image1.jpg", "image2.jpg"],
  "config_uploaded": true,
  "validation_errors": []
}
```

**Status Codes:**
- `200 OK` - Job found
- `404 Not Found` - Job not found

**Notes:**
- Returns complete job information including upload status
- Automatically validates job if status is `UPLOADING` or `READY`
- `validation_errors` array contains any validation issues
- `status` can be: `uploading`, `ready`, `pending`, `running`, `completed`, `failed`, `cancelled`

**Job Status Values:**
- `uploading`: Job created, files being uploaded
- `ready`: All required files uploaded, ready to start training
- `pending`: Job created but not started
- `running`: Training in progress
- `completed`: Training completed successfully
- `failed`: Training failed with error
- `cancelled`: Job was cancelled

---

#### POST `/jobs/{job_id}/start`

Start training for a job.

**Request:**
- Method: `POST`
- Path Parameters:
  - `job_id` (string) - Job identifier

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "pending",
  "message": "Training started successfully"
}
```

**Status Codes:**
- `200 OK` - Training started successfully
- `404 Not Found` - Job not found
- `400 Bad Request` - Job is not ready to start (validation errors)
- `503 Service Unavailable` - Maximum concurrent jobs reached

**Notes:**
- Automatically validates job before starting
- Changes job status from `READY` to `PENDING`
- Training begins asynchronously
- Returns immediately with `PENDING` status

**Example:**
```bash
curl -X POST "http://localhost:8000/jobs/{job_id}/start"
```

---

#### GET `/jobs/{job_id}/results`

Download training results.

**Request:**
- Method: `GET`
- Path Parameters:
  - `job_id` (string) - Job identifier
- Query Parameters:
  - `file_type` (string, optional) - Filter results by type:
    - `ply` - PLY files only
    - `checkpoint` - Checkpoint files only
    - `render` - Render images only
    - `stats` - Statistics JSON files only
    - `all` or omitted - All results (default)

**Response:**
- Content-Type: `application/zip` or `application/octet-stream`
- Body: Zip file or individual file

**Status Codes:**
- `200 OK` - Results returned
- `404 Not Found` - Job not found or results not available
- `400 Bad Request` - Job is not completed

**Notes:**
- Returns zip file by default
- If `file_type` is specified and only one file matches, returns that file directly
- If multiple files match, returns a zip file

**Example:**
```bash
# Download all results
curl -O "http://localhost:8000/jobs/{job_id}/results"

# Download only PLY files
curl -O "http://localhost:8000/jobs/{job_id}/results?file_type=ply"

# Download only checkpoints
curl -O "http://localhost:8000/jobs/{job_id}/results?file_type=checkpoint"
```

---

#### GET `/jobs/{job_id}/logs`

Get training logs for a job.

**Request:**
- Method: `GET`
- Path Parameters:
  - `job_id` (string) - Job identifier

**Response:**
- Content-Type: `text/plain`
- Body: Log content (streaming)

**Status Codes:**
- `200 OK` - Logs returned
- `404 Not Found` - Job not found or log file not found

**Notes:**
- Returns logs as streaming text
- Falls back to error.log if training.log doesn't exist

**Example:**
```bash
curl "http://localhost:8000/jobs/{job_id}/logs" > training.log
```

---

#### DELETE `/jobs/{job_id}`

Cancel or delete a job.

**Request:**
- Method: `DELETE`
- Path Parameters:
  - `job_id` (string) - Job identifier

**Response:**
```json
{
  "message": "Job {job_id} cancelled successfully"
}
```

**Status Codes:**
- `200 OK` - Job cancelled/deleted successfully
- `404 Not Found` - Job not found
- `400 Bad Request` - Job cannot be cancelled (already completed/failed)
- `500 Internal Server Error` - Failed to cancel job

**Notes:**
- For `PENDING` or `RUNNING` jobs: cancels the training task
- For `UPLOADING` or `READY` jobs: deletes the job and cleans up files
- Completed or failed jobs cannot be cancelled

**Example:**
```bash
curl -X DELETE "http://localhost:8000/jobs/{job_id}"
```

---

#### GET `/health`

Health check endpoint.

**Request:**
- Method: `GET`

**Response:**
```json
{
  "status": "healthy",
  "active_jobs": 2,
  "total_jobs": 10
}
```

**Status Codes:**
- `200 OK` - Service is healthy

**Notes:**
- `active_jobs` counts jobs with status `PENDING` or `RUNNING`
- `total_jobs` counts all jobs

---

### API Workflow

#### Typical Training Flow

1. **Create Job**
   ```
   POST /jobs
   → Returns job_id
   ```

2. **Upload Files** (can be done in any order, multiple times)
   ```
   POST /jobs/{job_id}/ply
   POST /jobs/{job_id}/images
   POST /jobs/{job_id}/cameras  (can upload multiple .cam.json files)
   POST /jobs/{job_id}/config  (optional)
   ```

3. **Check Status** (auto-validates)
   ```
   GET /jobs/{job_id}
   → Returns status, upload progress, validation_errors
   ```

4. **Start Training**
   ```
   POST /jobs/{job_id}/start
   → Validates and starts training
   ```

5. **Monitor Progress**
   ```
   GET /jobs/{job_id}
   → Returns current_step, progress, status
   GET /jobs/{job_id}/logs
   → Streams training logs
   ```

6. **Download Results** (when status is "completed")
   ```
   GET /jobs/{job_id}/results?file_type=ply
   GET /jobs/{job_id}/results?file_type=checkpoint
   GET /jobs/{job_id}/results  (all results)
   ```

### Example Python Client

```python
import requests
import json
import time

BASE_URL = "http://localhost:8000"

# Step 1: Create upload job
response = requests.post(f"{BASE_URL}/jobs")
job_data = response.json()
job_id = job_data["job_id"]
print(f"Created job: {job_id}")

# Step 2: Upload PLY file
with open("path/to/model.ply", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/jobs/{job_id}/ply",
        files={"ply_file": f}
    )
print("PLY uploaded")

# Step 3: Upload images
image_files = [
    ("images", open("path/to/image1.jpg", "rb")),
    ("images", open("path/to/image2.jpg", "rb")),
    ("images", open("path/to/image3.jpg", "rb")),
]
response = requests.post(
    f"{BASE_URL}/jobs/{job_id}/images",
    files=image_files
)
print("Images uploaded")

# Step 4: Upload camera files
camera_files = [
    ("cameras", open("path/to/DSC00153.cam.json", "rb")),
    ("cameras", open("path/to/DSC00154.cam.json", "rb")),
]
response = requests.post(
    f"{BASE_URL}/jobs/{job_id}/cameras",
    files=camera_files
)
print("Cameras uploaded")

# Step 5: Check status (auto-validates)
response = requests.get(f"{BASE_URL}/jobs/{job_id}")
status = response.json()
print(f"Status: {status['status']}")
if status['validation_errors']:
    print(f"Validation errors: {status['validation_errors']}")
    # Fix errors and re-check status

# Step 6: Upload config (optional)
config = {
    "max_steps": 30000,
    "batch_size": 1,
    "sh_degree": 3,
}
response = requests.post(
    f"{BASE_URL}/jobs/{job_id}/config",
    data={"config_json": json.dumps(config)}
)
print("Config uploaded")

# Step 7: Start training (validates one more time before starting)
response = requests.post(f"{BASE_URL}/jobs/{job_id}/start")
print("Training started!")

# Step 8: Monitor training
while True:
    response = requests.get(f"{BASE_URL}/jobs/{job_id}")
    status = response.json()
    
    print(f"Status: {status['status']}, Progress: {status.get('progress', 0):.2%}")
    
    if status["status"] == "completed":
        break
    elif status["status"] == "failed":
        print(f"Failed: {status.get('error_message')}")
        break
    
    time.sleep(5)

# Step 9: Download results
response = requests.get(f"{BASE_URL}/jobs/{job_id}/results")
with open(f"{job_id}_results.zip", "wb") as f:
    f.write(response.content)
print("Results downloaded!")

# Step 10: Download logs
response = requests.get(f"{BASE_URL}/jobs/{job_id}/logs")
with open(f"{job_id}_logs.txt", "w") as f:
    f.write(response.text)
print("Logs downloaded!")
```

### Error Handling

The API returns appropriate HTTP status codes:

- `200`: Success
- `201`: Created (job created)
- `400`: Bad Request (invalid file format, missing files, invalid configuration, job not in correct state)
- `404`: Not Found (job not found, results not found)
- `409`: Conflict (duplicate filename)
- `500`: Internal Server Error (training errors, server errors)
- `503`: Service Unavailable (maximum concurrent jobs reached)

Error responses include a `detail` field with error information:

```json
{
  "detail": "Invalid camera JSON: Expecting value: line 1 column 1 (char 0)"
}
```

### Troubleshooting (API)

1. **Job creation fails**: 
   - Check that maximum concurrent jobs limit hasn't been reached
   - Verify server logs for errors

2. **File upload fails**:
   - Ensure job is in `UPLOADING` status
   - Check that all required files (PLY, camera files, images) are provided
   - Verify camera JSON format is valid
   - Ensure image filenames match camera file keys

3. **Training fails**:
   - Check job logs: `GET /jobs/{job_id}/logs`
   - Verify PLY file format is correct
   - Check GPU memory availability
   - Review error message in job status

4. **CUDA errors**:
   - Check GPU availability: `nvidia-smi`
   - Verify CUDA installation
   - Set `CUDA_VISIBLE_DEVICES` environment variable

5. **Memory errors**:
   - Reduce batch size in training config
   - Reduce number of training steps
   - Check `TRAIN_PLY_MAX_CONCURRENT_JOBS` limit

6. **Job stuck in pending**:
   - Check `TRAIN_PLY_MAX_CONCURRENT_JOBS` limit
   - Verify server logs for errors
   - Check GPU availability

---

## CLI Usage

The `train_from_ply.py` script provides a command-line interface for training Gaussian Splatting models from PLY files.

### File Structure

```
server/
├── __init__.py           # Package initialization
├── ply_loader.py         # PLY file parsing utility
├── camera_parser.py      # JSON camera data parser
├── train_from_ply.py     # Main training script
└── train_ply/            # REST API service for training
    ├── main.py           # FastAPI application
    ├── job_manager.py    # Job state management
    ├── models.py         # Pydantic models
    └── training_worker.py # Background training worker
```

### Basic Training

```bash
python server/train_from_ply.py \
    --ply_path path/to/model.ply \
    --camera_dir path/to/cameras \
    --data_dir path/to/your_data \
    --result_dir results/my_training
```

### Configuration Options

**Data Paths:**
- `--ply_path`: Path to input PLY file (required)
- `--camera_dir`: Directory containing `.cam.json` files with camera data (required)
- `--data_dir`: Directory containing images folder (required)
- `--result_dir`: Directory to save results (default: `results/ply_training`)

**Training Parameters:**
- `--max_steps`: Number of training steps (default: 30000)
- `--batch_size`: Batch size (default: 1)
- `--sh_degree`: Spherical harmonics degree (default: 3)
- `--ssim_lambda`: Weight for SSIM loss (default: 0.2)
- `--test_every`: Every N images is a test image (default: 8)

**Learning Rates:**
- `--means_lr`: Learning rate for positions (default: 1.6e-4)
- `--scales_lr`: Learning rate for scales (default: 5e-3)
- `--opacities_lr`: Learning rate for opacities (default: 5e-2)
- `--quats_lr`: Learning rate for quaternions (default: 1e-3)
- `--sh0_lr`: Learning rate for SH DC (default: 2.5e-3)
- `--shN_lr`: Learning rate for SH rest (default: 2.5e-3 / 20)

**Densification Strategy:**
- `default`: Use default densification strategy
- `mcmc`: Use MCMC densification strategy

**Other Options:**
- `--save_ply`: Save PLY files during training (default: True)
- `--ply_steps`: Steps at which to save PLY files (default: [7000, 30000])
- `--eval_steps`: Steps at which to evaluate (default: [7000, 30000])
- `--save_steps`: Steps at which to save checkpoints (default: [7000, 30000])
- `--normalize_world_space`: Normalize scene to unit sphere (default: False)
- `--patch_size`: Random crop size for training (optional)

### Distributed Training

For multi-GPU training:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python server/train_from_ply.py \
    --ply_path path/to/model.ply \
    --camera_dir path/to/cameras \
    --data_dir path/to/your_data \
    --steps_scaler 0.25  # Reduce steps by 4x since batch size is 4x
```

### Evaluation Only

To evaluate a trained checkpoint:

```bash
python server/train_from_ply.py \
    --ply_path path/to/model.ply \
    --camera_dir path/to/cameras \
    --data_dir path/to/your_data \
    --ckpt results/my_training/ckpts/ckpt_30000_rank0.pt
```

### Examples

**Example 1: Basic Training**
```bash
python server/train_from_ply.py \
    --ply_path data/point_cloud.ply \
    --camera_dir data/cameras \
    --data_dir data \
    --result_dir results/basic_training \
    --max_steps 30000 \
    --save_ply True
```

**Example 2: Quick Fine-tuning**
```bash
python server/train_from_ply.py \
    --ply_path data/point_cloud.ply \
    --camera_dir data/cameras \
    --data_dir data \
    --result_dir results/finetune \
    --max_steps 5000 \
    --means_lr 1e-5 \
    --sh0_lr 1e-4
```

**Example 3: MCMC Strategy**
```bash
python server/train_from_ply.py mcmc \
    --ply_path data/point_cloud.ply \
    --camera_dir data/cameras \
    --data_dir data \
    --result_dir results/mcmc_training
```

### API Reference

#### `load_ply(ply_path: str) -> Dict[str, torch.Tensor]`

Loads a PLY file and returns a dictionary with Gaussian parameters.

**Returns:**
- `means`: (N, 3) - Gaussian positions
- `scales`: (N, 3) - Gaussian scales (log space)
- `quats`: (N, 4) - Gaussian quaternions
- `opacities`: (N,) - Gaussian opacities (sigmoid space)
- `sh0`: (N, 1, 3) - SH DC coefficients
- `shN`: (N, K, 3) - SH rest coefficients

#### `CameraParser`

Parser for JSON camera data.

**Parameters:**
- `data_dir`: Directory containing images
- `camera_dir`: Directory containing `.cam.json` files with camera data
- `normalize`: Whether to normalize world space (default: False)

**Properties:**
- `image_names`: List of image names
- `image_paths`: List of image file paths
- `camtoworlds`: (N, 4, 4) camera-to-world matrices
- `Ks_dict`: Dictionary of intrinsic matrices
- `scene_scale`: Scene scale factor

#### `CameraDataset`

PyTorch dataset for camera data.

**Parameters:**
- `parser`: CameraParser instance
- `split`: "train" or "val"
- `test_every`: Every N images is a test image
- `patch_size`: Optional random crop size

### Troubleshooting (CLI)

1. **Import errors**: Make sure you're running from the gsplat root directory or have the paths set correctly.

2. **PLY parsing errors**: Ensure the PLY file is in binary little-endian format with the correct properties.

3. **Camera data errors**: Verify that:
   - JSON file is valid
   - Camera matrices are 4x4 (camtoworld) and 3x3 (K)
   - Image paths are correct
   - Image dimensions match the camera data

4. **CUDA out of memory**: Reduce batch size or use `--packed` mode:
   ```bash
   --batch_size 1 --packed True
   ```

5. **Coordinate system mismatch**: Try enabling normalization:
   ```bash
   --normalize_world_space True
   ```

---

## Data Format

### PLY File Format

The PLY file should be in binary little-endian format with the following properties:
- `x, y, z` - Gaussian positions (means)
- `f_dc_0, f_dc_1, f_dc_2` - Spherical harmonics DC coefficients
- `f_rest_*` - Higher-order spherical harmonics coefficients (variable number based on SH degree)
- `opacity` - Gaussian opacity (in sigmoid space)
- `scale_0, scale_1, scale_2` - Gaussian scales (in log space)
- `rot_0, rot_1, rot_2, rot_3` - Gaussian quaternions

### Camera JSON Format

For CLI usage, create individual `.cam.json` files in a `cameras/` directory. Each file should contain a single camera entry:

**Example: `cameras/DSC00153.cam.json`**
```json
{
  "DSC00153.JPG": {
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
  }
}
```

Each `.cam.json` file contains one camera entry. The key should match the image filename.

**Alternative field names:**
- `camtoworld` can also be `c2w`
- `K` can also be `intrinsic`
- `width` and `height` are optional if images can be loaded to determine size

### Camera Files Format (REST API)

For REST API usage, upload individual `.cam.json` files, each containing a single camera entry:

```json
{
  "IMAGE_NAME.JPG": {
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
  }
}
```

### Directory Structure

**For CLI:**
```
your_data/
├── images/
│   ├── image_000.jpg
│   ├── image_001.jpg
│   └── ...
└── cameras/
    ├── DSC00153.cam.json
    ├── DSC00154.cam.json
    └── ...
```

**For REST API:**
Same structure as CLI, but camera files are uploaded individually via the API endpoint.

---

## Training Configuration

Training parameters can be provided via the `config_json` form field in the REST API or command-line arguments in CLI. All parameters are optional and will use defaults if not specified.

### Example Configuration

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

### Available Parameters

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

See `server/train_ply/models.py` for the complete list of available parameters.

---

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

---

## Notes

- The PLY file format assumes scales are stored in log space and opacities in sigmoid space (as logits)
- Camera matrices should follow OpenCV convention (camera-to-world)
- The script automatically determines SH degree from the PLY file properties
- Training uses L1 + SSIM loss by default
- Densification strategies help improve model quality during training
- The model is trained asynchronously in the REST API - jobs are queued and executed in the background
- Training can take a long time depending on the number of steps and data size
- GPU is required for training
- Job data is stored in `server/train_ply/jobs/` directory (REST API)
- Results are stored in `results/train_ply/` directory (or custom `TRAIN_PLY_RESULT_BASE_DIR`)
- Old jobs are automatically cleaned up after the retention period (REST API)
- Maximum concurrent jobs can be configured to prevent GPU memory issues (REST API)

## License

This code follows the same license as the gsplat project.
