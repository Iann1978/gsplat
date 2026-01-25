# Train PLY API Design

## Overview

This document describes the REST API for training Gaussian Splatting models from PLY files. The API follows a clear workflow: create job → upload files → start training → monitor progress → download results.

## Base URL

All endpoints are relative to the base URL of the service (e.g., `http://localhost:8000`).

## API Endpoints

### 1. Job Creation & Upload

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
- Image filenames must match keys in the cameras.json file

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
- See `TrainingConfig` model for available parameters

---

### 2. Job Management

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

---

### 3. Job Results

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

---

### 4. System

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

## Data Models

### JobStatus Enum

```typescript
enum JobStatus {
  UPLOADING = "uploading",
  READY = "ready",
  PENDING = "pending",
  RUNNING = "running",
  COMPLETED = "completed",
  FAILED = "failed",
  CANCELLED = "cancelled"
}
```

### JobInfo

Complete job information returned by `GET /jobs/{job_id}`:

```typescript
interface JobInfo {
  job_id: string;
  status: JobStatus;
  created_at: datetime;
  updated_at: datetime;
  
  // Progress information (null during upload phase)
  current_step?: number;
  max_steps?: number;
  progress?: number;  // 0.0 to 1.0
  
  // Result paths (null until completed)
  result_dir?: string;
  ply_files: string[];
  checkpoint_files: string[];
  
  // Error information
  error_message?: string;
  error_traceback?: string;
  
  // Training configuration
  config?: TrainingConfig;
  
  // Upload tracking
  ply_uploaded: boolean;
  cameras_uploaded: boolean;
  images_uploaded: string[];
  config_uploaded: boolean;
  validation_errors: string[];
}
```

### TrainingConfig

See `models.py` for complete TrainingConfig schema. Key parameters:

- `max_steps`: Number of training steps (default: 30000)
- `batch_size`: Batch size (default: 1)
- `sh_degree`: Spherical harmonics degree (default: 3)
- `ssim_lambda`: SSIM loss weight (default: 0.2)
- Learning rates for various parameters
- Regularization weights
- Output settings (save_ply, eval_steps, save_steps, etc.)

---

## Workflow

### Typical Training Flow

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

### Error Handling

- If validation fails during `GET /jobs/{job_id}`, `validation_errors` will contain details
- If validation fails during `POST /jobs/{job_id}/start`, returns 400 with error details
- Check `status` field to determine current state
- Check `error_message` and `error_traceback` if status is `FAILED`

---

## Migration from Old API

### Removed Endpoints

- ❌ `POST /train` - Use incremental upload flow instead
- ❌ `GET /train/{job_id}/upload-status` - Merged into `GET /jobs/{job_id}`
- ❌ `POST /train/{job_id}/validate` - Auto-validation in `GET /jobs/{job_id}` and `POST /jobs/{job_id}/start`
- ❌ `DELETE /train/{job_id}/upload` - Merged into `DELETE /jobs/{job_id}`

### Endpoint Mapping

| Old Endpoint | New Endpoint |
|-------------|--------------|
| `POST /train/create` | `POST /jobs` |
| `GET /train/{job_id}/status` | `GET /jobs/{job_id}` |
| `GET /train/{job_id}/upload-status` | `GET /jobs/{job_id}` (merged) |
| `POST /train/{job_id}/start` | `POST /jobs/{job_id}/start` |
| `POST /train/{job_id}/validate` | Auto-validation (no separate endpoint) |
| `POST /train/{job_id}/ply` | `POST /jobs/{job_id}/ply` |
| `POST /train/{job_id}/image` | `POST /jobs/{job_id}/images` |
| `POST /train/{job_id}/cameras` | `POST /jobs/{job_id}/cameras` |
| `POST /train/{job_id}/config` | `POST /jobs/{job_id}/config` |
| `GET /train/{job_id}/results` | `GET /jobs/{job_id}/results` |
| `GET /train/{job_id}/logs` | `GET /jobs/{job_id}/logs` |
| `DELETE /train/{job_id}` | `DELETE /jobs/{job_id}` |
| `DELETE /train/{job_id}/upload` | `DELETE /jobs/{job_id}` (merged) |
| `GET /health` | `GET /health` (unchanged) |

---

## Design Principles

1. **RESTful Resource Naming**: All endpoints use `/jobs/{job_id}` as the resource identifier
2. **Single Source of Truth**: `GET /jobs/{job_id}` returns all job information including upload status
3. **Auto-Validation**: Validation happens automatically when checking status or starting training
4. **Clear State Machine**: Job status follows a clear progression: `uploading` → `ready` → `pending` → `running` → `completed`/`failed`
5. **Idempotent Operations**: Upload endpoints can be called multiple times safely (except duplicates)
6. **Comprehensive Error Handling**: All endpoints return appropriate HTTP status codes and error messages
