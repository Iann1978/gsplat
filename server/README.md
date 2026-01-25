# PLY-based Gaussian Splatting Training

This module provides a standalone training script for refining Gaussian Splatting models from PLY files using custom camera data.

## Overview

The `train_from_ply.py` script allows you to:
- Load an existing Gaussian Splatting model from a PLY file
- Train/refine the model using images with known camera poses and intrinsics
- Export refined models back to PLY format

## File Structure

```
server/
├── __init__.py           # Package initialization
├── ply_loader.py         # PLY file parsing utility
├── camera_parser.py      # JSON camera data parser
├── train_from_ply.py     # Main training script
├── train_ply/            # REST API service for training
│   ├── main.py           # FastAPI application
│   ├── job_manager.py    # Job state management
│   ├── models.py         # Pydantic models
│   ├── training_worker.py # Background training worker
│   └── README.md         # API documentation
└── README.md             # This file
```

## Prerequisites

- Python 3.8+
- PyTorch with CUDA support
- gsplat library installed
- Required dependencies from `examples/requirements.txt`

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

### Directory Structure

```
your_data/
├── images/
│   ├── image_000.jpg
│   ├── image_001.jpg
│   └── ...
└── cameras.json
```

## Usage

### Basic Training

```bash
python server/train_from_ply.py \
    --ply_path path/to/model.ply \
    --camera_json path/to/cameras.json \
    --data_dir path/to/your_data \
    --result_dir results/my_training
```

### Configuration Options

Key configuration parameters:

**Data Paths:**
- `--ply_path`: Path to input PLY file (required)
- `--camera_json`: Path to JSON file with camera data (required)
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
    --camera_json path/to/cameras.json \
    --data_dir path/to/your_data \
    --steps_scaler 0.25  # Reduce steps by 4x since batch size is 4x
```

### Evaluation Only

To evaluate a trained checkpoint:

```bash
python server/train_from_ply.py \
    --ply_path path/to/model.ply \
    --camera_json path/to/cameras.json \
    --data_dir path/to/your_data \
    --ckpt results/my_training/ckpts/ckpt_30000_rank0.pt
```

## Output

The training script creates the following directory structure:

```
result_dir/
├── ckpts/              # Model checkpoints (.pt files)
├── ply/                 # Exported PLY files
├── renders/             # Rendered validation images
├── stats/               # Training statistics (JSON)
├── tb/                  # TensorBoard logs
└── cfg.yml              # Training configuration
```

## Examples

### Example 1: Basic Training

```bash
python server/train_from_ply.py \
    --ply_path data/point_cloud.ply \
    --camera_json data/cameras.json \
    --data_dir data \
    --result_dir results/basic_training \
    --max_steps 30000 \
    --save_ply True
```

### Example 2: Quick Fine-tuning

```bash
python server/train_from_ply.py \
    --ply_path data/point_cloud.ply \
    --camera_json data/cameras.json \
    --data_dir data \
    --result_dir results/finetune \
    --max_steps 5000 \
    --means_lr 1e-5 \
    --sh0_lr 1e-4
```

### Example 3: MCMC Strategy

```bash
python server/train_from_ply.py mcmc \
    --ply_path data/point_cloud.ply \
    --camera_json data/cameras.json \
    --data_dir data \
    --result_dir results/mcmc_training
```

## API Reference

### `load_ply(ply_path: str) -> Dict[str, torch.Tensor]`

Loads a PLY file and returns a dictionary with Gaussian parameters.

**Returns:**
- `means`: (N, 3) - Gaussian positions
- `scales`: (N, 3) - Gaussian scales (log space)
- `quats`: (N, 4) - Gaussian quaternions
- `opacities`: (N,) - Gaussian opacities (sigmoid space)
- `sh0`: (N, 1, 3) - SH DC coefficients
- `shN`: (N, K, 3) - SH rest coefficients

### `CameraParser`

Parser for JSON camera data.

**Parameters:**
- `data_dir`: Directory containing images
- `camera_json`: Path to JSON file with camera data
- `normalize`: Whether to normalize world space (default: False)

**Properties:**
- `image_names`: List of image names
- `image_paths`: List of image file paths
- `camtoworlds`: (N, 4, 4) camera-to-world matrices
- `Ks_dict`: Dictionary of intrinsic matrices
- `scene_scale`: Scene scale factor

### `CameraDataset`

PyTorch dataset for camera data.

**Parameters:**
- `parser`: CameraParser instance
- `split`: "train" or "val"
- `test_every`: Every N images is a test image
- `patch_size`: Optional random crop size

## Troubleshooting

### Common Issues

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

## Notes

- The PLY file format assumes scales are stored in log space and opacities in sigmoid space (as logits)
- Camera matrices should follow OpenCV convention (camera-to-world)
- The script automatically determines SH degree from the PLY file properties
- Training uses L1 + SSIM loss by default
- Densification strategies help improve model quality during training

## REST API Service

For programmatic access and integration, a FastAPI-based REST API service is available in `server/train_ply/`. This service provides:

- **Bulk Upload API**: Upload all files (PLY, camera JSON, images) in a single request
- **Incremental Upload API**: Upload files one-by-one before starting training
- **Job Management**: Track training jobs with status updates and progress monitoring
- **Result Download**: Download trained models, checkpoints, renders, and statistics
- **Async Processing**: Long-running training jobs execute asynchronously

### Quick Start

```bash
# Start the API server
python -m server.train_ply.main

# Or using uvicorn
uvicorn server.train_ply.main:app --host 0.0.0.0 --port 8000
```

Visit `http://localhost:8000/docs` for interactive API documentation.

### Incremental Upload Feature

The API supports incremental uploads, allowing you to:
- Create an upload job
- Upload PLY file separately
- Upload images one-by-one
- Upload camera JSON
- Upload training configuration
- Validate completeness before starting
- Start training when ready

This is particularly useful for:
- Large files that may timeout in bulk uploads
- Dynamically generated files
- Better error handling and validation
- More control over the upload process

See `server/train_ply/README.md` for complete API documentation and examples.

## License

This code follows the same license as the gsplat project.
