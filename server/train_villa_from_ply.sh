#!/bin/bash
# Training script for villa dataset from PLY file

set -e  # Exit on error

# Get script directory (server/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default paths
PLY_PATH="${SCRIPT_DIR}/examples/villa/point_cloud_6999.ply"
CAMERAS_JSON="${SCRIPT_DIR}/examples/villa/cameras.json"
DATA_DIR="${PROJECT_ROOT}/examples/data/villa"
RESULT_DIR="${SCRIPT_DIR}/examples/results/villa"

# Calculate training steps: 10 steps per image
NUM_IMAGES=$(python3 -c "import json; print(len(json.load(open('$CAMERAS_JSON'))))")
STEPS_PER_IMAGE=10
MAX_STEPS=$((NUM_IMAGES * STEPS_PER_IMAGE))

# Verify files exist
if [ ! -f "$PLY_PATH" ]; then
    echo "Error: PLY file not found: $PLY_PATH"
    exit 1
fi

if [ ! -f "$CAMERAS_JSON" ]; then
    echo "Error: Cameras JSON file not found: $CAMERAS_JSON"
    exit 1
fi

if [ ! -d "$DATA_DIR/images" ]; then
    echo "Error: Images directory not found: $DATA_DIR/images"
    exit 1
fi

# Print configuration
echo "=========================================="
echo "Training Villa Dataset from PLY"
echo "=========================================="
echo "PLY file:        $PLY_PATH"
echo "Cameras JSON:    $CAMERAS_JSON"
echo "Data directory:  $DATA_DIR"
echo "Result directory: $RESULT_DIR"
echo "Number of images: $NUM_IMAGES"
echo "Steps per image:  $STEPS_PER_IMAGE"
echo "Max steps:        $MAX_STEPS"
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    echo "CUDA devices:    $CUDA_VISIBLE_DEVICES"
fi
echo "=========================================="
echo ""

# Change to project root for proper imports
cd "$PROJECT_ROOT"

# Training command
# Note: For multi-GPU training, set CUDA_VISIBLE_DEVICES before running:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 ./train_villa_from_ply.sh
python server/train_from_ply.py default \
    --ply-path "$PLY_PATH" \
    --camera-json "$CAMERAS_JSON" \
    --data-dir "$DATA_DIR" \
    --result-dir "$RESULT_DIR" \
    --max-steps "$MAX_STEPS" \
    --batch-size 1 \
    --sh-degree 3 \
    --ssim-lambda 0.2 \
    --test-every 8 \
    --save-ply \
    --ply-steps "$MAX_STEPS" \
    --eval-steps "$MAX_STEPS" \
    --save-steps "$MAX_STEPS" \
    --no-normalize-world-space \
    --camera-model pinhole \
    --disable-viewer

echo ""
echo "=========================================="
echo "Training completed!"
echo "Results saved to: $RESULT_DIR"
echo "=========================================="
