#!/usr/bin/env bash
set -euo pipefail

# This workstation has reproduced asynchronous illegal memory accesses under
# the joint graph workload. Keep every CUDA launch synchronous for both debug
# and production so the failing operation is reported before the driver state
# can be damaged.
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"

PYTHON="${PYTHON:-/home/xuen/.venv/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-model_bin}"
MODEL_DIR="${MODEL_DIR:-data/models/neural}"
HIDDEN_SIZE="${HIDDEN_SIZE:-32}"
EPOCHS="${EPOCHS:-4}"
PATIENCE="${PATIENCE:-2}"
RECOGNIZER_BATCH_SIZE="${RECOGNIZER_BATCH_SIZE:-16}"
QUANTIZER_BATCH_SIZE="${QUANTIZER_BATCH_SIZE:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
QUANTIZER_WEIGHT="${QUANTIZER_WEIGHT:-0.01}"

mkdir -p "$OUTPUT_ROOT/neural-joint" "$OUTPUT_ROOT/lac-joint" "$MODEL_DIR"

echo "stage=neural-binary cuda_launch_blocking=$CUDA_LAUNCH_BLOCKING"
"$PYTHON" scripts/train_joint.py train \
  --recognizer-kind binary \
  --train data/generated/cws-train.txt \
  --dev data/generated/cws-dev.txt \
  --mode hybrid \
  --type-map data/codes/type.hx.txt \
  --output-dir "$OUTPUT_ROOT/neural-joint" \
  --epochs "$EPOCHS" \
  --patience "$PATIENCE" \
  --recognizer-batch-size "$RECOGNIZER_BATCH_SIZE" \
  --quantizer-batch-size "$QUANTIZER_BATCH_SIZE" \
  --hidden-size "$HIDDEN_SIZE" \
  --learning-rate "$LEARNING_RATE" \
  --quantizer-weight "$QUANTIZER_WEIGHT" \
  --max-nonfinite-batches 8 \
  --max-span 5 \
  --device cuda \
  --no-amp

"$PYTHON" scripts/train_joint.py evaluate \
  "$OUTPUT_ROOT/neural-joint/best.pt" \
  --data data/generated/cws-test.txt \
  --mode hybrid \
  --type-map data/codes/type.hx.txt \
  --device cuda

"$PYTHON" scripts/train_joint.py export \
  "$OUTPUT_ROOT/neural-joint/best.pt" "$MODEL_DIR"

echo "stage=lac-syntax cuda_launch_blocking=$CUDA_LAUNCH_BLOCKING"
"$PYTHON" scripts/train_joint.py train \
  --recognizer-kind syntax \
  --train data/generated/lac-train.txt \
  --dev data/generated/lac-dev.txt \
  --mode lac \
  --type-map data/codes/pos.hx.txt \
  --output-dir "$OUTPUT_ROOT/lac-joint" \
  --epochs "$EPOCHS" \
  --patience "$PATIENCE" \
  --recognizer-batch-size "$RECOGNIZER_BATCH_SIZE" \
  --quantizer-batch-size "$QUANTIZER_BATCH_SIZE" \
  --hidden-size "$HIDDEN_SIZE" \
  --learning-rate "$LEARNING_RATE" \
  --quantizer-weight "$QUANTIZER_WEIGHT" \
  --max-nonfinite-batches 8 \
  --max-span 5 \
  --device cuda \
  --no-amp

"$PYTHON" scripts/train_joint.py evaluate \
  "$OUTPUT_ROOT/lac-joint/best.pt" \
  --data data/generated/lac-test.txt \
  --mode lac \
  --type-map data/codes/pos.hx.txt \
  --device cuda

"$PYTHON" scripts/train_joint.py export \
  "$OUTPUT_ROOT/lac-joint/best.pt" "$MODEL_DIR"

echo "models=$MODEL_DIR status=complete"
