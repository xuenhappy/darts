#!/usr/bin/env bash
set -euo pipefail

# GraphLoss uses an exact forward-backward posterior with bounded analytic edge
# gradients, so production no longer builds the unsafe deep DP autograd graph.
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

PYTHON="${PYTHON:-/home/xuen/.venv/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-model_bin}"
MODEL_DIR="${MODEL_DIR:-data/models/neural}"
HIDDEN_SIZE="${HIDDEN_SIZE:-64}"
MAX_SPAN="${MAX_SPAN:-8}"
GRAPH_MAX_SPAN="${GRAPH_MAX_SPAN:-5}"
TARGET_BATCH_EDGES="${TARGET_BATCH_EDGES:-3000}"
MIN_GRAPH_EDGES="${MIN_GRAPH_EDGES:-3}"
MAX_GRAPH_EDGES="${MAX_GRAPH_EDGES:-2800}"
MAX_GRAPH_NODES="${MAX_GRAPH_NODES:-550}"
MAX_GRAPH_WORDPIECES="${MAX_GRAPH_WORDPIECES:-108}"
EPOCHS="${EPOCHS:-10}"
PATIENCE="${PATIENCE:-3}"
RECOGNIZER_BATCH_SIZE="${RECOGNIZER_BATCH_SIZE:-64}"
# GraphLossSparse accumulates complete candidate DAGs. Very large graph batches
# are numerically unstable and do not behave like ordinary dense mini-batches.
QUANTIZER_BATCH_SIZE="${QUANTIZER_BATCH_SIZE:-4}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
QUANTIZER_WEIGHT="${QUANTIZER_WEIGHT:-0.1}"
JOINT_UPDATE_MODE="${JOINT_UPDATE_MODE:-alternating}"
GRAPH_AUXILIARY_WEIGHT="${GRAPH_AUXILIARY_WEIGHT:-0.1}"
GRAPH_AUXILIARY_UNLABELLED_WEIGHT="${GRAPH_AUXILIARY_UNLABELLED_WEIGHT:-0.05}"
GRAPH_AUXILIARY_MAX_EDGES="${GRAPH_AUXILIARY_MAX_EDGES:-256}"
DEVICE="${DEVICE:-cuda}"
if [[ -z "${GRAPH_DETACH_CONTEXT:-}" ]]; then
  if [[ "$DEVICE" == "cpu" ]]; then
    GRAPH_DETACH_CONTEXT=0
  else
    GRAPH_DETACH_CONTEXT=1
  fi
fi
# AMP backward is currently unsafe for the shared recognizer/FP32 graph loss.
# Keep it opt-in until both branches can use one numerically consistent dtype.
AMP="${AMP:-0}"
# Two same-GPU processes are supported as an experimental throughput mode, but
# remain opt-in because concurrent backward currently reproduces CUDA faults.
PARALLEL="${PARALLEL:-0}"
RUN_BINARY="${RUN_BINARY:-1}"
RUN_LAC="${RUN_LAC:-1}"

# Avoid two concurrent jobs each creating a full CPU thread pool.
if [[ -z "${OMP_NUM_THREADS:-}" ]]; then
  CPU_COUNT="$(nproc)"
  if [[ "$PARALLEL" == "1" ]]; then
    export OMP_NUM_THREADS="$(( CPU_COUNT > 2 ? CPU_COUNT / 2 : 1 ))"
  else
    export OMP_NUM_THREADS="$CPU_COUNT"
  fi
fi
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"

mkdir -p "$OUTPUT_ROOT/neural-joint" "$OUTPUT_ROOT/lac-joint" "$MODEL_DIR"

amp_args=()
if [[ "$AMP" == "1" ]]; then
  amp_args+=(--amp)
else
  amp_args+=(--no-amp)
fi

common_train_args=(
  --epochs "$EPOCHS"
  --patience "$PATIENCE"
  --recognizer-batch-size "$RECOGNIZER_BATCH_SIZE"
  --quantizer-batch-size "$QUANTIZER_BATCH_SIZE"
  --hidden-size "$HIDDEN_SIZE"
  --learning-rate "$LEARNING_RATE"
  --quantizer-weight "$QUANTIZER_WEIGHT"
  --joint-update-mode "$JOINT_UPDATE_MODE"
  --graph-auxiliary-weight "$GRAPH_AUXILIARY_WEIGHT"
  --graph-auxiliary-unlabelled-weight "$GRAPH_AUXILIARY_UNLABELLED_WEIGHT"
  --graph-auxiliary-max-edges "$GRAPH_AUXILIARY_MAX_EDGES"
  --max-nonfinite-batches 0
  --max-span "$MAX_SPAN"
  --graph-max-span "$GRAPH_MAX_SPAN"
  --target-batch-edges "$TARGET_BATCH_EDGES"
  --min-graph-edges "$MIN_GRAPH_EDGES"
  --max-graph-edges "$MAX_GRAPH_EDGES"
  --max-graph-nodes "$MAX_GRAPH_NODES"
  --max-graph-wordpieces "$MAX_GRAPH_WORDPIECES"
  --device "$DEVICE"
  "${amp_args[@]}"
)
if [[ "$GRAPH_DETACH_CONTEXT" == "1" ]]; then
  common_train_args+=(--graph-detach-context)
else
  common_train_args+=(--no-graph-detach-context)
fi

train_binary() {
  echo "stage=neural-binary action=train device=$DEVICE amp=$AMP"
  binary_resume=()
  if [[ -n "${BINARY_RESUME:-}" ]]; then
    binary_resume+=(--resume "$BINARY_RESUME")
  fi
  "$PYTHON" scripts/train_joint.py train \
    --recognizer-kind binary \
    --train data/generated/cws-train.txt \
    --dev data/generated/cws-dev.txt \
    --mode hybrid \
    --type-map data/codes/type.hx.txt \
    --output-dir "$OUTPUT_ROOT/neural-joint" \
    "${binary_resume[@]}" \
    "${common_train_args[@]}"
}

train_lac() {
  echo "stage=lac-syntax action=train device=$DEVICE amp=$AMP"
  lac_resume=()
  if [[ -n "${LAC_RESUME:-}" ]]; then
    lac_resume+=(--resume "$LAC_RESUME")
  fi
  "$PYTHON" scripts/train_joint.py train \
    --recognizer-kind syntax \
    --train data/generated/lac-train.txt \
    --dev data/generated/lac-dev.txt \
    --mode lac \
    --type-map data/codes/pos.hx.txt \
    --output-dir "$OUTPUT_ROOT/lac-joint" \
    "${lac_resume[@]}" \
    "${common_train_args[@]}"
}

if [[ "$PARALLEL" == "1" && "$RUN_BINARY" == "1" && "$RUN_LAC" == "1" ]]; then
  echo "scheduler=parallel cuda_launch_blocking=$CUDA_LAUNCH_BLOCKING omp_threads=$OMP_NUM_THREADS"
  train_binary &
  binary_pid=$!
  train_lac &
  lac_pid=$!

  status=0
  wait "$binary_pid" || status=1
  wait "$lac_pid" || status=1
  if (( status != 0 )); then
    echo "status=failed reason=parallel-training-child-failed" >&2
    exit "$status"
  fi
else
  echo "scheduler=sequential cuda_launch_blocking=$CUDA_LAUNCH_BLOCKING omp_threads=$OMP_NUM_THREADS"
  if [[ "$RUN_BINARY" == "1" ]]; then
    train_binary
  fi
  if [[ "$RUN_LAC" == "1" ]]; then
    train_lac
  fi
fi

evaluate_binary() {
  echo "stage=neural-binary action=evaluate"
  "$PYTHON" scripts/train_joint.py evaluate \
    "$OUTPUT_ROOT/neural-joint/best.pt" \
    --data data/generated/cws-test.txt \
    --mode hybrid \
    --type-map data/codes/type.hx.txt \
    --recognizer-batch-size "$RECOGNIZER_BATCH_SIZE" \
    --quantizer-batch-size "$QUANTIZER_BATCH_SIZE" \
    --device "$DEVICE"
}

evaluate_lac() {
  echo "stage=lac-syntax action=evaluate"
  "$PYTHON" scripts/train_joint.py evaluate \
    "$OUTPUT_ROOT/lac-joint/best.pt" \
    --data data/generated/lac-test.txt \
    --mode lac \
    --type-map data/codes/pos.hx.txt \
    --recognizer-batch-size "$RECOGNIZER_BATCH_SIZE" \
    --quantizer-batch-size "$QUANTIZER_BATCH_SIZE" \
    --device "$DEVICE"
}

if [[ "$PARALLEL" == "1" ]]; then
  evaluate_binary &
  binary_eval_pid=$!
  evaluate_lac &
  lac_eval_pid=$!
  status=0
  wait "$binary_eval_pid" || status=1
  wait "$lac_eval_pid" || status=1
  if (( status != 0 )); then
    echo "status=failed reason=parallel-evaluation-child-failed" >&2
    exit "$status"
  fi
else
  evaluate_binary
  evaluate_lac
fi

"$PYTHON" scripts/train_joint.py export "$OUTPUT_ROOT/neural-joint/best.pt" "$MODEL_DIR"
"$PYTHON" scripts/train_joint.py export "$OUTPUT_ROOT/lac-joint/best.pt" "$MODEL_DIR"

echo "models=$MODEL_DIR status=complete"
