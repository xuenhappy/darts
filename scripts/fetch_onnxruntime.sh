#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="${DARTS_ONNXRUNTIME_VERSION:-1.17.3}"
ARCHIVE="onnxruntime-linux-x64-${VERSION}.tgz"
URL="${DARTS_ONNXRUNTIME_URL:-https://github.com/microsoft/onnxruntime/releases/download/v${VERSION}/${ARCHIVE}}"
CACHE_DIR="${DARTS_DEPS_DIR:-$ROOT_DIR/build/deps}"
TARGET_DIR="${DARTS_ONNXRUNTIME_DIR:-$CACHE_DIR/onnxruntime-${VERSION}}"

if [[ -d "$TARGET_DIR" && -f "$TARGET_DIR/include/onnxruntime_cxx_api.h" ]]; then
  echo "$TARGET_DIR"
  exit 0
fi

mkdir -p "$CACHE_DIR"
tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

archive_path="$CACHE_DIR/$ARCHIVE"
if [[ ! -f "$archive_path" ]]; then
  curl -L --retry 3 --retry-delay 2 -o "$archive_path" "$URL"
fi

rm -rf "$TARGET_DIR"
mkdir -p "$TARGET_DIR"
tar -xzf "$archive_path" -C "$tmp_dir"
extracted="$(find "$tmp_dir" -maxdepth 1 -mindepth 1 -type d | head -n1)"
cp -a "$extracted/." "$TARGET_DIR/"

if [[ ! -f "$TARGET_DIR/include/onnxruntime_cxx_api.h" ]]; then
  echo "failed to fetch a usable onnxruntime bundle from $URL" >&2
  exit 1
fi

echo "$TARGET_DIR"
