#!/usr/bin/env bash
set -euo pipefail

# 从源码编译安装 MNN（libMNN + 头文件）——darts 的推理运行时后端
# （2026-10-25 起，替代 onnxruntime）。
#
# 输入优先级（均未设置时 git clone，需要网络）：
#   MNN_SOURCE_DIR  已就绪的 MNN 源码目录
#   MNN_TARBALL     源码 tarball 路径
# 安装前缀：DARTS_MNN_DIR（默认 <repo>/build/MNN-install）
# 成功后把安装前缀打印到 stdout（供 build_all.sh / bootstrap_devel_env.sh 消费）。
#
# 纯 CPU 构建：-DMNN_BUILD_CONVERTER=OFF -DMNN_BUILD_TOOLS=OFF（模型转换用
# pip 包 MNN.tools.mnnconvert），ARM 平台自动开 NEON/ARM82。

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_DIR="${DARTS_MNN_DIR:-$ROOT_DIR/build/MNN-install}"
BUILD_DIR="$ROOT_DIR/build/mnn-build"
STAMP="$INSTALL_DIR/.mnn-build.done"

if [[ -f "$STAMP" && -e "$INSTALL_DIR/lib/libMNN.so" ]]; then
  echo "$INSTALL_DIR"
  exit 0
fi

mkdir -p "$(dirname "$INSTALL_DIR")"

SRC_DIR="${MNN_SOURCE_DIR:-}"
if [[ -z "$SRC_DIR" && -n "${MNN_TARBALL:-}" ]]; then
  SRC_DIR="$ROOT_DIR/build/mnn-src"
  rm -rf "$SRC_DIR"
  mkdir -p "$SRC_DIR"
  tar -xzf "$MNN_TARBALL" -C "$SRC_DIR" --strip-components=1
fi
if [[ -z "$SRC_DIR" ]]; then
  SRC_DIR="$ROOT_DIR/build/mnn-src"
  git clone --depth 1 https://github.com/alibaba/MNN.git "$SRC_DIR"
fi

command -v cmake >/dev/null 2>&1 || {
  echo "cmake is required (see scripts/bootstrap_linux_deps.sh)" >&2
  exit 2
}

cmake -S "$SRC_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
  -DMNN_BUILD_SHARED_LIB=ON -DMNN_BUILD_CONVERTER=OFF -DMNN_BUILD_DEMO=OFF \
  -DMNN_BUILD_TOOLS=OFF -DMNN_BUILD_TEST=OFF \
  -DMNN_CUDA=OFF -DMNN_OPENCL=OFF -DMNN_VULKAN=OFF -DMNN_METAL=OFF \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR"
cmake --build "$BUILD_DIR" --parallel "$(nproc)"
cmake --install "$BUILD_DIR"

[[ -e "$INSTALL_DIR/lib/libMNN.so" ]] || {
  echo "libMNN.so missing after install" >&2
  exit 3
}
date -Is >"$STAMP"
echo "$INSTALL_DIR"
