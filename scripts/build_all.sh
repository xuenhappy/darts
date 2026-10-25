#!/usr/bin/env bash
set -euo pipefail

# 一键编译 darts（C++ 核心 + cdarts Python 扩展）。阶段：
#   1. 本地 sysroot 依赖（libfmt/jsoncpp/protobuf，bootstrap_local_deps.sh，
#      stamp 缓存，幂等）
#   2. MNN 运行时库（DARTS_MNN_DIR 已设则直接采用，否则经 build_mnn.sh
#      从源码编译）
#   3. meson setup + compile（DARTS_MESON_BUILD_DIR，默认 build/meson）
#   4. --test 时运行 C++ 测试（默认跳过）
#
# 用法：scripts/build_all.sh [--clean] [--test]
# 环境：DARTS_MNN_DIR / DARTS_MESON_BUILD_DIR / DARTS_DEPS_DIR /
#       MNN_SOURCE_DIR / MNN_TARBALL（见 build_mnn.sh）/ PYTHON_BIN

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${DARTS_MESON_BUILD_DIR:-$ROOT_DIR/build/meson}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

clean=0
run_test=0
for arg in "$@"; do
  case "$arg" in
    --clean) clean=1 ;;
    --test) run_test=1 ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

if ((clean)); then
  rm -rf "$BUILD_DIR" "$ROOT_DIR"/build "$ROOT_DIR/python/build" \
    "$ROOT_DIR/python/dist" "$ROOT_DIR"/python/*.egg-info
fi

# 1) 依赖：系统 pkg-config 已提供（jsoncpp/fmt/protobuf）则直接用，
#    否则解包 deb 到本地 sysroot（bootstrap_local_deps.sh，stamp 幂等）
if pkg-config --exists jsoncpp fmt protobuf 2>/dev/null; then
  export DARTS_DEPS_DIR="${DARTS_DEPS_DIR:-}"
else
  DARTS_DEPS_DIR="${DARTS_DEPS_DIR:-$(bash "$ROOT_DIR/scripts/bootstrap_local_deps.sh")}"
  export DARTS_DEPS_DIR
fi

# 2) MNN 运行时（已设前缀且库存在则跳过编译）
if [[ -z "${DARTS_MNN_DIR:-}" || ! -e "$DARTS_MNN_DIR/lib/libMNN.so" ]]; then
  DARTS_MNN_DIR="$(bash "$ROOT_DIR/scripts/build_mnn.sh")"
fi
export DARTS_MNN_DIR

# 3) meson
command -v meson >/dev/null 2>&1 || {
  echo "meson is required (pip install meson ninja, or use scripts/bootstrap_devel_env.sh)" >&2
  exit 2
}
PYTHON_INCLUDE_DIR="$("$PYTHON_BIN" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
if [[ -f "$BUILD_DIR/meson-private/coredata.dat" ]]; then
  setup_cmd=(meson setup --reconfigure "$BUILD_DIR")
else
  setup_cmd=(meson setup "$BUILD_DIR" "$ROOT_DIR")
fi
"${setup_cmd[@]}" \
  -Dmnn-dir="$DARTS_MNN_DIR" \
  -Dbuild-python=true -Dbuild-tests="$( ((run_test)) && echo true || echo false )" \
  -Dlink-static=true \
  -Dpython-include-dir="$PYTHON_INCLUDE_DIR"
meson compile -C "$BUILD_DIR"

if ((run_test)); then
  meson test -C "$BUILD_DIR"
fi

echo "build ok → $BUILD_DIR（cdarts 扩展需拷入 python/darts/ 或 pip install python/）"
