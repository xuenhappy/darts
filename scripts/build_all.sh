#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_DIR="$ROOT_DIR/python"
BUILD_DIR="${DARTS_MESON_BUILD_DIR:-$ROOT_DIR/build/meson}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ONNXRUNTIME_DIR="${DARTS_ONNXRUNTIME_DIR:-}"
DEPS_DIR="${DARTS_DEPS_DIR:-}"

usage() {
  cat <<'EOF'
Usage: scripts/build_all.sh [--clean] [--test]

Environment variables:
  DARTS_MESON_BUILD_DIR   Meson build directory (default: build/meson)
  DARTS_ONNXRUNTIME_DIR   ONNX Runtime prefix fallback
  DARTS_DEBUG_MEMORY      1 to enable dmalloc flags
  PYTHON_BIN              Python interpreter to run packaging commands
EOF
}

clean=0
run_test=0
for arg in "$@"; do
  case "$arg" in
    --clean) clean=1 ;;
    --test) run_test=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $arg" >&2; usage; exit 2 ;;
  esac
done

if [[ $clean -eq 1 ]]; then
  rm -rf "$BUILD_DIR" "$ROOT_DIR"/build "$PY_DIR"/build "$PY_DIR"/dist "$PY_DIR"/*.egg-info
fi

if [[ -z "$ONNXRUNTIME_DIR" ]]; then
  ONNXRUNTIME_DIR="$(bash "$ROOT_DIR/scripts/fetch_onnxruntime.sh")"
fi

if [[ -z "$DEPS_DIR" ]]; then
  DEPS_DIR="$(bash "$ROOT_DIR/scripts/bootstrap_local_deps.sh")"
fi

if ! command -v meson >/dev/null 2>&1; then
  echo "meson is required" >&2
  exit 1
fi

export PATH="$DEPS_DIR/sysroot/usr/bin:$PATH"
export PKG_CONFIG_SYSROOT_DIR="$DEPS_DIR/sysroot"
pkg_config_dirs="$(find "$DEPS_DIR/sysroot/usr/lib" -type d -name pkgconfig -print | paste -sd: -)"
export PKG_CONFIG_LIBDIR="${pkg_config_dirs:+$pkg_config_dirs:}$DEPS_DIR/sysroot/usr/share/pkgconfig"
export DARTS_ONNXRUNTIME_DIR="$ONNXRUNTIME_DIR"
export DARTS_DEPS_DIR="$DEPS_DIR"

BUILD_VENV="${DARTS_BUILD_VENV:-$BUILD_DIR/.build-venv}"
if [[ ! -x "$BUILD_VENV/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$BUILD_VENV"
fi
BUILD_PYTHON="$BUILD_VENV/bin/python"
"$BUILD_PYTHON" -m pip install -U pip build "setuptools>=68" wheel \
  "Cython>=3.0" "meson>=1.3" "ninja>=1.11" >/dev/null
export PATH="$BUILD_VENV/bin:$PATH"
PYTHON_INCLUDE_DIR="$("$BUILD_PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"

pushd "$ROOT_DIR" >/dev/null
meson setup "$BUILD_DIR" . -Dbuild-python=true -Dbuild-tests=false -Dlink-static=true \
  -Donnxruntime-dir="$ONNXRUNTIME_DIR" -Dpython-include-dir="$PYTHON_INCLUDE_DIR"
meson compile -C "$BUILD_DIR"
popd >/dev/null

pushd "$PY_DIR" >/dev/null
"$BUILD_PYTHON" -m build --wheel --no-isolation
popd >/dev/null

if [[ $run_test -eq 1 ]]; then
  wheel_path="$(ls -1t "$PY_DIR"/dist/*.whl | head -n1)"
  venv_dir="${DARTS_TEST_VENV:-$ROOT_DIR/build/test-venv}"
  "$PYTHON_BIN" -m venv "$venv_dir"
  # shellcheck disable=SC1091
  source "$venv_dir/bin/activate"
  python -m pip install -U pip >/dev/null
  python -m pip install --force-reinstall --ignore-requires-python "$wheel_path"
  python - <<'PY'
import darts
from darts import DSegment, PyAtomList
print("darts import ok")
alist = PyAtomList("中文分词测试")
print("atoms:", len(alist))
print("package:", darts.__file__)
PY
  deactivate
fi
