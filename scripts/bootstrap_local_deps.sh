#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPS_DIR="${DARTS_DEPS_DIR:-$ROOT_DIR/build/deps}"
SYSROOT="$DEPS_DIR/sysroot"
STAMP="$DEPS_DIR/.bootstrap-v4.done"

if [[ -f "$STAMP" && -d "$SYSROOT" ]]; then
  echo "$DEPS_DIR"
  exit 0
fi

mkdir -p "$DEPS_DIR"

if ! command -v apt >/dev/null 2>&1 || ! command -v dpkg-deb >/dev/null 2>&1; then
  echo "apt and dpkg-deb are required to bootstrap local dependencies" >&2
  exit 1
fi

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

pushd "$tmp_dir" >/dev/null
apt download \
  libfmt-dev libfmt10 \
  libjsoncpp-dev libjsoncpp26 \
  libprotobuf-dev libprotobuf32t64 libprotobuf-lite32t64 \
  protobuf-compiler zlib1g-dev zlib1g >/dev/null
mkdir -p "$SYSROOT"
for deb in ./*.deb; do
  dpkg-deb -x "$deb" "$SYSROOT"
done
popd >/dev/null

cat > "$STAMP" <<'EOF'
ok
EOF

echo "$DEPS_DIR"
