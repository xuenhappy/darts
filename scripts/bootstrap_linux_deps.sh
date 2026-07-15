#!/usr/bin/env bash
set -euo pipefail

if ! command -v apt-get >/dev/null 2>&1; then
  echo "apt-get is required on Linux" >&2
  exit 1
fi

sudo apt-get update
sudo apt-get install -y \
  build-essential \
  pkg-config \
  protobuf-compiler \
  libprotobuf-dev \
  libjsoncpp-dev \
  libfmt-dev \
  python3-dev \
  python3-venv \
  python3-pip \
  python3-setuptools \
  python3-wheel \
  curl \
  ca-certificates
