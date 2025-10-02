#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="public/compiled"
mkdir -p "$OUT_DIR"

build() {
  local goos=$1
  local goarch=$2
  local output=$3

  echo "🛠 Building $goos/$goarch -> $output"

  # Each build runs in its own subshell so env vars don't leak
  (
    GOOS=$goos GOARCH=$goarch CGO_ENABLED=1 \
      go build -o "$OUT_DIR/$output" .
  )
}

# Linux builds
build linux amd64 iso-demo_linux_amd64
build linux arm64 iso-demo_linux_arm64

# macOS builds
build darwin amd64 iso-demo_darwin_amd64
build darwin arm64 iso-demo_darwin_arm64
