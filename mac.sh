#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="public/compiled"
mkdir -p "$OUT_DIR"

build() {
  local goarch=$1
  local output=$2

  echo "🍎 Building darwin/$goarch -> $output"

  (
    GOOS=darwin GOARCH=$goarch CGO_ENABLED=1 \
      go build -o "$OUT_DIR/$output" .
  )
}

# macOS builds only
build amd64 iso-demo_darwin_amd64
build arm64 iso-demo_darwin_arm64

echo ""
echo "✅ macOS builds complete:"
ls -la "$OUT_DIR"
