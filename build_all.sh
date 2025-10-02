#!/usr/bin/env bash
set -u  # keep unset variable errors, but not -e so we don't exit on failure
set -o pipefail

OUTDIR="public/compiled"
mkdir -p "$OUTDIR"

echo "🚀 Building Paragon ISO Demo into $OUTDIR..."

APP="iso-demo"

build_target() {
  local os=$1
  local arch=$2
  local outfile=$3

  echo "🛠 $os/$arch"
  if GOOS=$os GOARCH=$arch go build -o "$outfile" .; then
    echo "   ✅ Built $outfile"
  else
    echo "   ❌ Failed to build for $os/$arch (skipping)"
  fi
}

# Linux AMD64
build_target linux amd64 "$OUTDIR/${APP}_linux_amd64"

# Linux ARM64
build_target linux arm64 "$OUTDIR/${APP}_linux_arm64"

# macOS Intel
build_target darwin amd64 "$OUTDIR/${APP}_darwin_amd64"

# macOS Apple Silicon
build_target darwin arm64 "$OUTDIR/${APP}_darwin_arm64"

# Windows 64-bit
build_target windows amd64 "$OUTDIR/${APP}_windows_amd64.exe"

# Windows ARM64
build_target windows arm64 "$OUTDIR/${APP}_windows_arm64.exe"

echo "✅ Build script finished. Files in $OUTDIR/:"
ls -lh "$OUTDIR"
