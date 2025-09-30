#!/usr/bin/env bash
set -euo pipefail

OUTDIR="public/compiled"
mkdir -p "$OUTDIR"

echo "🚀 Building Paragon ISO Demo into $OUTDIR..."

# Common filename root (use git describe or version tag if you like)
APP="iso-demo"

# Linux AMD64
echo "🛠 linux/amd64"
GOOS=linux GOARCH=amd64 go build -o "$OUTDIR/${APP}_linux_amd64" .

# Linux ARM64
echo "🛠 linux/arm64"
GOOS=linux GOARCH=arm64 go build -o "$OUTDIR/${APP}_linux_arm64" .

# macOS Intel
echo "🛠 darwin/amd64"
GOOS=darwin GOARCH=amd64 go build -o "$OUTDIR/${APP}_darwin_amd64" .

# macOS Apple Silicon
echo "🛠 darwin/arm64"
GOOS=darwin GOARCH=arm64 go build -o "$OUTDIR/${APP}_darwin_arm64" .

# Windows 64-bit
echo "🛠 windows/amd64"
GOOS=windows GOARCH=amd64 go build -o "$OUTDIR/${APP}_windows_amd64.exe" .

# Windows ARM64
echo "🛠 windows/arm64"
GOOS=windows GOARCH=arm64 go build -o "$OUTDIR/${APP}_windows_arm64.exe" .

echo "✅ All builds complete. Files in $OUTDIR/"
ls -lh "$OUTDIR"
