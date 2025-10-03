# Paragon ISO Telemetry Harness

**Technical Name:** Cross-Device Inference Telemetry & Drift Harness  
**Short Name:** Paragon ISO Demo

This project is a **portable, distributed inference telemetry system**.  
It lets you build models once, run them across heterogeneous hardware (CPU, GPU, WebGPU), and automatically collect:

- System information (CPU, GPU, RAM, OS).
- WebGPU initialization times (mounting cost).
- Per-sample inference latencies for CPU vs GPU.
- Raw output logits (the exact outputs of the neural net).
- Drift metrics (CPU vs GPU numerical differences).
- ADHD10 diagnostic buckets (correct, off-by-1, wrong, agreement/disagreement).

The reports are serialized as JSON and uploaded back to a host, enabling **cross-device reproducibility analysis**.  
This is effectively an open **inference conformance lab**, normally only run behind closed doors at major ML companies.

---

## Goals

- **Reproducibility:** Verify that Paragon models produce identical results across devices, backends, and drivers.
- **Drift Analysis:** Detect subtle floating-point inconsistencies or regressions.
- **Benchmarking:** Measure inference throughput (CPU vs GPU) and WebGPU startup overhead.
- **Telemetry:** Collect structured reports across machines for aggregation and comparison.

Think of it as a public, distributed alternative to MLPerf, with full transparency down to the sample level.

---

## Project Structure

```
iso-demo/
├── all_gen.py                          # Python script for generating assets (e.g., PNGs)
├── build_all.sh                        # Shell script for cross-platform builds
├── compare.go                          # Go module for CPU vs GPU comparisons
├── distributed_ML_infrastructure_testing_framework.docx  # Supporting documentation
├── evaluate.go                         # Go module for ADHD10 evaluation
├── go.mod                              # Go module definition
├── go.sum                              # Go dependencies lockfile
├── LICENSE                             # Apache 2.0 license
├── main.go                             # Entry point for the CLI demo
├── mnist.go                            # Go module for MNIST data handling
├── models.go                           # Go module for model loading and management
├── sysbench.go                         # Go module for system benchmarking
├── sysprobe.go                         # Go module for system information probing
├── telecmd.go                          # Go module for telemetry commands
├── telemetrics.go                      # Go module for metrics collection
├── train.go                            # Go module for model training
├── webupload.go                        # Go module for web uploads
├── websrv.go                           # Go module for the web server
└── public/                             # Static assets served by the web server
    ├── compiled/                       # Built binaries for different platforms
    │   └── iso-demo_linux_amd64        # Example: Linux AMD64 binary
    ├── mnist/                          # Raw MNIST dataset files
    │   ├── t10k-images-idx3-ubyte
    │   ├── t10k-images-idx3-ubyte.gz
    │   ├── t10k-labels-idx1-ubyte
    │   ├── t10k-labels-idx1-ubyte.gz
    │   ├── train-images-idx3-ubyte
    │   ├── train-images-idx3-ubyte.gz
    │   ├── train-labels-idx1-ubyte
    │   └── train-labels-idx1-ubyte.gz
    ├── mnist_png/                      # Converted MNIST images as PNGs
    │   └── all/
    │       ├── 0/  # Directory for digit 0 PNGs
    │       ├── 1/
    │       ├── 2/
    │       ├── 3/
    │       ├── 4/
    │       ├── 5/
    │       ├── 6/
    │       ├── 7/
    │       ├── 8/
    │       └── 9/
    ├── models/                         # Pre-trained MNIST models (JSON format)
    │   ├── manifest.json               # Model inventory
    │   ├── mnist_L1.json
    │   ├── mnist_L2.json
    │   ├── mnist_M1.json
    │   ├── mnist_M2.json
    │   ├── mnist_M3.json
    │   ├── mnist_S1.json
    │   ├── mnist_S2.json
    │   ├── mnist_S3.json
    │   ├── mnist_XL1.json
    │   └── mnist_XL2.json
    └── reports/                        # Telemetry JSON reports from runs
        ├── telemetry_5019cf92fc8afcc888642b771c20765e_1759362264.json
        ├── telemetry_6f27147ba1c111302fc17bd98e4f4570_1759362510.json
        ├── telemetry_7802fc797b5a04c208b2b1ffab57cc75_1759362484.json
        └── telemetry_b04fe9e9a31ebf94b7bbaf3c4d59d6bc_1759362078.json
```

---

## Requirements

- **Go 1.22+** (for building and running the demo).
- **Vulkan drivers** (for WebGPU backend):

  - **Linux:**

    - Debian/Ubuntu:

      ```bash
      sudo apt install vulkan-tools mesa-vulkan-drivers
      ```

    - Fedora:

      ```bash
      sudo dnf install vulkan vulkan-tools
      ```

  - **Windows:**

    - Ensure your GPU drivers (NVIDIA/AMD/Intel) are up to date.
    - If you want extra validation tools, install the [LunarG Vulkan SDK](https://vulkan.lunarg.com/sdk/home#windows).

  - **macOS:** WebGPU automatically falls back to Metal (no Vulkan needed).

- **Git** (to clone the repo).

---

## Running inside windows

# Force WebGPU to use Vulkan backend in powershell

```
$env:WGPU_BACKEND = "vulkan"
```

# Enable logging for debugging

```
$env:RUST_LOG = "wgpu_core=trace,wgpu_hal=trace"
```

# Optional: avoid AMD switchable graphics layer issues

```
$env:DISABLE_LAYER_AMD_SWITCHABLE_GRAPHICS_1 = "1"
```

# Run iso-demo_windows_amd64.exe

```
.\iso-demo_windows_amd64.exe
```

Optional:

- `jq`, `curl` (for inspecting results and fetching binaries).

---

## Building

From the repo root:

```bash
# Native build for your platform
go build -o iso-demo .

# Cross builds (Linux, macOS, Windows, ARM64)
./build_all.sh
```

The binaries are placed in `public/compiled/`.

---

## Building for Windows on Linux (Fedora/RHEL)

To cross-compile your Go project for Windows on a Fedora or RHEL-based Linux distribution, follow these steps. This assumes you have CGO dependencies and need the MinGW toolchain.

### Install the MinGW Toolchain

```bash
sudo dnf install mingw64-gcc mingw64-gcc-c++
```

### Set Environment Variables

Tell Go to use the Windows cross-compiler:

```bash
export CC=x86_64-w64-mingw32-gcc
export CXX=x86_64-w64-mingw32-g++
```

### Build the Binary

Enable CGO and cross-compile to Windows:

```bash
GOOS=windows GOARCH=amd64 CGO_ENABLED=1 go build -o public/compiled/iso-demo_windows_amd64.exe .
```

## Running

Start the demo binary:

```bash
./iso-demo
```

You’ll see the interactive menu:

```
=== Paragon ISO Demo ===
1) Show computer info (JSON)
2) Run MNIST experiment (download/train/test via PILOT)
3) Export MNIST images to PNG
4) Create the models for testing
5) Benchmark models CPU on digit samples
6) Benchmark models GPU on digit samples
7) Compare CPU vs GPU (choose model)
8) Train model(s)
9) Evaluate a model (ADHD metrics)
10) Run CPU numeric microbench
11) Web server: start/stop/status
12) Telemetry: pull models from host → run → push report
0) Exit
```

---

## Typical Workflow

1. **Host machine** (acts as the central server):

   ```bash
   ./iso-demo
   # choose option 11 → start web server
   ```

   This serves models under `/models/` and accepts uploads under `/upload`.

2. **Client machine** (to run telemetry):

   - Fetch the binary:

     ```bash
     curl -O http://<host-ip>:8080/compiled/iso-demo_linux_amd64
     chmod +x iso-demo_linux_amd64
     ./iso-demo_linux_amd64
     ```

   - Run telemetry:

     ```
     Select: 12
     Target host base: http://<host-ip>:8080
     Source environment:
       1) native
       2) wasm-bun
       3) wasm-ionic
     Select [1-3]:
     ```

   - This pulls models + MNIST data, runs inference, generates a report,
     and pushes the JSON back to the host under `/reports/`.

3. **Inspect results:**

   ```bash
   curl http://<host-ip>:8080/reports/
   curl http://<host-ip>:8080/reports/telemetry_<machineid>_*.json | jq
   ```

---

## Report Structure

Each telemetry JSON includes:

- `system_info`: CPU, GPU, OS, RAM.
- `per_model`: For each model file tested:

  - `webgpu_init_time_ms`: GPU init cost.
  - `cpu`/`gpu`: Per-digit timings, predictions, raw outputs.
  - `drift`: MaxAbs and MAE between CPU/GPU outputs.
  - `adhd10`: Accuracy, agreement counts, bucket roll-ups, and per-sample bucket labels.

Example snippet:

```json
{
  "model_file": "mnist_L1.json",
  "webgpu_init_time_ms": 18.5,
  "cpu": [
    {"digit": 0, "elapsed_ms": 0.09, "pred": 0, "output": [...]},
    {"digit": 1, "elapsed_ms": 0.08, "pred": 1, "output": [...]}
  ],
  "gpu": [
    {"digit": 0, "elapsed_ms": 0.06, "pred": 0, "output": [...]},
    {"digit": 1, "elapsed_ms": 0.05, "pred": 1, "output": [...]}
  ],
  "adhd10": {
    "top1_accuracy_cpu": 0.5,
    "top1_accuracy_gpu": 0.5,
    "cpu_vs_gpu_agree_count": 10,
    "buckets": {
      "cpu_correct": 5,
      "cpu_wrong": 5,
      "gpu_correct": 5,
      "gpu_wrong": 5,
      "cpu_gpu_agree": 10
    },
    "per_sample": [
      {"digit": 0, "cpu_pred": 0, "gpu_pred": 0, "cpu_bucket": "correct", "gpu_bucket": "correct", "agreement": "agree"},
      ...
    ]
  }
}
```

---

## Why This Matters

This project is not “just a demo.”
It’s a **production-grade inference observability pipeline** that most ML teams build privately:

- Reproducibility audits across CPU/GPU/driver boundaries.
- Drift detection with per-sample transparency.
- Portable reporting, sharable across environments.
- Open, distributed alternative to MLPerf or closed MLOps benchmarks.

This is **inference infrastructure engineering** at research quality.

## Manually creating report with python pip install

```
pip install pandas python-docx matplotlib seaborn numpy
```

---

## License

Apache 2

---
