package main

import (
	"bytes"
	"crypto/md5"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/openfluke/paragon/v3"
)

type TelemetrySource string

const (
	SourceNative    TelemetrySource = "native"
	SourceWASMBun   TelemetrySource = "wasm-bun"
	SourceWASMIonic TelemetrySource = "wasm-ionic"
)

type TelemetryReport struct {
	Version    string          `json:"version"` // schema version
	Source     TelemetrySource `json:"source"`  // native | wasm-bun | wasm-ionic
	MachineID  string          `json:"machine_id"`
	System     SystemInfo      `json:"system_info"`
	FromHost   string          `json:"from_host"` // http://ip:port of the model host
	ModelsUsed []string        `json:"models_used"`
	Samples    []int           `json:"samples"` // digits 0..9 used (first index per digit)
	StartedAt  time.Time       `json:"started_at"`
	EndedAt    time.Time       `json:"ended_at"`
	Notes      string          `json:"notes,omitempty"`
	PerModel   []ModelRun      `json:"per_model"`
}

type ModelRun struct {
	ModelFile        string            `json:"model_file"`
	WebGPUInitOK     bool              `json:"webgpu_init_ok"`
	WebGPUInitTimeMS float64           `json:"webgpu_init_time_ms"`
	CPU              []SampleTiming    `json:"cpu"` // per digit
	GPU              []SampleTiming    `json:"gpu"` // per digit (may be CPU fallback if GPU init failed)
	Drift            []DriftMetrics    `json:"drift"`
	ADHD10           ADHDScore         `json:"adhd10"`            // buckets + per-sample labels + summary across the 10 fixed samples
	Summary          map[string]any    `json:"summary,omitempty"` // extra roll-ups if you want later
	Meta             map[string]string `json:"meta,omitempty"`    // extra tags
}

type SampleTiming struct {
	Digit     int       `json:"digit"`
	Idx       int       `json:"idx"`
	ElapsedMS float64   `json:"elapsed_ms"`
	Pred      int       `json:"pred"`
	Top1Score float64   `json:"top1_score"`
	Output    []float64 `json:"output"` // exact output vector for this sample (rounded)
}

type DriftMetrics struct {
	Digit  int     `json:"digit"`
	Idx    int     `json:"idx"`
	MaxAbs float64 `json:"max_abs"`
	MAE    float64 `json:"mae"`
}

// --- ADHD buckets & per-sample labels ---

type ADHDScore struct {
	Top1AccuracyCPU    float64 `json:"top1_accuracy_cpu"`
	Top1AccuracyGPU    float64 `json:"top1_accuracy_gpu"`
	CPUvsGPUAgreeCount int     `json:"cpu_vs_gpu_agree_count"`
	AvgDriftMAE        float64 `json:"avg_drift_mae"`
	MaxDriftMaxAbs     float64 `json:"max_drift_max_abs"`

	// Bucket roll-ups for strict 1:1 device/model comparison
	Buckets   ADHDBuckets  `json:"buckets"`
	PerSample []ADHDSample `json:"per_sample"`
}

type ADHDBuckets struct {
	CPUCorrect int `json:"cpu_correct"`
	CPUWrong   int `json:"cpu_wrong"`
	GPUCorrect int `json:"gpu_correct"`
	GPUWrong   int `json:"gpu_wrong"`

	// Nuance buckets (MNIST often has near-misses)
	CPUOffBy1 int `json:"cpu_off_by_1"`
	GPUOffBy1 int `json:"gpu_off_by_1"`

	// CPU/GPU prediction agreement
	Agree    int `json:"cpu_gpu_agree"`
	Disagree int `json:"cpu_gpu_disagree"`
}

type ADHDSample struct {
	Digit     int    `json:"digit"`
	Idx       int    `json:"idx"`
	CPUPred   int    `json:"cpu_pred"`
	GPUPred   int    `json:"gpu_pred"`
	CPUBucket string `json:"cpu_bucket"` // "correct" | "off_by_1" | "wrong"
	GPUBucket string `json:"gpu_bucket"` // "
	Agreement string `json:"agreement"`  // "agree" | "disagree"
}

type modelManifest struct {
	ID       string `json:"id"`
	Filename string `json:"filename"`
}

// --- MNIST ensure/download helpers ---

var mnistFiles = []string{
	"train-images-idx3-ubyte",
	"train-labels-idx1-ubyte",
	"t10k-images-idx3-ubyte",
	"t10k-labels-idx1-ubyte",
}

func ensureLocalMNIST(hostBase string) error {
	localDir := filepath.Join("public", "mnist")
	if err := os.MkdirAll(localDir, 0755); err != nil {
		return err
	}
	// If all files already exist, we're done.
	allPresent := true
	for _, fn := range mnistFiles {
		if _, err := os.Stat(filepath.Join(localDir, fn)); err != nil {
			allPresent = false
			break
		}
	}
	if allPresent {
		return nil
	}
	// Pull each missing file from host /mnist/<name>
	base := strings.TrimRight(hostBase, "/") + "/mnist"
	for _, fn := range mnistFiles {
		dst := filepath.Join(localDir, fn)
		if _, err := os.Stat(dst); err == nil {
			continue
		}
		src := base + "/" + fn
		if err := httpDownload(src, dst); err != nil {
			return fmt.Errorf("mnist download failed: %s -> %s: %w", src, dst, err)
		}
	}
	return nil
}

// ---- public API ----

// Pull models from host, run telemetry, save local JSON, and push back.
func RunTelemetryPipeline(hostBase string, source TelemetrySource) (string, error) {
	// 1) fetch manifest and download models
	modelDirLocal := filepath.Join("public", "models_remote")
	if err := os.MkdirAll(modelDirLocal, 0755); err != nil {
		return "", err
	}

	manifest, err := fetchManifest(hostBase)
	if err != nil {
		return "", fmt.Errorf("fetch manifest: %w", err)
	}
	if len(manifest) == 0 {
		return "", fmt.Errorf("manifest empty at %s", hostBase)
	}

	var modelFiles []string
	for _, m := range manifest {
		if m.Filename == "" {
			continue
		}
		url := strings.TrimRight(hostBase, "/") + "/models/" + m.Filename
		dst := filepath.Join(modelDirLocal, m.Filename)
		if err := httpDownload(url, dst); err != nil {
			return "", fmt.Errorf("download %s: %w", m.Filename, err)
		}
		modelFiles = append(modelFiles, dst)
	}

	// 2) collect system info & machine id
	sys := Collect()
	machineID := hashSystemInfo(sys)

	// 2.5) ensure MNIST exists locally (pull from host if needed)
	if err := ensureLocalMNIST(hostBase); err != nil {
		return "", fmt.Errorf("ensure mnist: %w", err)
	}

	// 3) prepare samples: first index per digit (0..9)
	images, labels, err := loadMNISTData("./public/mnist")
	if err != nil {
		return "", fmt.Errorf("load mnist: %w", err)
	}
	firstIdx := firstIndexPerDigit(labels)
	var digits []int
	for d := 0; d <= 9; d++ {
		digits = append(digits, d)
	}

	// 4) run for each model
	start := time.Now()
	var per []ModelRun
	for _, mf := range modelFiles {
		mr, err := runModelTelemetry(mf, images, firstIdx)
		if err != nil {
			fmt.Printf("⚠️  model %s: %v\n", filepath.Base(mf), err)
			continue
		}
		// ADHD-style: buckets + per-sample labels + summary across the 10 fixed samples
		mr.ADHD10 = computeADHD10(mr)
		per = append(per, mr)
	}
	end := time.Now()

	report := TelemetryReport{
		Version:    "1.2.0",
		Source:     source,
		MachineID:  machineID,
		System:     sys,
		FromHost:   hostBase,
		ModelsUsed: baseNames(modelFiles),
		Samples:    digits,
		StartedAt:  start.UTC(),
		EndedAt:    end.UTC(),
		PerModel:   per,
	}

	// 5) save locally
	outDir := filepath.Join("public", "reports_local")
	if err := os.MkdirAll(outDir, 0755); err != nil {
		return "", err
	}
	fn := fmt.Sprintf("telemetry_%s_%d.json", machineID, time.Now().Unix())
	localPath := filepath.Join(outDir, fn)
	if err := writeJSON(localPath, report); err != nil {
		return "", err
	}

	// 6) push back to host (multipart POST /upload)
	if err := uploadFile(hostBase, localPath, fn); err != nil {
		return "", fmt.Errorf("push report: %w", err)
	}

	return localPath, nil
}

// ---- internals ----

func runModelTelemetry(modelPath string, images [][][]float64, firstIdx map[int]int) (ModelRun, error) {
	// Load saved network (float32)
	loaded, err := paragon.LoadNamedNetworkFromJSONFile(modelPath)
	if err != nil {
		return ModelRun{}, fmt.Errorf("load: %w", err)
	}
	tmp, ok := loaded.(*paragon.Network[float32])
	if !ok {
		return ModelRun{}, fmt.Errorf("not float32: %T", loaded)
	}

	// Rebuild fresh network to ensure GPU-safe buffers
	shapes := make([]struct{ Width, Height int }, len(tmp.Layers))
	acts := make([]string, len(tmp.Layers))
	trains := make([]bool, len(tmp.Layers))
	for i, L := range tmp.Layers {
		shapes[i] = struct{ Width, Height int }{L.Width, L.Height}
		a := "linear"
		if L.Height > 0 && L.Width > 0 && L.Neurons[0][0] != nil {
			a = L.Neurons[0][0].Activation
		}
		acts[i], trains[i] = a, true
	}
	nnCPU, _ := paragon.NewNetwork[float32](shapes, acts, trains)
	state, _ := tmp.MarshalJSONModel()
	_ = nnCPU.UnmarshalJSONModel(state)

	// Clone for GPU
	nnGPU, _ := paragon.NewNetwork[float32](shapes, acts, trains)
	_ = nnGPU.UnmarshalJSONModel(state)
	nnGPU.WebGPUNative = true

	var gpuInitOK bool
	startInit := time.Now()
	if err := nnGPU.InitializeOptimizedGPU(); err != nil {
		gpuInitOK = false
		nnGPU.WebGPUNative = false
	} else {
		gpuInitOK = true
		// warmup cost once (pick any sample)
		if idx, ok := firstIdx[0]; ok {
			nnGPU.Forward(images[idx])
			_ = nnGPU.ExtractOutput()
		}
		defer nnGPU.CleanupOptimizedGPU()
	}
	initMS := float64(time.Since(startInit).Microseconds()) / 1000.0

	// per-digit timings and drift
	var cpuTimes []SampleTiming
	var gpuTimes []SampleTiming
	var drift []DriftMetrics

	for d := 0; d <= 9; d++ {
		idx, ok := firstIdx[d]
		if !ok {
			continue
		}
		sample := images[idx]

		// CPU
		startCPU := time.Now()
		nnCPU.Forward(sample)
		outCPU := nnCPU.ExtractOutput()
		elapsedCPU := float64(time.Since(startCPU).Microseconds()) / 1000.0

		// GPU (or CPU fallback if GPU init failed)
		startGPU := time.Now()
		nnGPU.Forward(sample)
		outGPU := nnGPU.ExtractOutput()
		elapsedGPU := float64(time.Since(startGPU).Microseconds()) / 1000.0

		cpuTimes = append(cpuTimes, SampleTiming{
			Digit: d, Idx: idx, ElapsedMS: elapsedCPU,
			Pred: argmax64(outCPU), Top1Score: top1(outCPU),
			Output: roundSlice(outCPU, 6),
		})
		gpuTimes = append(gpuTimes, SampleTiming{
			Digit: d, Idx: idx, ElapsedMS: elapsedGPU,
			Pred: argmax64(outGPU), Top1Score: top1(outGPU),
			Output: roundSlice(outGPU, 6),
		})

		mx, mae := driftMaxAndMAE(outCPU, outGPU)
		drift = append(drift, DriftMetrics{Digit: d, Idx: idx, MaxAbs: mx, MAE: mae})
	}

	return ModelRun{
		ModelFile:        filepath.Base(modelPath),
		WebGPUInitOK:     gpuInitOK,
		WebGPUInitTimeMS: initMS,
		CPU:              cpuTimes,
		GPU:              gpuTimes,
		Drift:            drift,
	}, nil
}

func firstIndexPerDigit(labels [][][]float64) map[int]int {
	firstIdx := make(map[int]int)
	for i, lbl := range labels {
		for d, v := range lbl[0] {
			if v == 1.0 {
				if _, seen := firstIdx[d]; !seen {
					firstIdx[d] = i
				}
				break
			}
		}
	}
	return firstIdx
}

func top1(out []float64) float64 {
	if len(out) == 0 {
		return 0
	}
	best := out[0]
	for i := 1; i < len(out); i++ {
		if out[i] > best {
			best = out[i]
		}
	}
	return best
}

// ADHD-style buckets + per-sample labels over the 10 fixed samples
func computeADHD10(m ModelRun) ADHDScore {
	if len(m.CPU) == 0 || len(m.GPU) == 0 || len(m.Drift) == 0 {
		return ADHDScore{}
	}

	var accCPU, accGPU float64
	var agreeCount int
	var sumMAE, maxMaxAbs float64
	n := 0

	var buckets ADHDBuckets
	per := make([]ADHDSample, 0, len(m.CPU))

	for i := range m.CPU {
		c := m.CPU[i]
		g := m.GPU[i]
		d := m.Drift[i]

		// correctness vs ground truth label
		cCorrect := (c.Pred == c.Digit)
		gCorrect := (g.Pred == g.Digit)
		if cCorrect {
			accCPU += 1
			buckets.CPUCorrect++
		} else {
			buckets.CPUWrong++
		}
		if gCorrect {
			accGPU += 1
			buckets.GPUCorrect++
		} else {
			buckets.GPUWrong++
		}

		// nuance: off-by-1
		if absInt(c.Pred-c.Digit) == 1 {
			buckets.CPUOffBy1++
		}
		if absInt(g.Pred-g.Digit) == 1 {
			buckets.GPUOffBy1++
		}

		// agreement between CPU/GPU predictions
		agree := (c.Pred == g.Pred)
		if agree {
			agreeCount++
		} else {
			buckets.Disagree++
		}
		buckets.Agree = agreeCount // keep in sync

		// drift rollups
		sumMAE += d.MAE
		if d.MaxAbs > maxMaxAbs {
			maxMaxAbs = d.MaxAbs
		}

		// per-sample bucket labels for exact 1:1 diffs
		per = append(per, ADHDSample{
			Digit:     c.Digit,
			Idx:       c.Idx,
			CPUPred:   c.Pred,
			GPUPred:   g.Pred,
			CPUBucket: labelBucket(c.Pred, c.Digit),
			GPUBucket: labelBucket(g.Pred, g.Digit),
			Agreement: ternary(agree, "agree", "disagree"),
		})

		n++
	}

	return ADHDScore{
		Top1AccuracyCPU:    safeDiv(accCPU, float64(n)),
		Top1AccuracyGPU:    safeDiv(accGPU, float64(n)),
		CPUvsGPUAgreeCount: agreeCount,
		AvgDriftMAE:        safeDiv(sumMAE, float64(n)),
		MaxDriftMaxAbs:     maxMaxAbs,
		Buckets:            buckets,
		PerSample:          per,
	}
}

func absInt(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func labelBucket(pred, label int) string {
	if pred == label {
		return "correct"
	}
	if absInt(pred-label) == 1 {
		return "off_by_1"
	}
	return "wrong"
}

func ternary[T any](cond bool, a, b T) T {
	if cond {
		return a
	}
	return b
}

// stable machine ID from normalized SystemInfo
func hashSystemInfo(si SystemInfo) string {
	clone := si
	clone.GPUModel = strings.ToLower(clone.GPUModel)
	clone.CPUModel = strings.ToLower(clone.CPUModel)
	b, _ := json.Marshal(clone)
	sum := md5.Sum(b)
	return hex.EncodeToString(sum[:])
}

// ---- math/util helpers ----

func safeDiv(a, b float64) float64 {
	if b == 0 {
		return 0
	}
	return a / b
}

func roundSlice(xs []float64, places int) []float64 {
	if xs == nil {
		return nil
	}
	scale := pow10(places)
	out := make([]float64, len(xs))
	for i, v := range xs {
		// round half away from zero-ish
		if v >= 0 {
			out[i] = float64(int64(v*scale+0.5)) / scale
		} else {
			out[i] = float64(int64(v*scale-0.5)) / scale
		}
	}
	return out
}

func pow10(n int) float64 {
	p := 1.0
	for i := 0; i < n; i++ {
		p *= 10
	}
	return p
}

// ---- HTTP helpers ----

func fetchManifest(hostBase string) ([]modelManifest, error) {
	u := strings.TrimRight(hostBase, "/") + "/models/manifest.json"
	resp, err := http.Get(u)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("status %s from %s", resp.Status, u)
	}
	var manifest []modelManifest
	if err := json.NewDecoder(resp.Body).Decode(&manifest); err != nil {
		return nil, err
	}
	return manifest, nil
}

func httpDownload(url, dst string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("GET %s: %s", url, resp.Status)
	}
	if err := os.MkdirAll(filepath.Dir(dst), 0755); err != nil {
		return err
	}
	f, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = io.Copy(f, resp.Body)
	return err
}

func uploadFile(hostBase, path, name string) error {
	u := strings.TrimRight(hostBase, "/") + "/upload"

	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)
	fw, err := w.CreateFormFile("file", filepath.Base(name))
	if err != nil {
		return err
	}
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	if _, err := io.Copy(fw, f); err != nil {
		return err
	}
	_ = w.WriteField("name", name)
	_ = w.Close()

	req, err := http.NewRequest(http.MethodPost, u, &buf)
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", w.FormDataContentType())

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("upload failed: %s — %s", resp.Status, strings.TrimSpace(string(body)))
	}
	return nil
}

func baseNames(paths []string) []string {
	out := make([]string, 0, len(paths))
	for _, p := range paths {
		out = append(out, filepath.Base(p))
	}
	return out
}
