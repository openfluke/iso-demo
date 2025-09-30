// compare.go
package main

import (
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/openfluke/paragon/v3"
)

// Compare CPU vs GPU on the first available sample of each digit (0–9)
// for a single chosen model JSON. Prints predictions with top-3 probs,
// per-run timings, and drift (max abs + MAE).
func compareSingleModel(modelPath string) {
	// Load dataset
	images, labels, err := loadMNISTData("./public/mnist")
	if err != nil {
		fmt.Println("❌ Failed to load MNIST:", err)
		return
	}

	// First index for each digit 0..9
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

	fmt.Printf("\n📦 Model: %s\n", modelPath)

	// Load once (type-aware) to discover topology and weights
	loaded, err := paragon.LoadNamedNetworkFromJSONFile(modelPath)
	if err != nil {
		fmt.Printf("❌ Load failed: %v\n", err)
		return
	}
	tmp, ok := loaded.(*paragon.Network[float32])
	if !ok {
		fmt.Printf("⚠️ Skipping (not float32) %T\n", loaded)
		return
	}

	// Rebuild topology fresh (ensures runtime & GPU metadata are properly initialized)
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
	base, _ := tmp.MarshalJSONModel()

	// We'll lazy-init GPU once; for each digit we rehydrate a fresh instance
	var gpuUsable bool
	var gpuInitErr error

	for d := 0; d <= 9; d++ {
		idx, ok := firstIdx[d]
		if !ok {
			continue
		}
		sample := images[idx] // [][]float64 (28x28)

		// --- CPU run ---
		nnCPU, _ := paragon.NewNetwork[float32](shapes, acts, trains)
		_ = nnCPU.UnmarshalJSONModel(base)
		nnCPU.WebGPUNative = false

		startCPU := time.Now()
		nnCPU.Forward(sample)
		outCPU := nnCPU.ExtractOutput()
		elapsedCPU := time.Since(startCPU)
		predCPU := argmax64(outCPU)

		// --- GPU run ---
		nnGPU, _ := paragon.NewNetwork[float32](shapes, acts, trains)
		_ = nnGPU.UnmarshalJSONModel(base)

		// Initialize GPU once (first digit); reuse "usable" flag afterward
		if !gpuUsable && gpuInitErr == nil {
			nnGPU.WebGPUNative = true
			gpuInitErr = nnGPU.InitializeOptimizedGPU()
			if gpuInitErr != nil {
				fmt.Printf("⚠️ GPU init failed: %v\n", gpuInitErr)
				nnGPU.WebGPUNative = false
			} else {
				// Adapter details are already logged by wgpu; nothing extra here
				gpuUsable = true
			}
		} else {
			nnGPU.WebGPUNative = gpuUsable
			if gpuUsable {
				// Re-init buffers/shaders for this instance
				if err := nnGPU.InitializeOptimizedGPU(); err != nil {
					fmt.Printf("⚠️ GPU re-init failed: %v\n", err)
					nnGPU.WebGPUNative = false
				}
			}
		}

		startGPU := time.Now()
		nnGPU.Forward(sample)
		outGPU := nnGPU.ExtractOutput()
		elapsedGPU := time.Since(startGPU)
		if gpuUsable {
			nnGPU.CleanupOptimizedGPU()
		}
		predGPU := argmax64(outGPU)

		// Drift metrics
		maxAbs, mae := driftMaxAndMAE(outCPU, outGPU)

		// Top-3 pretty strings (change K to 10 if you want full vector)
		cpuTop := formatTopK(outCPU, 3)
		gpuTop := formatTopK(outGPU, 3)

		fmt.Printf(
			"Digit %d (idx=%d) → CPU pred=%d %s ⏱ %v | GPU pred=%d %s ⏱ %v | drift_max=%.6f mae=%.6f\n",
			d, idx, predCPU, cpuTop, elapsedCPU, predGPU, gpuTop, elapsedGPU, maxAbs, mae,
		)
	}
}

func driftMaxAndMAE(a, b []float64) (maxAbs float64, mae float64) {
	if len(a) == 0 || len(a) != len(b) {
		return 0, 0
	}
	sum := 0.0
	for i := range a {
		d := a[i] - b[i]
		if d < 0 {
			d = -d
		}
		sum += d
		if d > maxAbs {
			maxAbs = d
		}
	}
	mae = sum / float64(len(a))
	return
}

// formatTopK returns a compact string of the top-k classes with probs, e.g. "[8:0.72, 3:0.12, 5:0.08]"
func formatTopK(p []float64, k int) string {
	type pair struct {
		i int
		v float64
	}
	ps := make([]pair, len(p))
	for i := range p {
		ps[i] = pair{i: i, v: p[i]}
	}
	sort.Slice(ps, func(i, j int) bool { return ps[i].v > ps[j].v })

	if k > len(ps) {
		k = len(ps)
	}
	parts := make([]string, k)
	for i := 0; i < k; i++ {
		parts[i] = fmt.Sprintf("%d:%.4f", ps[i].i, ps[i].v)
	}
	return "[" + strings.Join(parts, ", ") + "]"
}
