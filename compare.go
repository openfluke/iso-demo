// compare.go
package main

import (
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/openfluke/paragon/v3"
)

func formatAll(out []float64) string {
	parts := make([]string, len(out))
	for i, v := range out {
		parts[i] = fmt.Sprintf("%d:%.4f", i, v)
	}
	return "[" + strings.Join(parts, ", ") + "]"
}

func compareSingleModel(modelPath string) {
	// Load MNIST once
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

	// Load once (type-aware), then rebuild fresh topology
	loaded, err := paragon.LoadNamedNetworkFromJSONFile(modelPath)
	if err != nil {
		fmt.Printf("❌ Load failed: %v\n", err)
		return
	}
	tmp, ok := loaded.(*paragon.Network[float32])
	if !ok {
		fmt.Printf("⚠️ Skipping (not float32): %T\n", loaded)
		return
	}

	// Derive shapes/acts
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
	state, _ := tmp.MarshalJSONModel()

	// Build CPU once
	nnCPU, _ := paragon.NewNetwork[float32](shapes, acts, trains)
	_ = nnCPU.UnmarshalJSONModel(state)
	nnCPU.WebGPUNative = false

	// Build GPU once
	nnGPU, _ := paragon.NewNetwork[float32](shapes, acts, trains)
	_ = nnGPU.UnmarshalJSONModel(state)
	nnGPU.WebGPUNative = true
	startInit := time.Now()
	if err := nnGPU.InitializeOptimizedGPU(); err != nil {
		fmt.Printf("⚠️ GPU init failed: %v\n   Falling back to CPU-only compare.\n", err)
		nnGPU.WebGPUNative = false
	} else {
		fmt.Printf("✅ WebGPU initialized in %v\n", time.Since(startInit))
		// Warmup to pay JIT/pipeline cost once
		if idx, ok := firstIdx[0]; ok {
			nnGPU.Forward(images[idx])
			_ = nnGPU.ExtractOutput()
		}
	}

	// Run digits 0..9
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
		elapsedCPU := time.Since(startCPU)
		predCPU := argmax64(outCPU)

		// GPU (may be CPU fallback if init failed)
		startGPU := time.Now()
		nnGPU.Forward(sample)
		outGPU := nnGPU.ExtractOutput()
		elapsedGPU := time.Since(startGPU)
		predGPU := argmax64(outGPU)

		maxAbs, mae := driftMaxAndMAE(outCPU, outGPU)

		fmt.Printf(
			"Digit %d (idx=%d)\n   CPU pred=%d %s ⏱ %v\n   GPU pred=%d %s ⏱ %v\n   drift_max=%.6f mae=%.6f\n",
			d, idx,
			predCPU, formatAll(outCPU), elapsedCPU,
			predGPU, formatAll(outGPU), elapsedGPU,
			maxAbs, mae,
		)
	}

	if nnGPU.WebGPUNative {
		nnGPU.CleanupOptimizedGPU()
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

func formatTopK(p []float64, k int) string {
	type pair struct {
		i int
		v float64
	}
	ps := make([]pair, len(p))
	for i := range p {
		ps[i] = pair{i, p[i]}
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
