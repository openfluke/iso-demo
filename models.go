package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/openfluke/paragon/v3"
)

type ModelSpec struct {
	ID        string   `json:"id"`
	Layers    []string `json:"layers"`      // e.g., ["784","64","10"]
	Activs    []string `json:"activations"` // e.g., ["linear","relu","softmax"]
	Trainable []bool   `json:"trainable"`
	Filename  string   `json:"filename"` // output filename
	Bytes     int64    `json:"bytes"`    // file size after save
	Params    int64    `json:"params"`   // optional: filled if paragon exposes it
}

func createModelZoo() {
	start := time.Now()

	// 1) Ensure output dir
	modelDir := MustPublicPath("models")

	fmt.Printf("📂 Model directory: %s\n", modelDir)

	if err := os.MkdirAll(modelDir, 0755); err != nil {
		fmt.Printf("❌ Failed to create model dir: %v\n", err)
		return
	}

	// 2) Define all MNIST-shape architectures (28*28=784 input → ... → 10 output)
	specs := []ModelSpec{
		{ID: "S1", Layers: []string{"784", "64", "10"}},
		{ID: "S2", Layers: []string{"784", "128", "10"}},
		{ID: "S3", Layers: []string{"784", "256", "10"}},
		{ID: "M1", Layers: []string{"784", "256", "256", "10"}},
		{ID: "M2", Layers: []string{"784", "384", "384", "10"}},
		{ID: "M3", Layers: []string{"784", "512", "512", "10"}},
		{ID: "L1", Layers: []string{"784", "768", "768", "768", "10"}},
		{ID: "L2", Layers: []string{"784", "1024", "1024", "1024", "10"}},
		{ID: "XL1", Layers: []string{"784", "1536", "1536", "1536", "1536", "10"}},
		{ID: "XL2", Layers: []string{"784", "2048", "2048", "2048", "2048", "10"}},
	}

	// helper to build Paragon shapes from Layers
	toParagonShapes := func(s ModelSpec) []struct{ Width, Height int } {
		// Represent as [in] [hidden...] [out], using Height as 1 except input 28x28.
		// Paragon’s example you showed used {28,28}, {N,N?}, {10,1}. We’ll keep height=1 for dense.
		shapes := make([]struct{ Width, Height int }, 0, len(s.Layers))
		for i, l := range s.Layers {
			switch i {
			case 0:
				// input = 28x28
				shapes = append(shapes, struct{ Width, Height int }{28, 28})
			case len(s.Layers) - 1:
				// output = 10x1
				shapes = append(shapes, struct{ Width, Height int }{10, 1})
			default:
				// hidden: Nx1
				var w int
				fmt.Sscanf(l, "%d", &w)
				shapes = append(shapes, struct{ Width, Height int }{w, 1})
			}
		}
		return shapes
	}

	// same activations for all: linear → relu...(for hidden)... → softmax
	buildActivs := func(s ModelSpec) []string {
		acts := make([]string, 0, len(s.Layers))
		for i := range s.Layers {
			if i == 0 {
				acts = append(acts, "linear") // input pass-through
			} else if i == len(s.Layers)-1 {
				acts = append(acts, "softmax")
			} else {
				acts = append(acts, "relu")
			}
		}
		return acts
	}

	buildTrainable := func(n int) []bool {
		tb := make([]bool, n)
		for i := range tb {
			tb[i] = true
		}
		return tb
	}

	manifest := make([]ModelSpec, 0, len(specs))

	for _, base := range specs {
		spec := base
		spec.Activs = buildActivs(spec)
		spec.Trainable = buildTrainable(len(spec.Layers))
		spec.Filename = fmt.Sprintf("mnist_%s.json", spec.ID)
		outPath := filepath.Join(modelDir, spec.Filename)

		// Skip if exists
		if _, err := os.Stat(outPath); err == nil {
			fi, _ := os.Stat(outPath)
			spec.Bytes = fi.Size()
			manifest = append(manifest, spec)
			fmt.Printf("⚠️  %s already exists (%s), skipping\n", spec.ID, outPath)
			continue
		}

		// Build & save
		startInit := time.Now()
		nn, err := paragon.NewNetwork[float32](toParagonShapes(spec), spec.Activs, spec.Trainable)
		if err != nil {
			fmt.Printf("❌ %s init failed: %v\n", spec.ID, err)
			continue
		}

		fmt.Printf("⏱ %s init: %v\n", spec.ID, time.Since(startInit))

		startSave := time.Now()
		if err := nn.SaveJSON(outPath); err != nil {
			fmt.Printf("❌ %s save failed: %v\n", spec.ID, err)
			continue
		}
		saveDur := time.Since(startSave)

		fi, _ := os.Stat(outPath)
		spec.Bytes = fi.Size()
		manifest = append(manifest, spec)
		fmt.Printf("💾 %s saved → %s (%d bytes) in %v\n", spec.ID, outPath, spec.Bytes, saveDur)
	}

	// 3) Write manifest
	manPath := filepath.Join(modelDir, "manifest.json")
	if err := writeJSON(manPath, manifest); err != nil {
		fmt.Printf("❌ manifest write failed: %v\n", err)
	} else {
		fmt.Printf("📜 manifest written → %s\n", manPath)
	}

	fmt.Printf("✅ Model zoo ready in %v\n", time.Since(start))
}

func writeJSON(path string, v any) error {
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	if err := enc.Encode(v); err != nil {
		f.Close()
		return err
	}
	f.Close()
	return os.Rename(tmp, path)
}

// ---- Benchmark: run first 0–9 samples through every saved model ----
func benchmarkModelsOnDigits(withGpu bool) {
	modelDir := MustPublicPath("models")

	// Load dataset once
	images, labels, err := loadMNISTData(MustPublicPath("mnist"))
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

	entries, err := os.ReadDir(modelDir)
	if err != nil {
		fmt.Println("❌ Failed to read models dir:", err)
		return
	}

	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") || e.Name() == "manifest.json" {
			continue
		}
		modelPath := filepath.Join(modelDir, e.Name())
		fmt.Printf("\n📦 Model: %s\n", e.Name())

		// 1) Load into a temp network (type-aware) so we can discover shapes/acts
		loaded, err := paragon.LoadNamedNetworkFromJSONFile(modelPath)
		if err != nil {
			fmt.Printf("❌ Load failed: %v\n", err)
			continue
		}

		tmp, ok := loaded.(*paragon.Network[float32])
		if !ok {
			fmt.Printf("⚠️ %s is not float32, skipping\n", e.Name())
			continue
		}

		// 2) Build a fresh network with the same topology using NewNetwork
		shapes := make([]struct{ Width, Height int }, len(tmp.Layers))
		acts := make([]string, len(tmp.Layers))
		train := make([]bool, len(tmp.Layers))
		for i, L := range tmp.Layers {
			shapes[i] = struct{ Width, Height int }{L.Width, L.Height}
			// assume activation consistent per layer; read first neuron
			a := "linear"
			if L.Height > 0 && L.Width > 0 && L.Neurons[0][0] != nil {
				a = L.Neurons[0][0].Activation
			}
			acts[i], train[i] = a, true
		}
		nn, err := paragon.NewNetwork[float32](shapes, acts, train)
		if err != nil {
			fmt.Printf("❌ NewNetwork failed: %v\n", err)
			continue
		}
		nn.TypeName = "float32"

		// 3) Copy weights/biases from tmp into this fresh, fully-initialized net
		bytesJSON, err := tmp.MarshalJSONModel()
		if err != nil {
			fmt.Printf("❌ MarshalJSONModel failed: %v\n", err)
			continue
		}
		if err := nn.UnmarshalJSONModel(bytesJSON); err != nil {
			fmt.Printf("❌ UnmarshalJSONModel failed: %v\n", err)
			continue
		}

		// 4) Optional GPU path (now safe because nn came from NewNetwork)
		if withGpu {
			nn.WebGPUNative, nn.Debug = true, false
			startGPU := time.Now()
			if err := nn.InitializeOptimizedGPU(); err != nil {
				fmt.Printf("⚠️ WebGPU init failed: %v\n   Falling back to CPU…\n", err)
				nn.WebGPUNative = false
			} else {
				fmt.Println("✅ WebGPU initialized")
			}
			fmt.Printf("⏱ WebGPU Init Time: %v\n", time.Since(startGPU))
			// ensure cleanup per model
			if nn.WebGPUNative {
				defer nn.CleanupOptimizedGPU()
			}
		}

		// 5) Run digits 0..9 (28×28 input — no flattening)
		for d := 0; d <= 9; d++ {
			idx, ok := firstIdx[d]
			if !ok {
				fmt.Printf("⚠️ No sample for digit %d\n", d)
				continue
			}

			start := time.Now()
			nn.Forward(images[idx])   // [][]float64, shape 28x28
			out := nn.ExtractOutput() // []float64
			elapsed := time.Since(start)

			pred := argmax64(out)
			fmt.Printf("Digit %d → pred=%d ⏱ %v\n", d, pred, elapsed)
		}
	}
}
