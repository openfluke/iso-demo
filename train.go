package main

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/openfluke/paragon/v3"
)

func runTrainMenu() {
	modelDir := filepath.Join("public", "models")

	entries, _ := os.ReadDir(modelDir)
	models := []string{}
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") || e.Name() == "manifest.json" {
			continue
		}
		models = append(models, e.Name())
	}
	if len(models) == 0 {
		fmt.Println("❌ No models found in public/models/")
		return
	}

	fmt.Println("\nAvailable models:")
	for i, m := range models {
		fmt.Printf("%d) %s\n", i+1, m)
	}
	fmt.Println("0) Back")

	reader := bufio.NewReader(os.Stdin)
	fmt.Print("Select model: ")
	choiceRaw, _ := reader.ReadString('\n')
	choice := strings.TrimSpace(choiceRaw)
	if choice == "0" {
		return
	}
	idx, err := strconv.Atoi(choice)
	if err != nil || idx < 1 || idx > len(models) {
		fmt.Println("❌ Invalid choice")
		return
	}

	modelPath := filepath.Join(modelDir, models[idx-1])
	fmt.Printf("\n▶ Training %s for 1 epoch\n", models[idx-1])
	trainOneEpoch(modelPath)
}

func trainOneEpoch(modelPath string) {
	// 1) Load MNIST
	images, labels, err := loadMNISTData("./public/mnist")
	if err != nil {
		fmt.Println("❌ Failed to load MNIST:", err)
		return
	}
	trainInputs, trainTargets, testInputs, testTargets := paragon.SplitDataset(images, labels, 0.8)

	fmt.Printf("\n📦 Model: %s\n", modelPath)

	// 2) Load the saved model (type-aware) to discover topology/weights
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

	// 3) Rebuild a fresh network with identical shapes/acts (so runtime/GPU metadata is correct)
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

	nn, err := paragon.NewNetwork[float32](shapes, acts, trains)
	if err != nil {
		fmt.Printf("❌ NewNetwork failed: %v\n", err)
		return
	}
	state, _ := tmp.MarshalJSONModel()
	if err := nn.UnmarshalJSONModel(state); err != nil {
		fmt.Printf("❌ UnmarshalJSONModel failed: %v\n", err)
		return
	}

	// 4) Initialize WebGPU once (same pattern as compare); warmup forward to pay JIT cost
	nn.WebGPUNative = true
	nn.Debug = false
	startGPU := time.Now()
	if err := nn.InitializeOptimizedGPU(); err != nil {
		fmt.Printf("⚠️ WebGPU init failed: %v\n   Falling back to CPU for training/inference.\n", err)
		nn.WebGPUNative = false
	} else {
		fmt.Println("✅ WebGPU initialized successfully")
		// Warmup once so later forwards during/after training are consistent
		if len(trainInputs) > 0 {
			nn.Forward(trainInputs[0])
			_ = nn.ExtractOutput()
		}
		defer nn.CleanupOptimizedGPU()
	}
	fmt.Printf("⏱ WebGPU Init Time: %v\n", time.Since(startGPU))

	// 5) Train for 1 epoch (CPU or GPU depending on your engine’s Train path)
	//    (If Train is CPU-only in your build, this still keeps GPU state valid for comparisons after.)
	lr := float64(0.01)
	fmt.Println("🧠 Training for 1 epoch...")
	start := time.Now()
	nn.Train(trainInputs, trainTargets, 1, lr, false, float32(2), float32(-2))
	fmt.Printf("⏱ Training completed in %v\n", time.Since(start))

	// 6) Quick sanity accuracy on 100 test samples
	N := 100
	if len(testInputs) < N {
		N = len(testInputs)
	}
	correct := 0
	for i := 0; i < N; i++ {
		nn.Forward(testInputs[i])
		out := nn.ExtractOutput()
		if argmax64(out) == argmax64(testTargets[i][0]) {
			correct++
		}
	}
	fmt.Printf("🎯 Quick test accuracy (%d samples): %.2f%%\n", N, 100*float64(correct)/float64(N))

	// 7) Save updated weights back to disk
	if err := nn.SaveJSON(modelPath); err != nil {
		fmt.Printf("❌ Failed to save trained model: %v\n", err)
	} else {
		fmt.Printf("💾 Updated model saved to %s\n", modelPath)
	}
}
