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

func runEvaluateMenu() {
	modelDir := MustPublicPath("models")

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
	fmt.Printf("\n▶ Evaluating %s\n", models[idx-1])
	evaluateModelADHD(modelPath)
}

func evaluateModelADHD(modelPath string) {
	// Load dataset
	images, labels, err := loadMNISTData(MustPublicPath("mnist"))
	if err != nil {
		fmt.Println("❌ Failed to load MNIST:", err)
		return
	}
	trainInputs, trainTargets, testInputs, testTargets := paragon.SplitDataset(images, labels, 0.8)

	// Load saved network
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

	// Rebuild fresh network with correct shapes/acts
	shapes := make([]struct{ Width, Height int }, len(tmp.Layers))
	acts := make([]string, len(tmp.Layers))
	trains := make([]bool, len(tmp.Layers))
	for i, L := range tmp.Layers {
		shapes[i] = struct{ Width, Height int }{L.Width, L.Height}
		act := "linear"
		if L.Height > 0 && L.Width > 0 && L.Neurons[0][0] != nil {
			act = L.Neurons[0][0].Activation
		}
		acts[i], trains[i] = act, true
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

	// Initialize GPU
	nn.WebGPUNative = true
	startGPU := time.Now()
	if err := nn.InitializeOptimizedGPU(); err != nil {
		fmt.Printf("⚠️ WebGPU init failed: %v\n   Falling back to CPU.\n", err)
		nn.WebGPUNative = false
	} else {
		fmt.Println("✅ WebGPU initialized successfully")
		// Warm-up forward
		if len(trainInputs) > 0 {
			nn.Forward(trainInputs[0])
			_ = nn.ExtractOutput()
		}
		defer nn.CleanupOptimizedGPU()
	}
	fmt.Printf("⏱ WebGPU Init Time: %v\n", time.Since(startGPU))

	// Run ADHD evaluation
	fmt.Println("🧪 Evaluating on training set...")
	trainScore := evaluateFullNetwork(nn, trainInputs, trainTargets, "Train")

	fmt.Println("\n🧪 Evaluating on test set...")
	testScore := evaluateFullNetwork(nn, testInputs, testTargets, "Test")

	fmt.Printf("\n✅ Evaluation complete.\nTrain Score: %.4f%% | Test Score: %.4f%%\n", trainScore, testScore)
}

func evaluateFullNetwork[T paragon.Numeric](nn *paragon.Network[T], inputs, targets [][][]float64, dataset string) float64 {
	start := time.Now()
	expected := make([]float64, len(inputs))
	actual := make([]float64, len(inputs))

	for i := range inputs {
		nn.Forward(inputs[i])     // runs on GPU if enabled
		out := nn.ExtractOutput() // fetch prediction
		expected[i] = float64(paragon.ArgMax(targets[i][0]))
		actual[i] = float64(paragon.ArgMax(out))
	}

	nn.EvaluateModel(expected, actual)
	score := nn.Performance.Score

	// Print ADHD metrics
	fmt.Printf("\n📈 ADHD Performance (%s Set):\n", dataset)
	for name, bucket := range nn.Performance.Buckets {
		fmt.Printf("- %s: %d samples (%.2f%%)\n", name, bucket.Count, float64(bucket.Count)/float64(nn.Performance.Total)*100)
	}
	fmt.Printf("- Total Samples: %d\n", nn.Performance.Total)
	fmt.Printf("- Failures (100%%+): %d (%.2f%%)\n", nn.Performance.Failures, float64(nn.Performance.Failures)/float64(nn.Performance.Total)*100)
	fmt.Printf("- Score: %.4f%%\n", score)
	fmt.Printf("⏱ Evaluate Time (%s): %v\n", dataset, time.Since(start))

	return score
}
