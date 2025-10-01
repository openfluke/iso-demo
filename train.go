package main

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/openfluke/paragon/v3"
)

// ─────────────────────────── MENU ───────────────────────────

func runTrainMenu() {
	reader := bufio.NewReader(os.Stdin)
	modelDir := filepath.Join("public", "models")

	// Build model list
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

	// Mode: single or all
	fmt.Println("\nTrain what?")
	fmt.Println("1) Single model")
	fmt.Println("2) All models")
	fmt.Println("0) Back")
	fmt.Print("Select: ")
	modeRaw, _ := reader.ReadString('\n')
	mode := strings.TrimSpace(modeRaw)
	if mode == "0" {
		return
	}
	if mode != "1" && mode != "2" {
		fmt.Println("❌ Invalid choice")
		return
	}

	var chosen []string
	if mode == "1" {
		fmt.Println("\nAvailable models:")
		for i, m := range models {
			fmt.Printf("%d) %s\n", i+1, m)
		}
		fmt.Println("0) Back")
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
		chosen = []string{models[idx-1]}
	} else {
		chosen = models
	}

	// Strategy
	fmt.Println("\nTraining strategy:")
	fmt.Println("1) Train N epochs")
	fmt.Println("2) Train until ADHD score ≥ target% (with max epochs)")
	fmt.Println("0) Back")
	fmt.Print("Select: ")
	stratRaw, _ := reader.ReadString('\n')
	strat := strings.TrimSpace(stratRaw)
	if strat == "0" {
		return
	}
	if strat != "1" && strat != "2" {
		fmt.Println("❌ Invalid choice")
		return
	}

	// Hyperparams
	lr := 0.01
	fmt.Printf("Learning rate [default %.4f]: ", lr)
	if s, _ := reader.ReadString('\n'); strings.TrimSpace(s) != "" {
		if v, err := strconv.ParseFloat(strings.TrimSpace(s), 64); err == nil && v > 0 {
			lr = v
		}
	}

	var epochs int
	var target float64
	var maxEpochs int

	if strat == "1" {
		fmt.Print("Epochs (e.g., 1, 3, 10): ")
		eRaw, _ := reader.ReadString('\n')
		e := strings.TrimSpace(eRaw)
		ep, err := strconv.Atoi(e)
		if err != nil || ep < 1 {
			fmt.Println("❌ Invalid epochs")
			return
		}
		epochs = ep
	} else {
		fmt.Print("Target ADHD score percent (e.g., 70.0): ")
		tRaw, _ := reader.ReadString('\n')
		t := strings.TrimSpace(tRaw)
		tv, err := strconv.ParseFloat(t, 64)
		if err != nil || tv <= 0 || tv > 100 {
			fmt.Println("❌ Invalid target percent")
			return
		}
		target = tv

		fmt.Print("Max epochs (safety cap, e.g., 10): ")
		meRaw, _ := reader.ReadString('\n')
		me := strings.TrimSpace(meRaw)
		mep, err := strconv.Atoi(me)
		if err != nil || mep < 1 {
			fmt.Println("❌ Invalid max epochs")
			return
		}
		maxEpochs = mep
	}

	startAll := time.Now()
	for i, name := range chosen {
		modelPath := filepath.Join(modelDir, name)
		fmt.Printf("\n▶ [%d/%d] Training %s\n", i+1, len(chosen), name)

		var err error
		if strat == "1" {
			err = trainModelEpochs(modelPath, epochs, lr)
		} else {
			err = trainModelUntilScore(modelPath, target, maxEpochs, lr)
		}
		if err != nil {
			fmt.Printf("   ❌ %s: %v\n", name, err)
		}
	}
	fmt.Printf("\n✅ Training batch complete in %v\n", time.Since(startAll))
}

// ─────────────────────────── CORE ───────────────────────────

// silence stdout during f(); restores after
func withSilencedStdout(f func()) {
	old := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w
	done := make(chan struct{})
	go func() {
		_, _ = io.Copy(io.Discard, r)
		close(done)
	}()
	f()
	_ = w.Close()
	<-done
	os.Stdout = old
}

func loadFloat32Model(modelPath string) (*paragon.Network[float32], error) {
	loaded, err := paragon.LoadNamedNetworkFromJSONFile(modelPath)
	if err != nil {
		return nil, fmt.Errorf("load failed: %w", err)
	}
	tmp, ok := loaded.(*paragon.Network[float32])
	if !ok {
		return nil, fmt.Errorf("not float32: %T", loaded)
	}

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
		return nil, fmt.Errorf("NewNetwork failed: %w", err)
	}
	state, _ := tmp.MarshalJSONModel()
	if err := nn.UnmarshalJSONModel(state); err != nil {
		return nil, fmt.Errorf("UnmarshalJSONModel failed: %w", err)
	}
	return nn, nil
}

// quiet ADHD score: no printing
func evalADHDScore[T paragon.Numeric](nn *paragon.Network[T], inputs, targets [][][]float64) float64 {
	expected := make([]float64, len(inputs))
	actual := make([]float64, len(inputs))
	for i := range inputs {
		nn.Forward(inputs[i])
		out := nn.ExtractOutput()
		expected[i] = float64(paragon.ArgMax(targets[i][0]))
		actual[i] = float64(paragon.ArgMax(out))
	}
	nn.EvaluateModel(expected, actual)
	return nn.Performance.Score
}

func withGPU[T paragon.Numeric](nn *paragon.Network[T], warm [][][]float64) (cleanup func(), used bool) {
	nn.WebGPUNative = true
	nn.Debug = false
	start := time.Now()
	if err := nn.InitializeOptimizedGPU(); err != nil {
		fmt.Printf("⚠️  WebGPU init failed: %v\n   Falling back to CPU.\n", err)
		nn.WebGPUNative = false
		return func() {}, false
	}
	fmt.Printf("✅ WebGPU initialized in %v\n", time.Since(start))
	if len(warm) > 0 {
		nn.Forward(warm[0])
		_ = nn.ExtractOutput()
	}
	return func() { nn.CleanupOptimizedGPU() }, true
}

func trainModelEpochs(modelPath string, epochs int, lr float64) error {
	images, labels, err := loadMNISTData("./public/mnist")
	if err != nil {
		return fmt.Errorf("load MNIST: %w", err)
	}
	trainInputs, trainTargets, testInputs, testTargets := paragon.SplitDataset(images, labels, 0.8)

	nn, err := loadFloat32Model(modelPath)
	if err != nil {
		return err
	}

	cleanup, _ := withGPU(nn, trainInputs)
	defer cleanup()

	fmt.Printf("🧠 Training %s for %d epoch(s) @ lr=%.4f …\n", filepath.Base(modelPath), epochs, lr)
	start := time.Now()
	//withSilencedStdout(func() {
	nn.Train(trainInputs, trainTargets, epochs, lr, false, float32(2), float32(-2))
	//})
	fmt.Printf("⏱ Training time: %v\n", time.Since(start))

	trainScore := evalADHDScore(nn, trainInputs, trainTargets)
	testScore := evalADHDScore(nn, testInputs, testTargets)
	fmt.Printf("🎯 ADHD scores → Train: %.4f%% | Test: %.4f%%\n", trainScore, testScore)

	if err := nn.SaveJSON(modelPath); err != nil {
		return fmt.Errorf("save model: %w", err)
	}
	fmt.Printf("💾 Saved → %s\n", modelPath)
	return nil
}

func trainModelUntilScore(modelPath string, targetPct float64, maxEpochs int, lr float64) error {
	images, labels, err := loadMNISTData("./public/mnist")
	if err != nil {
		return fmt.Errorf("load MNIST: %w", err)
	}
	trainInputs, trainTargets, testInputs, testTargets := paragon.SplitDataset(images, labels, 0.8)

	nn, err := loadFloat32Model(modelPath)
	if err != nil {
		return err
	}

	cleanup, _ := withGPU(nn, trainInputs)
	defer cleanup()

	fmt.Printf("🧠 Training %s until ADHD ≥ %.2f%% (max %d epochs) @ lr=%.4f …\n",
		filepath.Base(modelPath), targetPct, maxEpochs, lr)

	startAll := time.Now()
	best := -1.0
	var hitEpoch int = -1

	for ep := 1; ep <= maxEpochs; ep++ {
		epStart := time.Now()
		//withSilencedStdout(func() {
		nn.Train(trainInputs, trainTargets, 1, lr, false, float32(2), float32(-2))
		//})
		epDur := time.Since(epStart)

		trainScore := evalADHDScore(nn, trainInputs, trainTargets)
		testScore := evalADHDScore(nn, testInputs, testTargets)
		if testScore > best {
			best = testScore
		}

		fmt.Printf("   Epoch %2d: Train=%.4f%%  Test=%.4f%% (best=%.4f%%)  ⏱ %v\n",
			ep, trainScore, testScore, best, epDur)

		if testScore >= targetPct {
			hitEpoch = ep
			break
		}
	}

	fmt.Printf("⏱ Total training time: %v\n", time.Since(startAll))
	if hitEpoch > 0 {
		fmt.Printf("✅ Target reached at epoch %d (best Test=%.4f%%)\n", hitEpoch, best)
	} else {
		fmt.Printf("⚠️  Target not reached (best Test=%.4f%% after %d epochs)\n", best, maxEpochs)
	}

	if err := nn.SaveJSON(modelPath); err != nil {
		return fmt.Errorf("save model: %w", err)
	}
	fmt.Printf("💾 Saved → %s\n", modelPath)
	return nil
}
