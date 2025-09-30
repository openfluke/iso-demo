package main

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/openfluke/pilot"
	"github.com/openfluke/pilot/experiments"
)

func main() {
	// If a number is passed on the command line, run it directly
	if len(os.Args) > 1 {
		choice := strings.TrimSpace(os.Args[1])
		runChoice(choice)
		return
	}

	// Otherwise, fall back to the interactive loop
	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Println()
		fmt.Println("=== Paragon ISO Demo ===")
		fmt.Println("1) Show computer info (JSON)")
		fmt.Println("2) Run MNIST experiment (download/train/test via PILOT)")
		fmt.Println("3) Export MNIST images to PNG (public/mnist_png/all)")
		fmt.Println("4) Create the models for testing")
		fmt.Println("5) Benchmark models CPU on digit samples 1 item of each number (1 to 9)")
		fmt.Println("6) Benchmark models GPU on digit samples 1 item of each number (1 to 9)")
		fmt.Println("7) Compare CPU vs GPU (choose model)")
		fmt.Println("8) Train a model for 1 epoch")
		fmt.Println("9) Evaluate a model on Train/Test set (ADHD metrics)")

		fmt.Println("0) Exit")
		fmt.Print("Select: ")

		choiceRaw, _ := reader.ReadString('\n')
		choice := strings.TrimSpace(choiceRaw)
		runChoice(choice)
	}
}

func runChoice(choice string) {
	switch choice {
	case "1":
		doShowInfo()
	case "2":
		doRunExperiment()
	case "3":
		doExportPNGs()
	case "4":
		createModelZoo()
	case "5":
		benchmarkModelsOnDigits(false)
	case "6":
		benchmarkModelsOnDigits(true)
	case "7":
		runCompareMenu()
	case "8":
		runTrainMenu()
	case "9":
		runEvaluateMenu()

	case "0":
		fmt.Println("Bye.")
		os.Exit(0)
	default:
		fmt.Println("Unknown option. Please choose 0, 1, 2, or 3.")
	}
}

func doShowInfo() {
	info := Collect()
	fmt.Println(info.ToJSON())
}

func doRunExperiment() {
	fmt.Println("🚀 Launching PILOT MNIST experiment…")
	start := time.Now()
	if err := runPilotMNIST(); err != nil {
		fmt.Println("❌ Experiment failed:", err)
		return
	}
	fmt.Printf("✅ Experiment completed in %v\n", time.Since(start))
}

func doExportPNGs() {
	const mnistDir = "./public/mnist"
	startData := time.Now()
	images, labels, err := loadMNISTData(mnistDir)
	if err != nil {
		fmt.Println("❌ Failed to load MNIST from", mnistDir, "— run option 2 first to download.")
		fmt.Println("Error:", err)
		return
	}
	loadT := time.Since(startData)

	fmt.Printf("📊 Loaded %d samples (approx Train=%d, Test=%d)\n", len(images), len(images)*8/10, len(images)*2/10)
	fmt.Printf("⏱ Data Prep Time: %v\n", loadT)

	startExport := time.Now()
	if err := exportMNISTAsPNGs(images, labels, "all"); err != nil {
		fmt.Println("❌ PNG export failed:", err)
		return
	}
	fmt.Printf("✅ Exported %d images to public/mnist_png/all in %v\n", len(images), time.Since(startExport))
}

// --- Existing experiment launcher (kept from your code) ---
func runPilotMNIST() error {
	mnist := experiments.NewMNISTDatasetStage("./public/mnist")
	exp := pilot.NewExperiment("MNIST", mnist)
	return exp.RunAll()
}

func runCompareMenu() {
	modelDir := filepath.Join("public", "models")

	// list models
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
	fmt.Printf("\n▶ Running CPU vs GPU comparison for %s\n", models[idx-1])
	compareSingleModel(modelPath)
}
