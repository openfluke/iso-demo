package main

import (
	"bufio"
	"encoding/json"
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
		fmt.Println("8) Train model(s): N epochs or until target ADHD%")
		fmt.Println("9) Evaluate a model on Train/Test set (ADHD metrics)")
		fmt.Println("10) Run CPU numeric microbench (duration/filter/format)")

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
	case "10":
		runBenchMenu()

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

// --- Bench menu (wired to sysbench.go) ---
func runBenchMenu() {
	reader := bufio.NewReader(os.Stdin)

	// Duration
	fmt.Print("Benchmark duration [e.g., 2s, 1500ms, 3s] (default 2s): ")
	durRaw, _ := reader.ReadString('\n')
	durStr := strings.TrimSpace(durRaw)
	if durStr == "" {
		durStr = "2s"
	}
	dur, err := time.ParseDuration(durStr)
	if err != nil || dur <= 0 {
		fmt.Println("❌ Invalid duration")
		return
	}

	// Filter
	fmt.Println("Filter options:")
	fmt.Println(" - all     (all numeric types)")
	fmt.Println(" - ints    (int/uint types)")
	fmt.Println(" - floats  (float32,float64)")
	fmt.Println(" - custom  (comma list, e.g., int,int32,float32)")
	fmt.Print("Choose filter [all/ints/floats/custom] (default all): ")
	fRaw, _ := reader.ReadString('\n')
	fSel := strings.TrimSpace(strings.ToLower(fRaw))
	var filter string
	switch fSel {
	case "", "all":
		filter = "all"
	case "ints":
		filter = "ints"
	case "floats":
		filter = "floats"
	case "custom":
		fmt.Print("Enter comma-separated type list: ")
		cRaw, _ := reader.ReadString('\n')
		filter = strings.TrimSpace(cRaw)
	default:
		fmt.Println("❌ Invalid filter choice")
		return
	}

	// Output format
	fmt.Print("Output format [table/json] (default table): ")
	fmtFmtRaw, _ := reader.ReadString('\n')
	fmtFmt := strings.TrimSpace(strings.ToLower(fmtFmtRaw))
	if fmtFmt == "" {
		fmtFmt = "table"
	}
	if fmtFmt != "table" && fmtFmt != "json" {
		fmt.Println("❌ Invalid format")
		return
	}

	// Optional outfile
	fmt.Print("Write JSON to file as well? (leave blank to skip): ")
	outRaw, _ := reader.ReadString('\n')
	outFile := strings.TrimSpace(outRaw)

	// Run
	info, err := CollectBenchmarks(dur, filter)
	if err != nil {
		fmt.Println("❌ Benchmark error:", err)
		return
	}

	if fmtFmt == "json" {
		out := info.ToJSON()
		fmt.Println(out)
		if outFile != "" {
			if err := os.WriteFile(outFile, []byte(out), 0o644); err != nil {
				fmt.Printf("❌ Failed to write %s: %v\n", outFile, err)
				return
			}
			fmt.Printf("💾 JSON written → %s\n", outFile)
		}
		return
	}

	// Pretty table
	fmt.Printf("Numeric Microbench (dur=%.3gs, cpu=%d, filter=%s)\n",
		info.DurationSec, info.NumCPU, info.Filter)
	fmt.Println("-------------------------------------------------------------")
	fmt.Printf("%-10s | %-17s | %-17s\n", "Type", "Single-Threaded", "Multi-Threaded")
	fmt.Println("-------------------------------------------------------------")
	for _, r := range info.Results {
		fmt.Printf("%-10s | %-17s | %-17s\n",
			r.Type, humanize(r.Single), humanize(r.Multi))
	}
	fmt.Println("-------------------------------------------------------------")

	// Optional write JSON even in table mode
	if outFile != "" {
		bz, _ := json.MarshalIndent(info, "", "  ")
		if err := os.WriteFile(outFile, bz, 0o644); err != nil {
			fmt.Printf("❌ Failed to write %s: %v\n", outFile, err)
			return
		}
		fmt.Printf("💾 JSON written → %s\n", outFile)
	}
}

func humanize(n int) string {
	f := float64(n)
	switch {
	case f >= 1e12:
		return fmt.Sprintf("%.2fT", f/1e12)
	case f >= 1e9:
		return fmt.Sprintf("%.2fB", f/1e9)
	case f >= 1e6:
		return fmt.Sprintf("%.2fM", f/1e6)
	case f >= 1e3:
		return fmt.Sprintf("%.2fK", f/1e3)
	default:
		return fmt.Sprintf("%d", n)
	}
}
