package main

import (
	"fmt"
	"os"

	"github.com/openfluke/pilot"
	"github.com/openfluke/pilot/experiments"
)

func main() {
	info := Collect()
	fmt.Println(info.ToJSON())
	download()
}

func download() {
	fmt.Println("🚀 Launcher starting PILOT...")

	mnist := experiments.NewMNISTDatasetStage("./data/mnist")
	exp := pilot.NewExperiment("MNIST", mnist)

	if err := exp.RunAll(); err != nil {
		fmt.Println("❌ Experiment failed:", err)
		os.Exit(1)
	}
}
