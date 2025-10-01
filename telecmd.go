package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"
)

func runTelemetryMenu() {
	reader := bufio.NewReader(os.Stdin)

	fmt.Print("Target host base (e.g., http://192.168.1.20:8080): ")
	raw, _ := reader.ReadString('\n')
	host := strings.TrimSpace(raw)
	if host == "" {
		fmt.Println("❌ host required")
		return
	}

	fmt.Println("Source environment:")
	fmt.Println(" 1) native")
	fmt.Println(" 2) wasm-bun")
	fmt.Println(" 3) wasm-ionic")
	fmt.Print("Select [1-3] (default 1): ")
	rawS, _ := reader.ReadString('\n')
	rawS = strings.TrimSpace(rawS)

	src := SourceNative
	switch rawS {
	case "2":
		src = SourceWASMBun
	case "3":
		src = SourceWASMIonic
	}

	fmt.Printf("▶ Running telemetry against %s as %s…\n", host, src)
	path, err := RunTelemetryPipeline(host, src)
	if err != nil {
		fmt.Println("❌ Telemetry failed:", err)
		return
	}
	fmt.Println("✅ Telemetry saved locally →", path)
	fmt.Printf("📤 Uploaded report back to %s at /reports/\n", host)
	fmt.Println("   Tip: Open ", host, "/reports/ to see it.")
}
