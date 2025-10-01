package main

import (
	"encoding/json"
	"fmt"
	"runtime"
	"sort"
	"strings"
	"time"

	"github.com/openfluke/paragon/v3"
)

type BenchInfo struct {
	StartedAt     time.Time                          `json:"started_at"`
	EndedAt       time.Time                          `json:"ended_at"`
	DurationSec   float64                            `json:"duration_sec"`
	NumCPU        int                                `json:"num_cpu"`
	Filter        string                             `json:"filter"` // "all", "ints", "floats", or comma list (e.g., "int,float32")
	Results       []paragon.BenchmarkResult          `json:"results"`
	ResultsByType map[string]paragon.BenchmarkResult `json:"results_by_type,omitempty"`
}

func (b BenchInfo) ToJSON() string {
	bz, _ := json.MarshalIndent(b, "", "  ")
	return string(bz)
}

// CollectBenchmarks runs the Paragon numeric micro-bench for `duration` and
// returns structured results. `filter` can be:
//   - "all" (default) to keep all numeric types
//   - "ints" to keep integer types only
//   - "floats" to keep float32/float64 only
//   - a comma list: e.g. "int,int32,float32"
func CollectBenchmarks(duration time.Duration, filter string) (BenchInfo, error) {
	if filter == "" {
		filter = "all"
	}

	start := time.Now()
	raw := paragon.RunAllBenchmarks(duration) // returns JSON []BenchmarkResult
	end := time.Now()

	var results []paragon.BenchmarkResult
	if err := json.Unmarshal([]byte(raw), &results); err != nil {
		return BenchInfo{}, fmt.Errorf("failed to parse paragon benchmarks: %w", err)
	}

	results = applyBenchFilter(results, filter)

	// Stable order by type name for deterministic logs
	sort.Slice(results, func(i, j int) bool { return results[i].Type < results[j].Type })

	byType := make(map[string]paragon.BenchmarkResult, len(results))
	for _, r := range results {
		byType[r.Type] = r
	}

	info := BenchInfo{
		StartedAt:     start.UTC(),
		EndedAt:       end.UTC(),
		DurationSec:   end.Sub(start).Seconds(),
		NumCPU:        runtime.NumCPU(),
		Filter:        filter,
		Results:       results,
		ResultsByType: byType,
	}
	return info, nil
}

func applyBenchFilter(rs []paragon.BenchmarkResult, filter string) []paragon.BenchmarkResult {
	filter = strings.ToLower(strings.TrimSpace(filter))
	if filter == "" || filter == "all" {
		return rs
	}

	var keep = map[string]bool{}
	switch filter {
	case "ints":
		for _, t := range []string{"int", "int8", "int16", "int32", "int64", "uint", "uint8", "uint16", "uint32", "uint64"} {
			keep[t] = true
		}
	case "floats":
		keep["float32"] = true
		keep["float64"] = true
	default:
		// comma list
		for _, t := range strings.Split(filter, ",") {
			t = strings.TrimSpace(t)
			if t != "" {
				keep[t] = true
			}
		}
	}

	out := make([]paragon.BenchmarkResult, 0, len(rs))
	for _, r := range rs {
		if len(keep) == 0 || keep[strings.ToLower(r.Type)] {
			out = append(out, r)
		}
	}
	return out
}
