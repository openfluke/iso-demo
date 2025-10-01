package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/openfluke/paragon/v3"
)

type SystemInfo struct {
	Architecture string              `json:"architecture"` // x86_64, arm64 (normalized)
	OS           string              `json:"os"`           // linux, darwin, windows
	OSVersion    string              `json:"os_version"`   // e.g., "Ubuntu 22.04", "macOS 14.6", "Windows 11 10.0.22631"
	CPUModel     string              `json:"cpu_model"`
	GPUModel     string              `json:"gpu_model"`
	DeviceModel  string              `json:"device_model"` // laptop/desktop model where available
	RAMBytes     uint64              `json:"ram_bytes"`
	GPUs         []map[string]string `json:"gpus,omitempty"` // detailed WebGPU adapter info (if available)
}

func (s SystemInfo) ToJSON() string {
	b, _ := json.MarshalIndent(s, "", "  ")
	return string(b)
}

// Collect probes the current machine with per-OS strategies.
func Collect() SystemInfo {
	info := SystemInfo{
		Architecture: normalizeArch(runtime.GOARCH),
		OS:           runtime.GOOS,
	}

	// Best-effort WebGPU adapter enumeration (non-fatal if it fails)
	if gpuList, err := paragon.GetAllGPUInfo(); err == nil && len(gpuList) > 0 {
		info.GPUs = gpuList
	}

	switch runtime.GOOS {
	case "linux":
		info.OSVersion = probeLinuxVersion()
		info.RAMBytes = probeLinuxRAM()
		info.CPUModel = firstNonEmpty(
			readFirstLine("/proc/cpuinfo", "model name"),
			runOne("bash", "-lc", `lscpu | awk -F: '/Model name/ {print $2}'`),
		)
		info.GPUModel = firstNonEmpty(
			runOne("bash", "-lc", `lspci -nn | egrep -i 'vga|3d|display' | sed -E 's/.*: //g' | head -n1`),
			runOne("bash", "-lc", `glxinfo -B 2>/dev/null | awk -F: '/Device:/{sub(/^[ \t]+/,"",$2);print $2; exit}'`),
		)
		// DMI device model
		model := strings.TrimSpace(readFile("/sys/devices/virtual/dmi/id/product_name"))
		vendor := strings.TrimSpace(readFile("/sys/devices/virtual/dmi/id/sys_vendor"))
		if model != "" || vendor != "" {
			info.DeviceModel = strings.TrimSpace(strings.Join([]string{vendor, model}, " "))
		}
	case "darwin":
		info.OSVersion = probeMacOSVersion()
		info.RAMBytes = parseUint(runOne("sysctl", "-n", "hw.memsize"))
		info.CPUModel = runOne("sysctl", "-n", "machdep.cpu.brand_string")
		// GPU via system_profiler JSON (newer macOS) or text fallback
		jsonGPU := runOne("bash", "-lc", `system_profiler SPDisplaysDataType -json 2>/dev/null | jq -r '."SPDisplaysDataType"[0]."spdisplays_videoprocessors"[0] // empty'`)
		if jsonGPU == "" {
			info.GPUModel = runOne("bash", "-lc", `system_profiler SPDisplaysDataType | awk -F: '/Chipset Model:/{sub(/^[ \t]+/,"",$2);print $2; exit}'`)
		} else {
			info.GPUModel = strings.TrimSpace(jsonGPU)
		}
		info.DeviceModel = firstNonEmpty(
			runOne("sysctl", "-n", "hw.model"),
			runOne("bash", "-lc", `system_profiler SPHardwareDataType | awk -F: '/Model Identifier/{sub(/^[ \t]+/,"",$2);print $2; exit}'`),
		)
	case "windows":
		info.OSVersion = probeWindowsVersion()
		info.RAMBytes = winTotalRAM()
		// CPU/GPU/Model using WMIC (works broadly) with PowerShell fallbacks
		info.CPUModel = firstNonEmpty(
			runOne("wmic", "cpu", "get", "Name"),
			runOne("powershell", "-NoProfile", "Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name"),
		)
		info.CPUModel = firstLineClean(info.CPUModel)

		info.GPUModel = firstNonEmpty(
			firstLineClean(runOne("wmic", "path", "win32_VideoController", "get", "Name")),
			firstLineClean(runOne("powershell", "-NoProfile", "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name")),
		)

		sysStr := firstNonEmpty(
			runOne("wmic", "computersystem", "get", "Manufacturer,Model"),
			runOne("powershell", "-NoProfile", "Get-CimInstance Win32_ComputerSystem | Select-Object Manufacturer,Model | Format-Table -HideTableHeaders"),
		)
		info.DeviceModel = compactOneLine(sysStr)
	default:
		// Best-effort generic
		info.OSVersion = strings.TrimSpace(runOne("uname", "-sr"))
		info.RAMBytes = 0
		info.CPUModel = strings.TrimSpace(runOne("uname", "-p"))
		info.GPUModel = ""
		info.DeviceModel = ""
	}

	// Final cleanup/normalization
	info.CPUModel = compactOneLine(info.CPUModel)
	info.GPUModel = compactOneLine(info.GPUModel)
	info.DeviceModel = compactOneLine(info.DeviceModel)

	return info
}

// ---------- helpers ----------

func normalizeArch(goarch string) string {
	switch goarch {
	case "amd64":
		return "x86_64"
	case "arm64":
		return "arm64"
	default:
		return goarch
	}
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		v = strings.TrimSpace(v)
		if v != "" {
			return v
		}
	}
	return ""
}

func firstLineClean(s string) string {
	s = strings.ReplaceAll(s, "\r", "\n")
	sc := bufio.NewScanner(strings.NewReader(s))
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		l := strings.ToLower(line)
		if line != "" && !strings.Contains(l, "name") && !strings.Contains(l, "manufacturer") {
			return line
		}
	}
	return strings.TrimSpace(s)
}

func compactOneLine(s string) string {
	s = strings.ReplaceAll(s, "\r", " ")
	s = strings.ReplaceAll(s, "\n", " ")
	s = strings.Join(strings.Fields(s), " ")
	return strings.TrimSpace(s)
}

func runOne(name string, args ...string) string {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, name, args...)
	cmd.Env = os.Environ()
	out, err := cmd.CombinedOutput()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

// ----- Linux probes -----

func probeLinuxVersion() string {
	if s := parseOsReleasePrettyName(); s != "" {
		return s
	}
	if s := runOne("lsb_release", "-ds"); s != "" {
		return strings.Trim(s, `"`)
	}
	u := runOne("uname", "-sr")
	return u
}

func parseOsReleasePrettyName() string {
	b, err := os.ReadFile("/etc/os-release")
	if err != nil {
		return ""
	}
	for _, line := range bytes.Split(b, []byte("\n")) {
		if bytes.HasPrefix(line, []byte("PRETTY_NAME=")) {
			v := strings.TrimPrefix(string(line), "PRETTY_NAME=")
			return strings.Trim(v, `"`)
		}
	}
	return ""
}

func probeLinuxRAM() uint64 {
	// /proc/meminfo MemTotal: kB
	f, err := os.Open("/proc/meminfo")
	if err != nil {
		return 0
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := sc.Text()
		if strings.HasPrefix(line, "MemTotal:") {
			fields := strings.Fields(line)
			if len(fields) >= 2 {
				kb, _ := strconv.ParseUint(fields[1], 10, 64)
				return kb * 1024
			}
		}
	}
	return 0
}

func readFirstLine(path, contains string) string {
	b, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	for _, line := range strings.Split(string(b), "\n") {
		if strings.Contains(strings.ToLower(line), strings.ToLower(contains)) {
			if idx := strings.Index(line, ":"); idx >= 0 {
				return strings.TrimSpace(line[idx+1:])
			}
			return strings.TrimSpace(line)
		}
	}
	return ""
}

func readFile(path string) string {
	b, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	return string(b)
}

// ----- macOS probes -----

func probeMacOSVersion() string {
	name := runOne("sw_vers", "-productName")
	ver := runOne("sw_vers", "-productVersion")
	if name == "" && ver == "" {
		return "macOS"
	}
	return strings.TrimSpace(fmt.Sprintf("%s %s", name, ver))
}

// ----- Windows probes -----

func probeWindowsVersion() string {
	// WMIC captions + version (works on many systems)
	cap := firstLineClean(runOne("wmic", "os", "get", "Caption"))
	ver := firstLineClean(runOne("wmic", "os", "get", "Version"))
	if cap != "" || ver != "" {
		return strings.TrimSpace(fmt.Sprintf("%s %s", cap, ver))
	}
	// PowerShell fallback (Win11 prefers CIM)
	ps := runOne("powershell", "-NoProfile", "(Get-CimInstance Win32_OperatingSystem) | Select-Object -ExpandProperty Caption")
	if ps != "" {
		v := runOne("powershell", "-NoProfile", "(Get-CimInstance Win32_OperatingSystem) | Select-Object -ExpandProperty Version")
		return strings.TrimSpace(fmt.Sprintf("%s %s", ps, v))
	}
	return "Windows"
}

func winTotalRAM() uint64 {
	// WMIC returns KB when using TotalVisibleMemorySize
	out := runOne("wmic", "OS", "get", "TotalVisibleMemorySize", "/Value")
	if out != "" {
		for _, line := range strings.Split(out, "\n") {
			line = strings.TrimSpace(line)
			if strings.HasPrefix(line, "TotalVisibleMemorySize=") {
				kbStr := strings.TrimPrefix(line, "TotalVisibleMemorySize=")
				kb, err := strconv.ParseUint(strings.TrimSpace(kbStr), 10, 64)
				if err == nil {
					return kb * 1024
				}
			}
		}
	}
	// PowerShell fallback (bytes)
	v := runOne("powershell", "-NoProfile", "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory")
	if v != "" {
		return parseUint(v)
	}
	return 0
}

func parseUint(s string) uint64 {
	s = strings.TrimSpace(s)
	if s == "" {
		return 0
	}
	u, err := strconv.ParseUint(s, 10, 64)
	if err != nil {
		return 0
	}
	return u
}

// ---- Optional: error helpers (not used heavily here) ----
var errNotFound = errors.New("not found")
