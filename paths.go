package main

import (
	"errors"
	"flag"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
)

var (
	baseOnce sync.Once
	baseDir  string
	baseErr  error

	// optional CLI override (wire in main() if you want)
	flagBaseDir = flag.String("base", "", "Base data directory (overrides auto-detect)")
)

func BaseDir() (string, error) {
	baseOnce.Do(func() {
		// 1) ENV override
		if v := strings.TrimSpace(os.Getenv("PARAGON_DATA_DIR")); v != "" {
			if isDir(v) {
				baseDir = v
				return
			}
			baseErr = errors.New("PARAGON_DATA_DIR set but not a directory: " + v)
			return
		}
		// 2) Flag override (if flags parsed)
		if flag.Parsed() && *flagBaseDir != "" {
			if isDir(*flagBaseDir) {
				baseDir = *flagBaseDir
				return
			}
			baseErr = errors.New("--base provided but not a directory: " + *flagBaseDir)
			return
		}
		// 3) Adjacent to executable
		if exe, err := os.Executable(); err == nil {
			exeDir := filepath.Dir(exe)
			if d := filepath.Join(exeDir, "public"); isDir(d) {
				baseDir = d
				return
			}
		}
		// 4) Walk upward from CWD looking for "public"
		if cwd, err := os.Getwd(); err == nil {
			if d, ok := findUp(cwd, "public"); ok {
				baseDir = d
				return
			}
		}
		// 5) Fallback ./public
		if isDir("public") {
			if abs, err := filepath.Abs("public"); err == nil {
				baseDir = abs
				return
			}
			baseDir = "public"
			return
		}
		baseErr = errors.New(`could not locate a "public" directory; set PARAGON_DATA_DIR or use --base`)
	})
	return baseDir, baseErr
}

func PublicPath(parts ...string) (string, error) {
	b, err := BaseDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(append([]string{b}, parts...)...), nil
}

func MustPublicPath(parts ...string) string {
	p, err := PublicPath(parts...)
	if err != nil {
		panic(err)
	}
	return p
}

func EnsurePublicDir(parts ...string) (string, error) {
	p, err := PublicPath(parts...)
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(p, 0o755); err != nil {
		return "", err
	}
	return p, nil
}

func isDir(p string) bool {
	st, err := os.Stat(p)
	return err == nil && st.IsDir()
}

func findUp(start, name string) (string, bool) {
	dir := start
	for {
		candidate := filepath.Join(dir, name)
		if isDir(candidate) {
			return candidate, true
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	return "", false
}

// Optional: OS-specific user data roots if/when you want to store writable app data
func UserDataRoot(app string) string {
	home, _ := os.UserHomeDir()
	switch runtime.GOOS {
	case "darwin":
		return filepath.Join(home, "Library", "Application Support", app)
	case "windows":
		if r := os.Getenv("APPDATA"); r != "" {
			return filepath.Join(r, app)
		}
		return filepath.Join(home, "AppData", "Roaming", app)
	default: // linux/unix (XDG)
		if x := os.Getenv("XDG_DATA_HOME"); x != "" {
			return filepath.Join(x, app)
		}
		return filepath.Join(home, ".local", "share", app)
	}
}
