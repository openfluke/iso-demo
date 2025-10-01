package main

import (
	"fmt"
	"net"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/compress"
	"github.com/gofiber/fiber/v2/middleware/cors"
	"github.com/gofiber/fiber/v2/middleware/logger"
)

type webServer struct {
	app     *fiber.App
	addr    string
	dir     string
	running bool
	mu      sync.RWMutex
	errc    chan error
}

var ws webServer

// StartWeb starts a Fiber server in a goroutine and serves `dir` at `/`,
// and `dir/compiled` at `/compiled`. Binds 0.0.0.0 so your LAN can reach it.
func StartWeb(port int, dir string) error {
	ws.mu.Lock()
	defer ws.mu.Unlock()

	if ws.running {
		return fmt.Errorf("web server already running at http://%s", ws.addr)
	}
	if dir == "" {
		dir = "public"
	}
	if _, err := os.Stat(dir); err != nil {
		return fmt.Errorf("public dir %q not found: %w", dir, err)
	}

	ws.addr = fmt.Sprintf("0.0.0.0:%d", port)
	ws.dir = dir
	ws.errc = make(chan error, 1)

	app := fiber.New(fiber.Config{
		ServerHeader:          "OpenFluke-ISO",
		AppName:               "Paragon ISO Demo",
		DisableStartupMessage: true,
		ReadTimeout:           10 * time.Second,
		WriteTimeout:          30 * time.Second,
		IdleTimeout:           60 * time.Second,
	})

	// Middleware
	app.Use(logger.New())
	app.Use(cors.New(cors.Config{
		AllowOrigins: "*",
		AllowHeaders: "*",
	}))
	app.Use(compress.New(compress.Config{Level: compress.LevelBestSpeed}))

	// Health/info
	app.Get("/healthz", func(c *fiber.Ctx) error { return c.SendString("ok") })
	app.Get("/whoami", func(c *fiber.Ctx) error {
		return c.JSON(fiber.Map{
			"addr":       ws.addr,
			"public_dir": filepath.Clean(ws.dir),
			"lan_urls":   lanURLs(port),
			"started_at": time.Now().UTC(),
		})
	})

	// Static mounts with directory browsing
	app.Static("/", filepath.Clean(ws.dir), fiber.Static{
		Browse:        true,
		Index:         "index.html",
		CacheDuration: time.Hour,
	})
	compiled := filepath.Join(ws.dir, "compiled")
	if st, err := os.Stat(compiled); err == nil && st.IsDir() {
		app.Static("/compiled", filepath.Clean(compiled), fiber.Static{
			Browse:        true,
			CacheDuration: time.Hour,
		})
	}

	// Run in background
	go func() {
		// Listen returns an error when shutdown is called; we just forward it.
		ws.errc <- app.Listen(ws.addr)
	}()

	// Mark running
	ws.app = app
	ws.running = true
	printServerBanner(port, dir)
	return nil
}

// StopWeb gracefully shuts the server down.
func StopWeb() error {
	ws.mu.Lock()
	defer ws.mu.Unlock()

	if !ws.running || ws.app == nil {
		return fmt.Errorf("web server is not running")
	}
	// Trigger graceful shutdown
	err := ws.app.Shutdown()
	ws.running = false
	ws.app = nil
	// Drain any listen error (ignore on clean close)
	select {
	case <-ws.errc:
	default:
	}
	return err
}

// WebStatus returns whether the server is running and its bind address.
func WebStatus() (bool, string) {
	ws.mu.RLock()
	defer ws.mu.RUnlock()
	return ws.running, ws.addr
}

// ---- helpers ----

func lanURLs(port int) []string {
	var urls []string
	ifaces, _ := net.Interfaces()
	for _, ifc := range ifaces {
		if ifc.Flags&net.FlagUp == 0 || ifc.Flags&net.FlagLoopback != 0 {
			continue
		}
		addrs, _ := ifc.Addrs()
		for _, a := range addrs {
			if ipnet, ok := a.(*net.IPNet); ok && ipnet.IP.To4() != nil {
				urls = append(urls, fmt.Sprintf("http://%s:%d", ipnet.IP.String(), port))
			}
		}
	}
	urls = append(urls, fmt.Sprintf("http://127.0.0.1:%d", port))
	return urls
}

func printServerBanner(port int, dir string) {
	fmt.Println("🌐 Web server started")
	for _, u := range lanURLs(port) {
		fmt.Printf("   → %s\n", u)
	}
	fmt.Printf("   Serving: %s (and %s if present)\n", filepath.Clean(dir), filepath.Join(dir, "compiled"))
}

// parsePort is handy if you want to re-list LAN URLs from the menu.
func parsePort(addr string) int {
	parts := filepath.SplitList(addr) // not ideal; safer split by ':'
	_ = parts
	// Safer parse:
	colon := -1
	for i := len(addr) - 1; i >= 0; i-- {
		if addr[i] == ':' {
			colon = i
			break
		}
	}
	if colon >= 0 {
		if v, err := strconv.Atoi(addr[colon+1:]); err == nil {
			return v
		}
	}
	return 8080
}
