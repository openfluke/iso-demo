package main

import (
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/gofiber/fiber/v2"
)

func RegisterUpload(app *fiber.App, baseDir string) {
	reportsDir := filepath.Join(baseDir, "reports")

	// Ensure baseDir and reportsDir exist up front
	_ = os.MkdirAll(baseDir, 0755)
	_ = os.MkdirAll(reportsDir, 0755)

	app.Post("/upload", func(c *fiber.Ctx) error {
		// Safety: make sure the dir still exists (e.g., if it was deleted)
		if err := os.MkdirAll(reportsDir, 0755); err != nil {
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
				"error": "failed to create reports dir: " + err.Error(),
			})
		}

		// single file field "file"
		fh, err := c.FormFile("file")
		if err != nil {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error": "missing file field",
			})
		}

		// optional "name" param overrides the filename
		name := c.FormValue("name")
		if name == "" {
			name = fmt.Sprintf("%d_%s", time.Now().Unix(), fh.Filename)
		}

		dst := filepath.Join(reportsDir, name)
		if err := c.SaveFile(fh, dst); err != nil {
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
				"error": err.Error(),
			})
		}
		return c.JSON(fiber.Map{
			"saved":  true,
			"path":   dst,
			"public": fmt.Sprintf("/reports/%s", name),
		})
	})

	// Always expose /reports (directory browsing on)
	app.Static("/reports", reportsDir, fiber.Static{
		Browse: true,
	})
}
