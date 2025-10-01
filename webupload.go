package main

import (
	"fmt"
	"path/filepath"
	"time"

	"github.com/gofiber/fiber/v2"
)

func RegisterUpload(app *fiber.App, baseDir string) {
	reportsDir := filepath.Join(baseDir, "reports")

	app.Post("/upload", func(c *fiber.Ctx) error {
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
			// default to timestamp + original name (avoid overwriting)
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

	// also expose /reports statically if present (directory browsing on)
	app.Static("/reports", reportsDir, fiber.Static{
		Browse: true,
	})
}
