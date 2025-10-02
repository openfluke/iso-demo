package main

import (
	"encoding/binary"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"os"

	"path/filepath"
)

// Loads both training and test images, returns as one dataset
func loadMNISTData(dir string) ([][][]float64, [][][]float64, error) {
	images := make([][][]float64, 0)
	labels := make([][][]float64, 0)

	for _, set := range []string{"train", "t10k"} {
		imgPath := filepath.Join(dir, set+"-images-idx3-ubyte")
		lblPath := filepath.Join(dir, set+"-labels-idx1-ubyte")

		imgs, err := loadMNISTImages(imgPath)
		if err != nil {
			return nil, nil, err
		}

		lbls, err := loadMNISTLabels(lblPath)
		if err != nil {
			return nil, nil, err
		}

		images = append(images, imgs...)
		labels = append(labels, lbls...)
	}

	return images, labels, nil
}

func loadMNISTImages(path string) ([][][]float64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var header [16]byte
	if _, err := f.Read(header[:]); err != nil {
		return nil, err
	}
	num := int(binary.BigEndian.Uint32(header[4:8]))
	rows := int(binary.BigEndian.Uint32(header[8:12]))
	cols := int(binary.BigEndian.Uint32(header[12:16]))

	images := make([][][]float64, num)
	buf := make([]byte, rows*cols)
	for i := 0; i < num; i++ {
		if _, err := f.Read(buf); err != nil {
			return nil, err
		}
		img := make([][]float64, rows)
		for r := 0; r < rows; r++ {
			img[r] = make([]float64, cols)
			for c := 0; c < cols; c++ {
				img[r][c] = float64(buf[r*cols+c]) / 255.0
			}
		}
		images[i] = img
	}
	return images, nil
}

func loadMNISTLabels(path string) ([][][]float64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var header [8]byte
	if _, err := f.Read(header[:]); err != nil {
		return nil, err
	}
	num := int(binary.BigEndian.Uint32(header[4:8]))

	labels := make([][][]float64, num)
	for i := 0; i < num; i++ {
		var b [1]byte
		if _, err := f.Read(b[:]); err != nil {
			return nil, err
		}
		labels[i] = labelToOneHot(int(b[0]))
	}
	return labels, nil
}

func labelToOneHot(label int) [][]float64 {
	t := make([][]float64, 1)
	t[0] = make([]float64, 10)
	t[0][label] = 1.0
	return t
}

// Export all MNIST images as PNGs into public/mnist_png/[train|t10k]
func exportMNISTAsPNGs(images [][][]float64, labels [][][]float64, setName string) error {
	// Use MustPublicPath for cross-platform compatibility
	baseDir := MustPublicPath("mnist_png", setName)
	fmt.Printf("📂 Creating export directory: %s\n", baseDir)

	if err := os.MkdirAll(baseDir, 0755); err != nil {
		return fmt.Errorf("failed to create base directory %s: %w", baseDir, err)
	}

	for i, img := range images {
		// Progress indicator every 1000 images
		if i > 0 && i%1000 == 0 {
			fmt.Printf("   Processed %d/%d images...\n", i, len(images))
		}

		rows := len(img)
		cols := len(img[0])
		gray := image.NewGray(image.Rect(0, 0, cols, rows))

		for r := 0; r < rows; r++ {
			for c := 0; c < cols; c++ {
				v := uint8(img[r][c] * 255)
				gray.SetGray(c, r, color.Gray{Y: v})
			}
		}

		// optional: label directory
		labelIndex := -1
		if len(labels) > i && len(labels[i][0]) == 10 {
			for j, v := range labels[i][0] {
				if v == 1.0 {
					labelIndex = j
					break
				}
			}
		}

		var outPath string
		if labelIndex >= 0 {
			labelDir := filepath.Join(baseDir, fmt.Sprintf("%d", labelIndex))
			if err := os.MkdirAll(labelDir, 0755); err != nil {
				return fmt.Errorf("failed to create label directory %s: %w", labelDir, err)
			}
			outPath = filepath.Join(labelDir, fmt.Sprintf("img_%05d.png", i))
		} else {
			outPath = filepath.Join(baseDir, fmt.Sprintf("img_%05d.png", i))
		}

		// save png
		f, err := os.Create(outPath)
		if err != nil {
			return fmt.Errorf("failed to create %s: %w", outPath, err)
		}
		if err := png.Encode(f, gray); err != nil {
			f.Close()
			return fmt.Errorf("failed to encode PNG %s: %w", outPath, err)
		}
		f.Close()
	}

	fmt.Printf("✅ All images written to: %s\n", baseDir)
	return nil
}

func flattenMNIST64(img [][]float64) [][]float64 {
	rows, cols := len(img), len(img[0])
	out := make([][]float64, 1)
	out[0] = make([]float64, rows*cols)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			out[0][r*cols+c] = img[r][c]
		}
	}
	return out
}

func argmax64(v []float64) int {
	best, idx := v[0], 0
	for i := 1; i < len(v); i++ {
		if v[i] > best {
			best, idx = v[i], i
		}
	}
	return idx
}
