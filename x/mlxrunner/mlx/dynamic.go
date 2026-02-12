//go:build mlx

package mlx

// #include "dynamic.h"
// #include "generated.h"
// #include <stdlib.h>
import "C"

import (
	"fmt"
	"io/fs"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
	"unsafe"
)

var initError error

// CheckInit returns any error that occurred during MLX dynamic library initialization.
func CheckInit() error {
	return initError
}

func init() {
	switch runtime.GOOS {
	case "darwin":

	case "windows":
	default:
		return
	}

	// Build search paths: OLLAMA_LIBRARY_PATH dirs + executable-relative fallback
	var searchPaths []string
	if paths, ok := os.LookupEnv("OLLAMA_LIBRARY_PATH"); ok {
		searchPaths = append(searchPaths, filepath.SplitList(paths)...)
	}

	// Fallback: directory containing the current executable (production installs)
	if exe, err := os.Executable(); err == nil {
		if resolved, err := filepath.EvalSymlinks(exe); err == nil {
			searchPaths = append(searchPaths, filepath.Dir(resolved))
		}
	}

	if len(searchPaths) == 0 {
		slog.Debug("no MLX library search paths available, skipping mlx dynamic loading")
		return
	}

	for _, dir := range searchPaths {
		matches, err := fs.Glob(os.DirFS(dir), "libmlxc.*")
		if err != nil {
			slog.Error("failed to glob MLX library directory", "dir", dir, "error", err)
			continue
		}

		for _, match := range matches {
			libPath := filepath.Join(dir, match)
			slog.Info("Loading MLX dynamic library", "path", libPath)

			cPath := C.CString(libPath)
			defer C.free(unsafe.Pointer(cPath))

			var handle C.mlx_dynamic_handle
			if C.mlx_dynamic_load(&handle, cPath) != 0 {
				slog.Error("Failed to load MLX dynamic library", "path", libPath)
				continue
			}

			if C.mlx_dynamic_load_symbols(handle) != 0 {
				slog.Error("Failed to load MLX dynamic library symbols", "path", libPath)
				C.mlx_dynamic_unload(&handle)
				continue
			}

			slog.Info("Loaded MLX dynamic library", "path", libPath)
			mlxLoaded = true
			return
		}
	}

	initError = fmt.Errorf("failed to load any MLX dynamic library from search paths: %v", searchPaths)
	slog.Warn("MLX dynamic library not available", "error", initError)
	// Don't panic — this binary serves multiple purposes (server, client, runner).
	// The MLX library is only needed when actually running as the mlxrunner subprocess.
	// Callers that need MLX should check mlxLoaded before using C functions.
	slog.Debug("MLX dynamic library not found, mlxrunner functions unavailable", "searched", searchPaths)
}

// mlxLoaded tracks whether the MLX C library was successfully loaded.
var mlxLoaded bool

// Loaded returns whether the MLX C library was successfully loaded.
// Callers should check this before invoking any CGO MLX functions.
func Loaded() bool {
	return mlxLoaded
}
