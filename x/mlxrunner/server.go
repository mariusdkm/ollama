//go:build mlx

package mlxrunner

import (
	"bytes"
	"cmp"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"strconv"
	"time"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/sample"
)

func Execute(args []string) error {
	slog.SetDefault(logutil.NewLogger(os.Stderr, envconfig.LogLevel()))

	if err := mlx.CheckInit(); err != nil {
		return fmt.Errorf("MLX not available: %w", err)
	}

	var (
		modelName string
		port      int
	)

	flagSet := flag.NewFlagSet("mlxrunner", flag.ExitOnError)
	flagSet.StringVar(&modelName, "model", "", "Model name")
	flagSet.IntVar(&port, "port", 0, "Port to listen on")
	_ = flagSet.Bool("verbose", false, "Enable debug logging")
	flagSet.Parse(args)

	runner := Runner{
		Requests:     make(chan Request),
		CacheEntries: make(map[int32]*CacheEntry),
	}

	if err := runner.Load(modelName); err != nil {
		return err
	}

	mux := http.NewServeMux()

	// Health/status — used by imagegen wrapper's Ping() and WaitUntilRunning()
	statusHandler := func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewEncoder(w).Encode(map[string]any{
			"status":   0,
			"progress": 100,
		}); err != nil {
			slog.Error("Failed to encode response", "error", err)
			http.Error(w, "Internal Server Error", http.StatusInternalServerError)
			return
		}
	}
	mux.HandleFunc("GET /v1/status", statusHandler)
	mux.HandleFunc("GET /health", statusHandler)

	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case "POST":
			fallthrough
		case "GET":
			if err := json.NewEncoder(w).Encode(map[string]any{
				"Success": true,
			}); err != nil {
				slog.Error("Failed to encode response", "error", err)
				http.Error(w, "Internal Server Error", http.StatusInternalServerError)
				return
			}
		case "DELETE":
			// TODO: cleanup model and cache
		}
	})
	mux.Handle("POST /load", http.RedirectHandler("/v1/models", http.StatusPermanentRedirect))

	// Completions — used by imagegen wrapper's Completion()
	// Direct handlers instead of 308 redirects to avoid POST body issues
	completionsHandler := func(w http.ResponseWriter, r *http.Request) {
		request := Request{Responses: make(chan Response)}

		if err := json.NewDecoder(r.Body).Decode(&request.TextCompletionsRequest); err != nil {
			slog.Error("Failed to decode request", "error", err)
			http.Error(w, "Bad Request", http.StatusBadRequest)
			return
		}

		request.Options.MaxTokens = cmp.Or(request.Options.MaxTokens, request.Options.NumPredict)
		if request.Options.MaxTokens < 1 {
			request.Options.MaxTokens = 16 << 10
		}

		request.Pipeline = runner.TextGenerationPipeline
		request.Sampler = sample.New(
			request.Options.Temperature,
			request.Options.TopP,
			request.Options.MinP,
			request.Options.TopK,
		)

		runner.Requests <- request

		w.Header().Set("Content-Type", "application/jsonl")
		w.WriteHeader(http.StatusOK)
		enc := json.NewEncoder(w)
		for response := range request.Responses {
			if err := enc.Encode(response); err != nil {
				slog.Error("Failed to encode response", "error", err)
				return
			}

			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
		}
	}
	mux.HandleFunc("POST /v1/completions", completionsHandler)
	mux.HandleFunc("POST /completion", completionsHandler)

	// Tokenize — used by imagegen wrapper's Tokenize()
	// Accepts JSON {"content":"..."} (wrapper format) and returns {"tokens":[...]}
	tokenizeHandler := func(w http.ResponseWriter, r *http.Request) {
		var content string
		if r.Header.Get("Content-Type") == "application/json" {
			var req struct {
				Content string `json:"content"`
			}
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				http.Error(w, "Bad Request", http.StatusBadRequest)
				return
			}
			content = req.Content
		} else {
			var b bytes.Buffer
			if _, err := io.Copy(&b, r.Body); err != nil {
				http.Error(w, "Bad Request", http.StatusBadRequest)
				return
			}
			content = b.String()
		}

		tokens := runner.Tokenizer.Encode(content, true)

		if err := json.NewEncoder(w).Encode(struct {
			Tokens []int32 `json:"tokens"`
		}{Tokens: tokens}); err != nil {
			slog.Error("Failed to encode response", "error", err)
			http.Error(w, "Internal Server Error", http.StatusInternalServerError)
			return
		}
	}
	mux.HandleFunc("POST /v1/tokenize", tokenizeHandler)
	mux.HandleFunc("POST /tokenize", tokenizeHandler)

	return runner.Run("127.0.0.1", strconv.Itoa(port), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		recorder := &statusRecorder{ResponseWriter: w, code: http.StatusOK}
		t := time.Now()
		mux.ServeHTTP(recorder, r)

		var level slog.Level
		switch {
		case recorder.code >= 500:
			level = slog.LevelError
		case recorder.code >= 400:
			level = slog.LevelWarn
		case recorder.code >= 300:
			return
		}

		slog.Log(r.Context(), level, "ServeHTTP", "method", r.Method, "path", r.URL.Path, "took", time.Since(t), "status", recorder.Status())
	}))
}

type statusRecorder struct {
	http.ResponseWriter
	code int
}

func (w *statusRecorder) WriteHeader(code int) {
	w.code = code
	w.ResponseWriter.WriteHeader(code)
}

func (w *statusRecorder) Status() string {
	return strconv.Itoa(w.code) + " " + http.StatusText(w.code)
}

func (w *statusRecorder) Flush() {
	if f, ok := w.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}
