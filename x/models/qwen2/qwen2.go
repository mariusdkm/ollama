//go:build mlx

// Package qwen2 provides the Qwen2 implementation for MLX.
package qwen2

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"strings"

	"github.com/ollama/ollama/x/imagegen/manifest"
	"github.com/ollama/ollama/x/imagegen/tokenizer"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/nn"
)

// RopeScaling holds RoPE scaling configuration
type RopeScaling struct {
	Factor                        float32 `json:"factor"`
	Type                          string  `json:"type"`
	OriginalMaxPositionEmbeddings int32   `json:"original_max_position_embeddings"`
}

// Config holds Qwen2 model configuration
type Config struct {
	HiddenSize            int32   `json:"hidden_size"`
	NumHiddenLayers       int32   `json:"num_hidden_layers"`
	IntermediateSize      int32   `json:"intermediate_size"`
	NumAttentionHeads     int32   `json:"num_attention_heads"`
	NumKeyValueHeads      int32   `json:"num_key_value_heads"`
	VocabSize             int32   `json:"vocab_size"`
	RMSNormEps            float32 `json:"rms_norm_eps"`
	RopeTheta             float32 `json:"rope_theta"`
	MaxPositionEmbeddings int32   `json:"max_position_embeddings"`
	AttentionBias         bool    `json:"attention_bias"`
	TieWordEmbeddings     bool    `json:"tie_word_embeddings"`
	UseSlidingWindow      bool    `json:"use_sliding_window"`

	RopeScaling *RopeScaling `json:"rope_scaling"`

	// Quantization config from config.json (MLX-community models)
	QuantizationConfig *struct {
		GroupSize int `json:"group_size"`
		Bits      int `json:"bits"`
	} `json:"quantization_config"`

	// Quantization parameters (resolved during load)
	QuantGroupSize int    `json:"-"`
	QuantBits      int    `json:"-"`
	QuantMode      string `json:"-"`

	// Computed
	HeadDim  int32   `json:"-"` // hidden_size / num_attention_heads
	GQARatio int32   `json:"-"` // num_attention_heads / num_key_value_heads
	Scale    float32 `json:"-"` // 1/sqrt(HeadDim)
}

// Attention implements grouped-query attention
type Attention struct {
	QProj nn.LinearLayer
	KProj nn.LinearLayer
	VProj nn.LinearLayer
	OProj nn.LinearLayer
}

// Forward computes attention output
func (a *Attention) Forward(x *mlx.Array, c cache.Cache, B, L int32, cfg *Config) *mlx.Array {
	q := a.QProj.Forward(x)
	k := a.KProj.Forward(x)
	v := a.VProj.Forward(x)

	q = mlx.Reshape(q, B, L, cfg.NumAttentionHeads, cfg.HeadDim)
	q = mlx.Transpose(q, 0, 2, 1, 3)

	k = mlx.Reshape(k, B, L, cfg.NumKeyValueHeads, cfg.HeadDim)
	k = mlx.Transpose(k, 0, 2, 1, 3)

	v = mlx.Reshape(v, B, L, cfg.NumKeyValueHeads, cfg.HeadDim)
	v = mlx.Transpose(v, 0, 2, 1, 3)

	offset := 0
	if c != nil {
		offset = c.Offset()
	}
	q = mlx.RoPEWithBase(q, int(cfg.HeadDim), true, cfg.RopeTheta, 1.0, offset)
	k = mlx.RoPEWithBase(k, int(cfg.HeadDim), true, cfg.RopeTheta, 1.0, offset)

	if c != nil {
		k, v = c.Update(k, v)
	}

	k = nn.RepeatKV(k, cfg.GQARatio)
	v = nn.RepeatKV(v, cfg.GQARatio)

	out := mlx.ScaledDotProductAttentionCausal(q, k, v, cfg.Scale, L > 1)

	out = mlx.Reshape(mlx.Transpose(out, 0, 2, 1, 3), B, L, cfg.HiddenSize)

	return a.OProj.Forward(out)
}

// MLP implements SwiGLU feed-forward network
type MLP struct {
	GateProj nn.LinearLayer
	UpProj   nn.LinearLayer
	DownProj nn.LinearLayer
}

// Forward applies the SwiGLU MLP
func (m *MLP) Forward(x *mlx.Array) *mlx.Array {
	gate := mlx.SiLU(m.GateProj.Forward(x))
	up := m.UpProj.Forward(x)
	return m.DownProj.Forward(mlx.Mul(gate, up))
}

// TransformerBlock represents a single transformer layer
type TransformerBlock struct {
	Attention              *Attention
	MLP                    *MLP
	InputLayerNorm         *nn.RMSNorm
	PostAttentionLayerNorm *nn.RMSNorm
}

// Forward applies the transformer block
func (b *TransformerBlock) Forward(x *mlx.Array, c cache.Cache, B, L int32, cfg *Config) *mlx.Array {
	r := b.Attention.Forward(b.InputLayerNorm.Forward(x, cfg.RMSNormEps), c, B, L, cfg)
	h := mlx.Add(x, r)
	r = b.MLP.Forward(b.PostAttentionLayerNorm.Forward(h, cfg.RMSNormEps))
	return mlx.Add(h, r)
}

// Model represents the complete Qwen2 model
type Model struct {
	EmbedTokens nn.EmbeddingLayer
	Layers      []*TransformerBlock
	Norm        *nn.RMSNorm
	LMHead      nn.LinearLayer

	tok *tokenizer.Tokenizer
	*Config
}

// supportsGatherQMM returns true if the quantization mode has GatherQMM kernel support.
func supportsGatherQMM(mode string, bits int) bool {
	return mode == "affine" && (bits == 4 || bits == 8)
}

// quantizationParams returns groupSize, bits, mode for a quantization type string.
func quantizationParams(quantization string) (groupSize, bits int, mode string) {
	switch strings.ToUpper(quantization) {
	case "NVFP4":
		return 16, 4, "nvfp4"
	case "FP4", "Q4", "INT4":
		return 32, 4, "affine"
	case "MXFP8":
		return 32, 8, "mxfp8"
	case "FP8", "Q8", "INT8", "":
		return 64, 8, "affine"
	default:
		return 32, 8, "affine"
	}
}

// readBlobMetadata reads the __metadata__ from a safetensors blob header.
func readBlobMetadata(path string) (map[string]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var headerSize uint64
	if err := binary.Read(f, binary.LittleEndian, &headerSize); err != nil {
		return nil, err
	}
	if headerSize > 1024*1024 {
		return nil, fmt.Errorf("header too large: %d", headerSize)
	}

	data := make([]byte, headerSize)
	if _, err := io.ReadFull(f, data); err != nil {
		return nil, err
	}

	var header map[string]json.RawMessage
	if err := json.Unmarshal(data, &header); err != nil {
		return nil, err
	}

	metaRaw, ok := header["__metadata__"]
	if !ok {
		return nil, nil
	}

	var meta map[string]string
	if err := json.Unmarshal(metaRaw, &meta); err != nil {
		return nil, err
	}
	return meta, nil
}

// makeLinear creates a Linear or QuantizedLinear layer from the tensor map.
func makeLinear(tensors map[string]*mlx.Array, path string, cfg *Config) nn.LinearLayer {
	w := tensors[path+".weight"]
	if w == nil {
		return nil
	}

	scales := tensors[path+".weight_scale"]
	if scales != nil {
		qbiases := tensors[path+".weight_qbias"]
		bias := tensors[path+".bias"]
		return &nn.QuantizedLinear{
			Weight:    w,
			Scales:    scales,
			QBiases:   qbiases,
			Bias:      bias,
			GroupSize: cfg.QuantGroupSize,
			Bits:      cfg.QuantBits,
			Mode:      cfg.QuantMode,
		}
	}

	bias := tensors[path+".bias"]
	return nn.NewLinear(w, bias)
}

// LoadFromManifest loads a Qwen2 model from a manifest (Ollama blob storage).
func LoadFromManifest(modelManifest *manifest.ModelManifest) (*Model, error) {
	configData, err := modelManifest.ReadConfig("config.json")
	if err != nil {
		return nil, fmt.Errorf("load config: %w", err)
	}

	var cfg Config
	if err := json.Unmarshal(configData, &cfg); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}

	// Validate config
	if cfg.NumAttentionHeads == 0 || cfg.NumKeyValueHeads == 0 {
		return nil, fmt.Errorf("invalid config: num_attention_heads=%d, num_key_value_heads=%d", cfg.NumAttentionHeads, cfg.NumKeyValueHeads)
	}
	if cfg.HiddenSize%cfg.NumAttentionHeads != 0 {
		return nil, fmt.Errorf("hidden_size (%d) not divisible by num_attention_heads (%d)", cfg.HiddenSize, cfg.NumAttentionHeads)
	}
	if cfg.NumAttentionHeads%cfg.NumKeyValueHeads != 0 {
		return nil, fmt.Errorf("num_attention_heads (%d) not divisible by num_key_value_heads (%d)", cfg.NumAttentionHeads, cfg.NumKeyValueHeads)
	}
	if cfg.RopeTheta == 0 {
		return nil, fmt.Errorf("invalid config: rope_theta is 0 or missing")
	}
	if cfg.UseSlidingWindow {
		return nil, fmt.Errorf("sliding window attention not implemented")
	}

	// Compute derived fields
	cfg.HeadDim = cfg.HiddenSize / cfg.NumAttentionHeads
	cfg.GQARatio = cfg.NumAttentionHeads / cfg.NumKeyValueHeads
	cfg.Scale = float32(1.0 / math.Sqrt(float64(cfg.HeadDim)))

	// Load all tensors from manifest blobs into a flat map
	allTensors := make(map[string]*mlx.Array)
	seen := make(map[string]bool) // dedupe by digest
	var quantType string
	var quantGroupSize int

	for _, layer := range modelManifest.GetTensorLayers("") {
		if seen[layer.Digest] {
			continue
		}
		seen[layer.Digest] = true
		blobPath := modelManifest.BlobPath(layer.Digest)

		// Read quantization metadata from first blob
		if quantType == "" {
			if meta, err := readBlobMetadata(blobPath); err == nil && meta != nil {
				if qt := meta["quant_type"]; qt != "" {
					quantType = strings.ToUpper(qt)
				}
				if gs := meta["group_size"]; gs != "" {
					fmt.Sscanf(gs, "%d", &quantGroupSize)
				}
			}
		}

		for name, arr := range mlx.Load(blobPath) {
			// Map safetensors key naming to our naming convention.
			// Ollama's import system uses two naming conventions:
			// 1. Packed blobs (quantized during import): ".weight.scale" / ".weight.bias"
			// 2. Pre-quantized blobs (imported as-is): ".scales" / ".biases"
			// We normalize both to "_scale" / "_qbias" for makeLinear lookup.
			if strings.HasSuffix(name, ".scale") {
				baseName := strings.TrimSuffix(name, ".scale")
				allTensors[baseName+"_scale"] = arr
			} else if strings.HasSuffix(name, ".scales") {
				baseName := strings.TrimSuffix(name, ".scales")
				allTensors[baseName+".weight_scale"] = arr
			} else if strings.HasSuffix(name, ".biases") {
				baseName := strings.TrimSuffix(name, ".biases")
				allTensors[baseName+".weight_qbias"] = arr
			} else if strings.HasSuffix(name, ".bias") && !strings.HasSuffix(name, ".weight_qbias") {
				// Check if this is a quantization bias or a regular bias
				// by checking if there's a corresponding scale tensor
				baseName := strings.TrimSuffix(name, ".bias")
				if _, hasScale := allTensors[baseName+"_scale"]; hasScale {
					allTensors[baseName+"_qbias"] = arr
				} else {
					allTensors[name] = arr
				}
			} else {
				allTensors[name] = arr
			}
		}
	}

	// Set up quantization parameters.
	// Priority: blob __metadata__ > config.json quantization_config > defaults.
	if quantType != "" {
		_, cfg.QuantBits, cfg.QuantMode = quantizationParams(quantType)
		if quantGroupSize > 0 {
			cfg.QuantGroupSize = quantGroupSize
		} else {
			cfg.QuantGroupSize, _, _ = quantizationParams(quantType)
		}
	} else if cfg.QuantizationConfig != nil {
		// MLX-community pre-quantized models store params in config.json
		cfg.QuantGroupSize = cfg.QuantizationConfig.GroupSize
		cfg.QuantBits = cfg.QuantizationConfig.Bits
		cfg.QuantMode = "affine" // MLX-community uses affine quantization
	}

	// Load tokenizer
	tokData, err := modelManifest.ReadConfig("tokenizer.json")
	if err != nil {
		return nil, fmt.Errorf("load tokenizer config: %w", err)
	}

	tokConfig := &tokenizer.TokenizerConfig{
		ConfigJSON: configData,
	}

	if genConfigData, err := modelManifest.ReadConfig("generation_config.json"); err == nil {
		tokConfig.GenerationConfigJSON = genConfigData
	}

	if tokConfigData, err := modelManifest.ReadConfig("tokenizer_config.json"); err == nil {
		tokConfig.TokenizerConfigJSON = tokConfigData
	}

	tok, err := tokenizer.LoadFromBytesWithConfig(tokData, tokConfig)
	if err != nil {
		return nil, fmt.Errorf("parse tokenizer: %w", err)
	}

	m := &Model{
		Layers: make([]*TransformerBlock, cfg.NumHiddenLayers),
		Config: &cfg,
		tok:    tok,
	}

	// Load embedding (may be quantized)
	if w := allTensors["model.embed_tokens.weight"]; w != nil {
		scales := allTensors["model.embed_tokens.weight_scale"]
		if scales != nil {
			qbiases := allTensors["model.embed_tokens.weight_qbias"]
			m.EmbedTokens = nn.NewQuantizedEmbedding(w, scales, qbiases, cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode)
		} else {
			m.EmbedTokens = nn.NewEmbedding(w)
		}
	}

	// Load final norm
	if w := allTensors["model.norm.weight"]; w != nil {
		m.Norm = nn.NewRMSNorm(w, cfg.RMSNormEps)
	}

	// Load LM head (or tie to embedding weights)
	m.LMHead = makeLinear(allTensors, "lm_head", &cfg)
	if m.LMHead == nil && cfg.TieWordEmbeddings {
		if w := allTensors["model.embed_tokens.weight"]; w != nil {
			m.LMHead = nn.NewLinear(w, nil)
		}
	}

	// Validate required top-level weights
	if m.EmbedTokens == nil {
		return nil, fmt.Errorf("missing model.embed_tokens.weight")
	}
	if m.Norm == nil {
		return nil, fmt.Errorf("missing model.norm.weight")
	}
	if m.LMHead == nil {
		return nil, fmt.Errorf("missing lm_head.weight (and tie_word_embeddings is %v)", cfg.TieWordEmbeddings)
	}

	// Load layers
	for i := int32(0); i < cfg.NumHiddenLayers; i++ {
		prefix := fmt.Sprintf("model.layers.%d", i)

		block := &TransformerBlock{}

		// Load attention
		attn := &Attention{}
		attn.QProj = makeLinear(allTensors, prefix+".self_attn.q_proj", &cfg)
		attn.KProj = makeLinear(allTensors, prefix+".self_attn.k_proj", &cfg)
		attn.VProj = makeLinear(allTensors, prefix+".self_attn.v_proj", &cfg)
		attn.OProj = makeLinear(allTensors, prefix+".self_attn.o_proj", &cfg)
		block.Attention = attn

		// Load MLP
		mlp := &MLP{}
		mlp.GateProj = makeLinear(allTensors, prefix+".mlp.gate_proj", &cfg)
		mlp.UpProj = makeLinear(allTensors, prefix+".mlp.up_proj", &cfg)
		mlp.DownProj = makeLinear(allTensors, prefix+".mlp.down_proj", &cfg)
		block.MLP = mlp

		// Load layer norms
		if w := allTensors[prefix+".input_layernorm.weight"]; w != nil {
			block.InputLayerNorm = nn.NewRMSNorm(w, cfg.RMSNormEps)
		}
		if w := allTensors[prefix+".post_attention_layernorm.weight"]; w != nil {
			block.PostAttentionLayerNorm = nn.NewRMSNorm(w, cfg.RMSNormEps)
		}

		// Validate layer weights
		if block.Attention.QProj == nil || block.Attention.KProj == nil ||
			block.Attention.VProj == nil || block.Attention.OProj == nil {
			return nil, fmt.Errorf("layer %d: missing attention projection weights", i)
		}
		if block.MLP.GateProj == nil || block.MLP.UpProj == nil || block.MLP.DownProj == nil {
			return nil, fmt.Errorf("layer %d: missing MLP projection weights", i)
		}
		if block.InputLayerNorm == nil || block.PostAttentionLayerNorm == nil {
			return nil, fmt.Errorf("layer %d: missing layernorm weights", i)
		}

		m.Layers[i] = block
	}

	mlx.Eval(mlx.Collect(m)...)

	return m, nil
}

// Forward computes the forward pass of the model
func (m *Model) Forward(tokens *mlx.Array, caches []cache.Cache) *mlx.Array {
	dims := tokens.Dims()
	B, L := int32(dims[0]), int32(dims[1])

	h := m.EmbedTokens.Forward(tokens)

	for i, layer := range m.Layers {
		var c cache.Cache
		if caches != nil {
			c = caches[i]
		}
		h = layer.Forward(h, c, B, L, m.Config)
	}

	h = m.Norm.Forward(h, m.RMSNormEps)
	return h
}

// Unembed applies the LM head to get logits.
func (m *Model) Unembed(x *mlx.Array) *mlx.Array {
	return m.LMHead.Forward(x)
}

// NumLayers returns the number of transformer layers
func (m *Model) NumLayers() int { return len(m.Layers) }

// MaxContextLength returns the maximum context length
func (m *Model) MaxContextLength() int32 { return m.MaxPositionEmbeddings }

// VocabSize returns the vocabulary size
func (m *Model) VocabSize() int32 { return m.Config.VocabSize }

// Tokenizer returns the model's tokenizer
func (m *Model) Tokenizer() *tokenizer.Tokenizer { return m.tok }

// NewCache creates a new KV cache for the model
func (m *Model) NewCache(maxSeqLen int32) []cache.Cache {
	caches := make([]cache.Cache, len(m.Layers))
	for i := range caches {
		caches[i] = cache.NewKVCache()
	}
	return caches
}
