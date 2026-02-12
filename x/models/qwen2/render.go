//go:build mlx

package qwen2

import (
	"strings"

	"github.com/ollama/ollama/api"
)

// Renderer renders messages for Qwen2 models.
//
// Qwen2.5 uses ChatML format with the following tokens:
// - <|im_start|>system\n{content}<|im_end|>\n
// - <|im_start|>user\n{content}<|im_end|>\n
// - <|im_start|>assistant\n{content}<|im_end|>\n
type Renderer struct{}

// Render renders messages into the Qwen2 ChatML format.
func (r *Renderer) Render(messages []api.Message, tools []api.Tool, thinkValue *api.ThinkValue) (string, error) {
	var sb strings.Builder

	for _, message := range messages {
		switch message.Role {
		case "system":
			sb.WriteString("<|im_start|>system\n")
			sb.WriteString(message.Content)
			sb.WriteString("<|im_end|>\n")
		case "user":
			sb.WriteString("<|im_start|>user\n")
			sb.WriteString(message.Content)
			sb.WriteString("<|im_end|>\n")
		case "assistant":
			sb.WriteString("<|im_start|>assistant\n")
			if message.Content != "" {
				sb.WriteString(message.Content)
			}
			sb.WriteString("<|im_end|>\n")
		}
	}

	// Start the assistant response
	sb.WriteString("<|im_start|>assistant\n")

	return sb.String(), nil
}

// FormatPrompt applies a simple single-turn prompt format for raw prompts.
func (m *Model) FormatPrompt(prompt string) string {
	return prompt
}

// NewRenderer returns a new Renderer for formatting multi-turn conversations.
func (m *Model) NewRenderer() *Renderer {
	return &Renderer{}
}
