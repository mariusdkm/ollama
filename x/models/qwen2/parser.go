//go:build mlx

package qwen2

import (
	"github.com/ollama/ollama/api"
)

// Parser parses Qwen2 model output.
// For the initial implementation, this is a simple passthrough parser
// that returns all output as content without parsing tool calls or thinking.
type Parser struct{}

// HasToolSupport returns false as tool support is deferred for the initial implementation.
func (p *Parser) HasToolSupport() bool {
	return false
}

// HasThinkingSupport returns false as thinking support is deferred for the initial implementation.
func (p *Parser) HasThinkingSupport() bool {
	return false
}

// Init initializes the parser with tools and thinking configuration.
func (p *Parser) Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool {
	return tools
}

// Add processes new output text and returns it as content (passthrough).
func (p *Parser) Add(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error) {
	return s, "", nil, nil
}

// NewParser returns a new Parser for extracting content from output.
func (m *Model) NewParser() *Parser {
	return &Parser{}
}
