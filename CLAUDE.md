# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A generic API proxy that converts Anthropic/Claude API requests to OpenAI Chat Completions format and forwards to any OpenAI-compatible upstream service. Supports multiple providers with flexible model mapping.

**Key Architecture:**
- Single-file implementation (`main.go`) using only Go stdlib (no external dependencies)
- Bidirectional protocol translation: Anthropic Messages API ↔ OpenAI Chat Completions API
- Multi-provider support: configure multiple upstream services with independent API keys and models
- Model ID mapping: client-facing model IDs (`id`) are mapped to upstream remote IDs (`remote_id`)
- Supports both streaming and non-streaming responses
- Handles tool calls (function calling) in both directions

## Development Commands

### Running the Server

```bash
go run .
```

The server listens on `:8888` by default (configurable via `port` in config.json).

### Building

Standard build:
```bash
go build -o claude-proxy .
```

Cross-platform builds:
```bash
# Linux AMD64
GOOS=linux GOARCH=amd64 go build -trimpath -ldflags "-s -w" -o dist/claude-proxy_linux_amd64 .

# Windows AMD64
GOOS=windows GOARCH=amd64 go build -trimpath -ldflags "-s -w" -o dist/claude-proxy_windows_amd64.exe .

# macOS ARM64
GOOS=darwin GOARCH=arm64 go build -trimpath -ldflags "-s -w" -o dist/claude-proxy_darwin_arm64 .
```

If your environment blocks the default Go build cache:
```bash
export GOCACHE=/tmp/tmp-go-build-cache
export GOMODCACHE=/tmp/tmp-gomodcache
```

## Configuration

### config.json

Required configuration file (default path: `config.json`):

```json
{
  "ak": "your-proxy-api-key",
  "port": 8888,
  "upstream_timeout_seconds": 300,
  "log_body_max_chars": 4096,
  "log_stream_text_preview_chars": 256,
  "providers": [
    {
      "base_url": "https://api.example.com",
      "api_key": "your-upstream-api-key",
      "models": [
        {
          "id": "glm",
          "display_name": "glm4.7",
          "remote_id": "deepseek-chat"
        }
      ]
    }
  ]
}
```

**Configuration fields:**
- `ak` (optional): API key for inbound authentication. Clients must send `Authorization: Bearer <ak>` or `x-api-key: <ak>`.
- `port` (optional): Server port (default: `8888`)
- `upstream_timeout_seconds` (optional): Upstream request timeout (default: `300`)
- `log_body_max_chars` (optional): Max chars to log in request/response bodies (default: `4096`, set to `0` to disable)
- `log_stream_text_preview_chars` (optional): Max chars to log for streaming response preview (default: `256`, set to `0` to disable)
- `providers` (required): Array of upstream service providers
  - `base_url` (required): Base URL of upstream service (endpoint will be `{base_url}/v1/chat/completions`)
  - `api_key` (required): API key for upstream authentication (sent as `Authorization: Bearer ...`)
  - `models` (required): Array of model configurations
    - `id` (required): Client-facing model identifier
    - `display_name` (optional): Display name (defaults to `id`)
    - `remote_id` (optional): Model ID sent to upstream (defaults to `id`)

**IMPORTANT:** Never commit real API keys to version control.

### Environment Variables

- `CONFIG_PATH` - Path to config.json (default: `config.json`)

## Code Architecture

### Request Flow

1. **Inbound Request** (`handleMessages` at main.go:218)
   - Receives Anthropic Messages API format at `POST /v1/messages`
   - Optional authentication check via `ak` in config
   - Decodes `anthropicMessageRequest` struct
   - Maps client-facing model ID to provider's remote ID via `modelMap`

2. **Protocol Translation** (`convertAnthropicToOpenAI` at main.go:711)
   - Converts Anthropic message format to OpenAI Chat Completions format
   - Handles system messages, user/assistant messages, tool calls, and images
   - Maps Anthropic content blocks to OpenAI message parts

3. **Upstream Request** (`doUpstreamJSON` or `proxyStream` at main.go:311, 341)
   - Constructs upstream URL: `{base_url}/v1/chat/completions`
   - Adds `Authorization: Bearer <api_key>` header from the selected provider
   - Non-streaming: reads full response
   - Streaming: proxies SSE events with real-time translation

4. **Response Translation** (`convertOpenAIToAnthropic` at main.go:1016)
   - Converts OpenAI response back to Anthropic format
   - Maps finish reasons: `stop` → `end_turn`, `length` → `max_tokens`, `tool_calls` → `tool_use`
   - Reconstructs Anthropic content blocks from OpenAI message structure

### Streaming Architecture

The streaming implementation (`proxyStream` at main.go:341) performs real-time SSE translation:

- Reads OpenAI SSE chunks line-by-line
- Maintains state for content blocks (text and tool_use)
- Emits Anthropic SSE events: `message_start`, `content_block_start`, `content_block_delta`, `content_block_stop`, `message_delta`, `message_stop`
- Handles tool call deltas by accumulating partial JSON arguments

**Key State Management:**
- `nextContentBlockIndex` - Tracks content block indices
- `currentContentBlockIndex` - Currently open block
- `toolStates` - Maps OpenAI tool indices to Anthropic content block state
- Automatically closes blocks when switching between text and tool_use

### Provider and Model Mapping

At startup (`loadConfig` at main.go:100), the config is parsed and a `modelMap` is built:

- Each provider's models are registered with their client-facing `id` as the key
- The `modelMapping` struct stores: `ProviderIndex`, `RemoteID`, and `DisplayName`
- When a request comes in with `model: "glm"`, the proxy looks up the provider and maps it to the upstream `remote_id` (e.g., `deepseek-chat`)
- This allows different upstream providers to use different model IDs while presenting a unified interface to clients

### Authentication

Two-layer authentication model:

1. **Inbound Auth** (`checkInboundAuth` at main.go:299)
   - Optional: only if `ak` is set in config
   - Accepts `Authorization: Bearer <key>` or `x-api-key: <key>` headers
   - Uses constant-time comparison to prevent timing attacks

2. **Upstream Auth**
   - Always sends `Authorization: Bearer <api_key>` from the selected provider
   - Each provider has its own `api_key` in config

### Logging and Sanitization

Request/response bodies are logged with sanitization (`sanitizeOpenAIRequest` at main.go:1160):
- Redacts base64 image data URLs as `data:<redacted>`
- Truncates long strings based on `log_body_max_chars` config
- Preserves structure for debugging while protecting sensitive data

## Using with Claude Code

To use this proxy with Claude Code CLI, set these environment variables before running `claude`:

```bash
export ANTHROPIC_BASE_URL=http://localhost:8888
export ANTHROPIC_AUTH_TOKEN=your-proxy-api-key
export ANTHROPIC_DEFAULT_HAIKU_MODEL=glm
export ANTHROPIC_DEFAULT_SONNET_MODEL=glm
export ANTHROPIC_DEFAULT_OPUS_MODEL=glm

claude
```

Or use a different model (e.g., minimax):
```bash
export ANTHROPIC_DEFAULT_HAIKU_MODEL=minimax
export ANTHROPIC_DEFAULT_SONNET_MODEL=minimax
export ANTHROPIC_DEFAULT_OPUS_MODEL=minimax

claude
```

## Known Limitations

- Streaming conversion fully supports text deltas and tool call deltas
- Other Anthropic content block types (thinking blocks, etc.) are not implemented
- Request/response bodies are logged; keep `log_body_max_chars` small and avoid secrets in prompts
- No external dependencies means no advanced HTTP features (connection pooling, retry logic, etc.)
- The `/v1/models` endpoint returns a static list from config, not from the upstream service
