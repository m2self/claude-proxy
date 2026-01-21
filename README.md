# claude-proxy (Go)

A generic API proxy that converts Anthropic/Claude API requests to OpenAI Chat Completions format and forwards to any OpenAI-compatible upstream service.

Expose `POST /v1/messages` (Anthropic/Claude style), convert to OpenAI Chat Completions, and proxy to upstream (configured via `config.json`).

## Config

Edit `config.json`:

- `ak` optional: API key for inbound authentication. If set, clients must send `Authorization: Bearer <ak>` or `x-api-key: <ak>` header.
- `port` optional: server port (default `8888`)
- `upstream_timeout_seconds` optional: upstream request timeout in seconds (default `300`)
- `log_body_max_chars` optional: maximum characters to log for request/response bodies (default `4096`, set to `0` to disable)
- `log_stream_text_preview_chars` optional: maximum characters to log for streaming response preview (default `256`, set to `0` to disable)
- `providers` required: array of upstream service providers
  - `base_url` required: base URL of upstream service (e.g., `https://api.example.com`)
    - The completions endpoint will be constructed as `{base_url}/v1/chat/completions`
  - `api_key` required: used for upstream auth, sent as `Authorization: Bearer ...`
  - `models` required: array of model objects to expose via `/v1/models` endpoint
    - `id` required: model identifier used by clients (e.g., `glm`)
    - `display_name` optional: display name (defaults to `id` if not provided)
    - `remote_id` optional: model ID sent to upstream service (defaults to `id` if not provided)

Example:

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
        },
        {
          "id": "minimax",
          "display_name": "minimax2.1",
          "remote_id": "mimo-v2-flash"
        }
      ]
    }
  ]
}
```

Do not commit your real `ak` or `api_key` values.

## Env

- `CONFIG_PATH` default `config.json` (relative to working directory)

## Run

```bash
go run .
```

Or build and run:

```bash
go build -o claude-proxy
./claude-proxy
```

## CLAUDE CODE

use glm model:
```bash
export ANTHROPIC_BASE_URL=http://localhost:8888
export ANTHROPIC_AUTH_TOKEN=your-proxy-api-key
export ANTHROPIC_DEFAULT_HAIKU_MODEL=glm
export ANTHROPIC_DEFAULT_SONNET_MODEL=glm
export ANTHROPIC_DEFAULT_OPUS_MODEL=glm

claude
```

use minimax model:
```bash
export ANTHROPIC_BASE_URL=http://localhost:8888
export ANTHROPIC_AUTH_TOKEN=your-proxy-api-key
export ANTHROPIC_DEFAULT_HAIKU_MODEL=minimax
export ANTHROPIC_DEFAULT_SONNET_MODEL=minimax
export ANTHROPIC_DEFAULT_OPUS_MODEL=minimax

claude
```

## API

### GET /v1/models

Returns the list of available models configured in `config.json`.

- Inbound auth:
  - If `ak` is set in config, you must send `Authorization: Bearer <ak>` (or `x-api-key: <ak>`).

Response format (Anthropic style):

```json
{
  "object": "list",
  "data": [
    {
      "id": "glm",
      "object": "model",
      "created": 1234567890,
      "display_name": "glm4.7"
    },
    {
      "id": "minimax",
      "object": "model",
      "created": 1234567890,
      "display_name": "minimax2.1"
    }
  ]
}
```

Example:

```bash
curl -sS http://127.0.0.1:8888/v1/models \
  -H 'Authorization: Bearer your-proxy-api-key'
```

### POST /v1/messages

Sends a message to the upstream service.

- Inbound auth:
  - If `ak` is set in config, you must send `Authorization: Bearer <ak>` (or `x-api-key: <ak>`).
- Upstream auth:
  - Always sends `Authorization: Bearer <api_key>` to upstream.
- Model mapping:
  - The `model` field in the request uses the client-side `id` (e.g., `glm`)
  - The proxy converts it to the upstream `remote_id` (e.g., `deepseek-chat`) before forwarding

Example (non-stream):

```bash
curl -sS http://127.0.0.1:8888/v1/messages \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer your-proxy-api-key' \
  -d '{
    "model": "glm",
    "max_tokens": 256,
    "messages": [{"role": "user", "content": "hello"}]
  }'
```

Example (stream):

```bash
curl -N http://127.0.0.1:8888/v1/messages \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer your-proxy-api-key' \
  -d '{
    "model": "glm",
    "max_tokens": 256,
    "stream": true,
    "messages": [{"role": "user", "content": "hello"}]
  }'
```

### GET /

Health check endpoint.

Response:

```json
{
  "message": "claude-proxy",
  "health": "ok"
}
```

## Build

This project uses only Go stdlib (no external deps). If your environment blocks the default Go build cache path, set:

```bash
export GOCACHE=/tmp/tmp-go-build-cache
export GOMODCACHE=/tmp/tmp-gomodcache
```

Linux (amd64):

```bash
mkdir -p dist
GOOS=linux GOARCH=amd64 go build -trimpath -ldflags "-s -w" -o dist/claude-proxy_linux_amd64 .
```

Windows (amd64):

```bash
mkdir -p dist
GOOS=windows GOARCH=amd64 go build -trimpath -ldflags "-s -w" -o dist/claude-proxy_windows_amd64.exe .
```

macOS (arm64):

```bash
mkdir -p dist
GOOS=darwin GOARCH=arm64 go build -trimpath -ldflags "-s -w" -o dist/claude-proxy_darwin_arm64 .
```

## Notes / Limitations

- Multiple providers can be configured, each with their own `base_url`, `api_key`, and models
- Model IDs are mapped from client-side `id` to upstream `remote_id` before forwarding requests
- Streaming conversion supports `delta.content` text and `delta.tool_calls` tool-use blocks; other Anthropic blocks are not fully implemented
- Logs show forwarded request bodies; keep `log_body_max_chars` small and avoid secrets in prompts
- The `/v1/models` endpoint returns a static list from config, not from the upstream service
- All configuration is read from `config.json`; environment variable overrides are not supported (except `CONFIG_PATH`)
