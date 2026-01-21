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
- `tavily_key` optional: Tavily API key for web tools support.
- `tavily_url` optional: Tavily base URL (defaults to `https://api.tavily.com`).

Example:

```json
{
  "ak": "your-proxy-api-key",
  "port": 8888,
  "upstream_timeout_seconds": 300,
  "log_body_max_chars": 4096,
  "log_stream_text_preview_chars": 256,
  "tavily_key": "tvly-...",
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

Environment variables can override or complement `config.json`:

- `CONFIG_PATH`: path to config file (default `config.json`)
- `TAVILY_API_KEY`: overrides `tavily_key` in config
- `TAVILY_BASE_URL`: overrides `tavily_url` in config
- `TAVILY_PROXY_ADDRESS` or `LOCAL_PROXY_ADDRESS`: proxy for Tavily requests
- `TAVILY_MAX_RESULTS`: max search results (default `5`, max `20`)
- `TAVILY_SEARCH_DEPTH`: `basic` or `advanced` (default `basic`)
- `TAVILY_TOPIC`: `general`, `news`, or `finance` (default `general`)
- `TAVILY_FETCH_MAX_CHARS`: max characters for `web_fetch` (default `50000`, min `1000`)

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
    }
  ]
}
```

### POST /v1/messages

Sends a message to the upstream service.

- Inbound auth:
  - If `ak` is set in config, you must send `Authorization: Bearer <ak>` (or `x-api-key: <ak>`).
- Upstream auth:
  - Always sends `Authorization: Bearer <api_key>` to upstream.
- Web Tools:
  - Supports `web_search` and `web_fetch` if `tavily_key` or `TAVILY_API_KEY` is provided.

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

### GET /status

Health check endpoint.

Response:

```json
{
  "message": "claude-proxy",
  "health": "ok"
}
```

## Build

This project uses only Go stdlib (no external deps).

Linux (amd64):
```bash
GOOS=linux GOARCH=amd64 go build -trimpath -ldflags "-s -w" -o dist/claude-proxy_linux_amd64 .
```

Windows (amd64):
```bash
GOOS=windows GOARCH=amd64 go build -trimpath -ldflags "-s -w" -o dist/claude-proxy_windows_amd64.exe .
```

## Notes / Limitations

- Multiple providers can be configured, each with their own `base_url`, `api_key`, and models.
- Web tools (`web_search`, `web_fetch`) are powered by Tavily and require an API key.
- Streaming conversion supports `delta.content` text and `delta.tool_calls` tool-use blocks.
- Logs show forwarded request bodies; keep `log_body_max_chars` small.
- The `/v1/models` endpoint returns a static list from config.

