package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/subtle"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/url"
	"os"
	"regexp"
	"strconv"
	"strings"
	"time"
)

type providerConfig struct {
	BaseURL string        `json:"base_url"`
	APIKey  string        `json:"api_key"`
	Models  []modelConfig `json:"models"`
}

type modelConfig struct {
	ID          string `json:"id"`
	DisplayName string `json:"display_name"`
	RemoteID    string `json:"remote_id"`
}

type fileConfig struct {
	AK                        string           `json:"ak"`
	Port                      int              `json:"port"`
	UpstreamTimeoutSeconds    int              `json:"upstream_timeout_seconds"`
	LogBodyMaxChars           int              `json:"log_body_max_chars"`
	LogStreamTextPreviewChars int              `json:"log_stream_text_preview_chars"`
	Providers                 []providerConfig `json:"providers"`

	// --- Tavily web tools ---
	TavilyKey string `json:"tavily_key,omitempty"`
	TavilyURL string `json:"tavily_url,omitempty"`
}

type serverConfig struct {
	addr                string
	serverAPIKey        string
	timeout             time.Duration
	logBodyMax          int
	logStreamPreviewMax int
	providers           []providerConfig
	modelMap            map[string]modelMapping

	// --- Tavily web tools ---
	tavilyAPIKey        string
	tavilyBaseURL       string
	tavilyProxyURL      string
	tavilyMaxResults    int
	tavilySearchDepth   string
	tavilyTopic         string
	tavilyFetchMaxRunes int
}

type modelMapping struct {
	ProviderIndex int
	RemoteID      string
	DisplayName   string
}

func main() {
	cfg, err := loadConfig()
	if err != nil {
		log.Fatalf("config error: %v", err)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/models", func(w http.ResponseWriter, r *http.Request) {
		handleModels(w, r, cfg)
	})
	mux.HandleFunc("POST /v1/messages", func(w http.ResponseWriter, r *http.Request) {
		handleMessages(w, r, cfg)
	})
	mux.HandleFunc("/status", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"message": "claude-proxy",
			"health":  "ok",
		})
	})

	srv := &http.Server{
		Addr:              cfg.addr,
		Handler:           mux,
		ReadHeaderTimeout: 10 * time.Second,
		ReadTimeout:       60 * time.Second,
		WriteTimeout:      0, // allow streaming
		IdleTimeout:       60 * time.Second,
	}

	log.Printf("listening on %s", cfg.addr)
	log.Printf("providers: %d", len(cfg.providers))
	for i, p := range cfg.providers {
		log.Printf("  provider[%d]: %s (models: %d)", i, p.BaseURL, len(p.Models))
	}
	if cfg.serverAPIKey != "" {
		log.Printf("inbound auth: enabled")
	} else {
		log.Printf("inbound auth: disabled (AK missing / not set)")
	}
	if cfg.tavilyAPIKey != "" {
		log.Printf("tavily web tools: enabled (base=%s proxy=%q)", cfg.tavilyBaseURL, cfg.tavilyProxyURL)
	} else {
		log.Printf("tavily web tools: disabled (no tavily_key / TAVILY_API_KEY)")
	}
	log.Fatal(srv.ListenAndServe())
}

func loadConfig() (*serverConfig, error) {
	fc, err := loadFileConfig(strings.TrimSpace(envOr("CONFIG_PATH", "config.json")))
	if err != nil {
		return nil, err
	}

	port := 8888
	if fc.Port > 0 {
		port = fc.Port
	}
	addr := fmt.Sprintf(":%d", port)

	serverAPIKey := strings.TrimSpace(fc.AK)

	timeout := 5 * time.Minute
	if fc.UpstreamTimeoutSeconds > 0 {
		timeout = time.Duration(fc.UpstreamTimeoutSeconds) * time.Second
	}

	logBodyMax := 4096
	if fc.LogBodyMaxChars > 0 {
		logBodyMax = fc.LogBodyMaxChars
	}

	logStreamPreviewMax := 256
	if fc.LogStreamTextPreviewChars > 0 {
		logStreamPreviewMax = fc.LogStreamTextPreviewChars
	}

	if len(fc.Providers) == 0 {
		return nil, errors.New("no providers configured in config.json")
	}

	modelMap := make(map[string]modelMapping)
	for i, p := range fc.Providers {
		if strings.TrimSpace(p.BaseURL) == "" {
			return nil, fmt.Errorf("provider[%d]: missing base_url", i)
		}
		if strings.TrimSpace(p.APIKey) == "" {
			return nil, fmt.Errorf("provider[%d]: missing api_key", i)
		}
		for j, m := range p.Models {
			if strings.TrimSpace(m.ID) == "" {
				return nil, fmt.Errorf("provider[%d].models[%d]: missing id", i, j)
			}
			if _, exists := modelMap[m.ID]; exists {
				return nil, fmt.Errorf("duplicate model id: %q", m.ID)
			}
			remoteID := strings.TrimSpace(m.RemoteID)
			if remoteID == "" {
				remoteID = m.ID
			}
			displayName := strings.TrimSpace(m.DisplayName)
			if displayName == "" {
				displayName = m.ID
			}
			modelMap[m.ID] = modelMapping{
				ProviderIndex: i,
				RemoteID:      remoteID,
				DisplayName:   displayName,
			}
		}
	}

	// --- Tavily config (env can override config file) ---
	tavilyKey := strings.TrimSpace(envOr("TAVILY_API_KEY", fc.TavilyKey))
	tavilyBase := strings.TrimSpace(envOr("TAVILY_BASE_URL", fc.TavilyURL))
	if tavilyBase == "" {
		tavilyBase = "https://api.tavily.com"
	}
	tavilyProxy := strings.TrimSpace(envOr("TAVILY_PROXY_ADDRESS", envOr("LOCAL_PROXY_ADDRESS", "")))

	tavilyMaxResults := 5
	if raw := strings.TrimSpace(envOr("TAVILY_MAX_RESULTS", "")); raw != "" {
		if n, err := strconv.Atoi(raw); err == nil && n > 0 && n <= 20 {
			tavilyMaxResults = n
		}
	}
	tavilySearchDepth := strings.TrimSpace(envOr("TAVILY_SEARCH_DEPTH", "basic")) // basic|advanced
	tavilyTopic := strings.TrimSpace(envOr("TAVILY_TOPIC", "general"))            // general|news|finance

	tavilyFetchMaxRunes := 50000
	if raw := strings.TrimSpace(envOr("TAVILY_FETCH_MAX_CHARS", "")); raw != "" {
		if n, err := strconv.Atoi(raw); err == nil && n >= 1000 {
			tavilyFetchMaxRunes = n
		}
	}

	return &serverConfig{
		addr:                addr,
		serverAPIKey:        serverAPIKey,
		timeout:             timeout,
		logBodyMax:          logBodyMax,
		logStreamPreviewMax: logStreamPreviewMax,
		providers:           fc.Providers,
		modelMap:            modelMap,

		tavilyAPIKey:        tavilyKey,
		tavilyBaseURL:       tavilyBase,
		tavilyProxyURL:      tavilyProxy,
		tavilyMaxResults:    tavilyMaxResults,
		tavilySearchDepth:   tavilySearchDepth,
		tavilyTopic:         tavilyTopic,
		tavilyFetchMaxRunes: tavilyFetchMaxRunes,
	}, nil
}

func loadFileConfig(path string) (*fileConfig, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", path, err)
	}
	var fc fileConfig
	if err := json.Unmarshal(b, &fc); err != nil {
		return nil, fmt.Errorf("parse %s: %w", path, err)
	}
	return &fc, nil
}

func envOr(key, fallback string) string {
	if v, ok := os.LookupEnv(key); ok {
		return v
	}
	return fallback
}

func handleModels(w http.ResponseWriter, r *http.Request, cfg *serverConfig) {
	if cfg.serverAPIKey != "" && !checkInboundAuth(r, cfg.serverAPIKey) {
		log.Printf("models endpoint: unauthorized")
		writeJSONError(w, http.StatusUnauthorized, "unauthorized")
		return
	}

	data := make([]map[string]any, 0, len(cfg.modelMap))
	for id, mapping := range cfg.modelMap {
		data = append(data, map[string]any{
			"id":           id,
			"object":       "model",
			"created":      1234567890,
			"display_name": mapping.DisplayName,
		})
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"object": "list",
		"data":   data,
	})
}

func handleMessages(w http.ResponseWriter, r *http.Request, cfg *serverConfig) {
	reqID := fmt.Sprintf("req_%d", time.Now().UnixNano())
	if cfg.serverAPIKey != "" && !checkInboundAuth(r, cfg.serverAPIKey) {
		log.Printf("[%s] inbound unauthorized", reqID)
		writeJSONError(w, http.StatusUnauthorized, "unauthorized")
		return
	}

	var anthropicReq anthropicMessageRequest
	if err := json.NewDecoder(r.Body).Decode(&anthropicReq); err != nil {
		log.Printf("[%s] invalid inbound json: %v", reqID, err)
		writeJSONError(w, http.StatusBadRequest, "invalid_json")
		return
	}
	if strings.TrimSpace(anthropicReq.Model) == "" {
		log.Printf("[%s] missing model", reqID)
		writeJSONError(w, http.StatusBadRequest, "missing_model")
		return
	}

	modelID := strings.TrimSpace(anthropicReq.Model)
	mapping, exists := cfg.modelMap[modelID]
	if !exists {
		log.Printf("[%s] unknown model: %s", reqID, modelID)
		writeJSONError(w, http.StatusBadRequest, "unknown_model")
		return
	}

	anthropicReq.Model = mapping.RemoteID

	if anthropicReq.MaxTokens == 0 {
		anthropicReq.MaxTokens = 1024
	}

	openaiReq, err := convertAnthropicToOpenAI(&anthropicReq)
	if err != nil {
		log.Printf("[%s] request conversion failed: %v", reqID, err)
		writeJSONError(w, http.StatusBadRequest, "request_conversion_failed")
		return
	}

	provider := cfg.providers[mapping.ProviderIndex]
	upstreamURL := strings.TrimSuffix(provider.BaseURL, "/") + "/v1/chat/completions"

	logForwardedRequest(reqID, cfg, anthropicReq, openaiReq, upstreamURL)

	// --- Tavily-backed web tools loop (web_search / web_fetch) ---
	if requestWantsWebTools(anthropicReq.Tools) {
		if cfg.tavilyAPIKey == "" {
			log.Printf("[%s] web tools requested but Tavily key missing", reqID)
			writeJSONError(w, http.StatusBadRequest, "missing_tavily_api_key")
			return
		}

		allowedFetchURLs := extractAllowedFetchURLsFromAnthropicReq(&anthropicReq)

		if anthropicReq.Stream {
			loopReq := openaiReq
			loopReq.Stream = false
			if err := runWebToolsLoopAndStream(w, r, cfg, reqID, loopReq, upstreamURL, provider.APIKey, allowedFetchURLs); err != nil {
				log.Printf("[%s] web-tools stream error: %v", reqID, err)
			}
			return
		}

		loopReq := openaiReq
		loopReq.Stream = false
		respMsg, err := runWebToolsLoop(r.Context(), cfg, reqID, loopReq, upstreamURL, provider.APIKey, allowedFetchURLs, anthropicReq.Tools)
		if err != nil {
			log.Printf("[%s] web-tools request failed: %v", reqID, err)
			writeJSONError(w, http.StatusBadGateway, "upstream_request_failed")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(respMsg)
		return
	}

	// --- Regular proxy ---
	if anthropicReq.Stream {
		if streamErr := proxyStream(w, r, cfg, reqID, openaiReq, upstreamURL, provider.APIKey); streamErr != nil {
			log.Printf("[%s] stream proxy error: %v", reqID, streamErr)
		}
		return
	}

	openaiRespBody, resp, err := doUpstreamJSON(r.Context(), cfg, openaiReq, upstreamURL, provider.APIKey)
	if err != nil {
		log.Printf("[%s] upstream request failed: %v", reqID, err)
		writeJSONError(w, http.StatusBadGateway, "upstream_request_failed")
		return
	}
	defer resp.Body.Close()
	log.Printf("[%s] upstream status=%d", reqID, resp.StatusCode)
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(resp.StatusCode)
		_, _ = w.Write(openaiRespBody)
		logForwardedUpstreamBody(reqID, cfg, openaiRespBody)
		return
	}

	var openaiResp openaiChatCompletionResponse
	if err := json.Unmarshal(openaiRespBody, &openaiResp); err != nil {
		log.Printf("[%s] invalid upstream json: %v", reqID, err)
		logForwardedUpstreamBody(reqID, cfg, openaiRespBody)
		writeJSONError(w, http.StatusBadGateway, "invalid_upstream_json")
		return
	}
	anthropicResp := convertOpenAIToAnthropic(openaiResp)
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(anthropicResp)
}

func checkInboundAuth(r *http.Request, expected string) bool {
	auth := strings.TrimSpace(r.Header.Get("Authorization"))
	if strings.HasPrefix(strings.ToLower(auth), "bearer ") {
		got := strings.TrimSpace(auth[len("bearer "):])
		return subtle.ConstantTimeCompare([]byte(got), []byte(expected)) == 1
	}
	if got := strings.TrimSpace(r.Header.Get("x-api-key")); got != "" {
		return subtle.ConstantTimeCompare([]byte(got), []byte(expected)) == 1
	}
	return false
}

func doUpstreamJSON(ctx context.Context, cfg *serverConfig, openaiReq openaiChatCompletionRequest, upstreamURL string, apiKey string) ([]byte, *http.Response, error) {
	bodyBytes, err := json.Marshal(openaiReq)
	if err != nil {
		return nil, nil, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, upstreamURL, bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, nil, err
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+apiKey)

	client := &http.Client{Timeout: cfg.timeout}
	resp, err := client.Do(req)
	if err != nil {
		return nil, nil, err
	}

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		_ = resp.Body.Close()
		return nil, nil, err
	}
	_ = resp.Body.Close()
	resp.Body = io.NopCloser(bytes.NewReader(respBody))
	return respBody, resp, nil
}

func proxyStream(w http.ResponseWriter, r *http.Request, cfg *serverConfig, reqID string, openaiReq openaiChatCompletionRequest, upstreamURL string, apiKey string) error {
	openaiReq.Stream = true

	bodyBytes, err := json.Marshal(openaiReq)
	if err != nil {
		return err
	}
	upReq, err := http.NewRequestWithContext(r.Context(), http.MethodPost, upstreamURL, bytes.NewReader(bodyBytes))
	if err != nil {
		return err
	}
	upReq.Header.Set("Content-Type", "application/json")
	upReq.Header.Set("Authorization", "Bearer "+apiKey)

	client := &http.Client{Timeout: 0} // streaming: no client timeout
	upResp, err := client.Do(upReq)
	if err != nil {
		writeJSONError(w, http.StatusBadGateway, "upstream_request_failed")
		return err
	}
	defer upResp.Body.Close()

	log.Printf("[%s] upstream status=%d (stream)", reqID, upResp.StatusCode)
	if upResp.StatusCode < 200 || upResp.StatusCode >= 300 {
		raw, _ := io.ReadAll(upResp.Body)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(upResp.StatusCode)
		_, _ = w.Write(raw)
		logForwardedUpstreamBody(reqID, cfg, raw)
		return fmt.Errorf("upstream status %d", upResp.StatusCode)
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)

	flusher, ok := w.(http.Flusher)
	if !ok {
		writeJSONError(w, http.StatusInternalServerError, "streaming_not_supported")
		return errors.New("http.Flusher not supported")
	}

	// Minimal OpenAI SSE -> Anthropic SSE conversion (text + tool deltas).
	encoder := func(event string, payload any) error {
		b, err := json.Marshal(payload)
		if err != nil {
			return err
		}
		if _, err := fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, string(b)); err != nil {
			return err
		}
		flusher.Flush()
		return nil
	}

	messageID := fmt.Sprintf("msg_%d", time.Now().UnixMilli())
	_ = encoder("message_start", map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id":            messageID,
			"type":          "message",
			"role":          "assistant",
			"model":         openaiReq.Model,
			"content":       []any{},
			"stop_reason":   nil,
			"stop_sequence": nil,
			"usage": map[string]any{
				"input_tokens":  0,
				"output_tokens": 0,
			},
		},
	})

	reader := bufio.NewReader(upResp.Body)
	chunkCount := 0
	textChars := 0
	toolDeltaChunks := 0
	toolArgsChars := 0
	var finishReason string
	var preview strings.Builder
	sawDone := false
	type toolState struct {
		contentBlockIndex int
		id                string
		name              string
	}
	toolStates := map[int]*toolState{}

	nextContentBlockIndex := 0
	currentContentBlockIndex := -1
	currentBlockType := "" // "text" | "tool_use"
	hasTextBlock := false

	assignContentBlockIndex := func() int {
		idx := nextContentBlockIndex
		nextContentBlockIndex++
		return idx
	}

	closeCurrentBlock := func() {
		if currentContentBlockIndex >= 0 {
			_ = encoder("content_block_stop", map[string]any{
				"type":  "content_block_stop",
				"index": currentContentBlockIndex,
			})
			currentContentBlockIndex = -1
			currentBlockType = ""
		}
	}

	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return err
		}
		line = strings.TrimRight(line, "\r\n")
		if line == "" || strings.HasPrefix(line, ":") {
			continue
		}
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "[DONE]" {
			sawDone = true
			break
		}

		var chunk openaiChatCompletionChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}
		if len(chunk.Choices) == 0 {
			continue
		}

		chunkCount++
		delta := chunk.Choices[0].Delta

		// Tool calls: OpenAI streaming sends tool call deltas with partial arguments.
		if len(delta.ToolCalls) > 0 {
			for _, tc := range delta.ToolCalls {
				toolDeltaChunks++
				toolIndex := tc.Index
				if toolIndex < 0 {
					toolIndex = 0
				}
				state := toolStates[toolIndex]

				tcID := strings.TrimSpace(tc.ID)
				if tcID == "" {
					tcID = fmt.Sprintf("call_%d_%d", time.Now().UnixMilli(), toolIndex)
				}
				tcName := strings.TrimSpace(tc.Function.Name)
				if tcName == "" {
					tcName = fmt.Sprintf("tool_%d", toolIndex)
				}

				if state == nil {
					// Close any currently open block (text/tool) before starting a new tool block.
					closeCurrentBlock()
					idx := assignContentBlockIndex()
					state = &toolState{contentBlockIndex: idx, id: tcID, name: tcName}
					toolStates[toolIndex] = state

					_ = encoder("content_block_start", map[string]any{
						"type":  "content_block_start",
						"index": idx,
						"content_block": map[string]any{
							"type":  "tool_use",
							"id":    state.id,
							"name":  state.name,
							"input": map[string]any{},
						},
					})
					currentContentBlockIndex = idx
					currentBlockType = "tool_use"
				} else {
					currentContentBlockIndex = state.contentBlockIndex
					currentBlockType = "tool_use"
				}

				argsPart := tc.Function.Arguments
				if argsPart != "" {
					toolArgsChars += len([]rune(argsPart))
					_ = encoder("content_block_delta", map[string]any{
						"type":  "content_block_delta",
						"index": state.contentBlockIndex,
						"delta": map[string]any{
							"type":         "input_json_delta",
							"partial_json": argsPart,
						},
					})
				}
			}
		}

		if delta.Content != nil && *delta.Content != "" {
			textChars += len([]rune(*delta.Content))
			if cfg.logStreamPreviewMax > 0 && preview.Len() < cfg.logStreamPreviewMax {
				preview.WriteString(takeFirstRunes(*delta.Content, cfg.logStreamPreviewMax-preview.Len()))
			}
			if currentBlockType != "" && currentBlockType != "text" {
				closeCurrentBlock()
			}
			if !hasTextBlock {
				hasTextBlock = true
				idx := assignContentBlockIndex()
				_ = encoder("content_block_start", map[string]any{
					"type":  "content_block_start",
					"index": idx,
					"content_block": map[string]any{
						"type": "text",
						"text": "",
					},
				})
				currentContentBlockIndex = idx
				currentBlockType = "text"
			}
			_ = encoder("content_block_delta", map[string]any{
				"type":  "content_block_delta",
				"index": currentContentBlockIndex,
				"delta": map[string]any{
					"type": "text_delta",
					"text": *delta.Content,
				},
			})
		}

		if chunk.Choices[0].FinishReason != nil {
			finishReason = *chunk.Choices[0].FinishReason
			stopReason := mapFinishReason(*chunk.Choices[0].FinishReason)
			_ = encoder("message_delta", map[string]any{
				"type": "message_delta",
				"delta": map[string]any{
					"stop_reason":   stopReason,
					"stop_sequence": nil,
				},
				"usage": map[string]any{
					"input_tokens":            0,
					"output_tokens":           0,
					"cache_read_input_tokens": 0,
				},
			})
		}
	}

	closeCurrentBlock()

	if finishReason == "" {
		_ = encoder("message_delta", map[string]any{
			"type": "message_delta",
			"delta": map[string]any{
				"stop_reason":   "end_turn",
				"stop_sequence": nil,
			},
			"usage": map[string]any{
				"input_tokens":            0,
				"output_tokens":           0,
				"cache_read_input_tokens": 0,
			},
		})
	}

	_ = encoder("message_stop", map[string]any{
		"type": "message_stop",
	})
	if cfg.logStreamPreviewMax > 0 {
		log.Printf("[%s] stream summary chunks=%d text_chars=%d tool_delta_chunks=%d tool_args_chars=%d finish_reason=%q saw_done=%v preview=%q", reqID, chunkCount, textChars, toolDeltaChunks, toolArgsChars, finishReason, sawDone, preview.String())
	} else {
		log.Printf("[%s] stream summary chunks=%d text_chars=%d tool_delta_chunks=%d tool_args_chars=%d finish_reason=%q saw_done=%v", reqID, chunkCount, textChars, toolDeltaChunks, toolArgsChars, finishReason, sawDone)
	}
	return nil
}

func writeJSONError(w http.ResponseWriter, status int, code string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]any{
			"type":    "proxy_error",
			"code":    code,
			"message": code,
		},
	})
}

// ----------------------
// Tavily-backed Claude Code web tools
// ----------------------

type webToolSession struct {
	allowedFetchURLs map[string]struct{}
	maxUsesSearch    int
	maxUsesFetch     int
	usesSearch       int
	usesFetch        int

	searchAllowedDomains []string
	searchBlockedDomains []string
	fetchAllowedDomains  []string
	fetchBlockedDomains  []string
}

func requestWantsWebTools(tools []anthropicTool) bool {
	for _, t := range tools {
		name := strings.ToLower(strings.TrimSpace(t.Name))
		typ := strings.ToLower(strings.TrimSpace(t.Type))
		if name == "web_search" || name == "web_fetch" {
			return true
		}
		if strings.HasPrefix(typ, "web_search_") || strings.HasPrefix(typ, "web_fetch_") {
			return true
		}
	}
	return false
}

var urlRegex = regexp.MustCompile(`https?://[^\s<>"'\)\]]+`)

func extractAllowedFetchURLsFromAnthropicReq(req *anthropicMessageRequest) map[string]struct{} {
	allowed := map[string]struct{}{}

	sys := extractSystemText(req.System)
	for _, u := range urlRegex.FindAllString(sys, -1) {
		allowed[u] = struct{}{}
	}

	for _, m := range req.Messages {
		var asString string
		if err := json.Unmarshal(m.Content, &asString); err == nil {
			for _, u := range urlRegex.FindAllString(asString, -1) {
				allowed[u] = struct{}{}
			}
			continue
		}
		var blocks []anthropicContentBlock
		if err := json.Unmarshal(m.Content, &blocks); err != nil {
			continue
		}
		for _, blk := range blocks {
			if blk.Type == "text" && blk.Text != "" {
				for _, u := range urlRegex.FindAllString(blk.Text, -1) {
					allowed[u] = struct{}{}
				}
			}
			if blk.Type == "image" && blk.Source != nil && blk.Source.Type == "url" && blk.Source.URL != "" {
				allowed[blk.Source.URL] = struct{}{}
			}
			if blk.Type == "tool_result" && len(blk.Content) > 0 {
				for _, u := range urlRegex.FindAllString(string(blk.Content), -1) {
					allowed[u] = struct{}{}
				}
			}
		}
	}
	return allowed
}

func applyWebToolPolicies(sess *webToolSession, tools []anthropicTool) {
	sess.maxUsesSearch = 5
	sess.maxUsesFetch = 10

	for _, t := range tools {
		name := strings.ToLower(strings.TrimSpace(t.Name))
		typ := strings.ToLower(strings.TrimSpace(t.Type))
		isSearch := name == "web_search" || strings.HasPrefix(typ, "web_search_")
		isFetch := name == "web_fetch" || strings.HasPrefix(typ, "web_fetch_")
		if !isSearch && !isFetch {
			continue
		}

		if t.MaxUses > 0 {
			if isSearch {
				sess.maxUsesSearch = t.MaxUses
			}
			if isFetch {
				sess.maxUsesFetch = t.MaxUses
			}
		}
		if len(t.AllowedDomains) > 0 {
			if isSearch {
				sess.searchAllowedDomains = append([]string{}, t.AllowedDomains...)
			}
			if isFetch {
				sess.fetchAllowedDomains = append([]string{}, t.AllowedDomains...)
			}
		}
		if len(t.BlockedDomains) > 0 {
			if isSearch {
				sess.searchBlockedDomains = append([]string{}, t.BlockedDomains...)
			}
			if isFetch {
				sess.fetchBlockedDomains = append([]string{}, t.BlockedDomains...)
			}
		}
	}
}

func runWebToolsLoop(ctx context.Context, cfg *serverConfig, reqID string, openaiReq openaiChatCompletionRequest, upstreamURL, apiKey string, allowedFetchURLs map[string]struct{}, anthropicTools []anthropicTool) (anthropicMessageResponse, error) {
	sess := &webToolSession{
		allowedFetchURLs: allowedFetchURLs,
	}
	applyWebToolPolicies(sess, anthropicTools)

	prefixBlocks := make([]any, 0, 16)

	for step := 0; step < 12; step++ {
		raw, resp, err := doUpstreamJSON(ctx, cfg, openaiReq, upstreamURL, apiKey)
		if err != nil {
			return anthropicMessageResponse{}, err
		}
		_ = resp.Body.Close()

		var openaiResp openaiChatCompletionResponse
		if err := json.Unmarshal(raw, &openaiResp); err != nil {
			return anthropicMessageResponse{}, fmt.Errorf("invalid upstream json: %w", err)
		}
		if len(openaiResp.Choices) == 0 {
			return anthropicMessageResponse{}, errors.New("upstream returned no choices")
		}

		ch := openaiResp.Choices[0]
		if ch.Message.Content != nil && strings.TrimSpace(*ch.Message.Content) != "" {
			prefixBlocks = append(prefixBlocks, map[string]any{
				"type": "text",
				"text": *ch.Message.Content,
			})
		}

		// No tools => final answer
		if len(ch.Message.ToolCalls) == 0 {
			final := convertOpenAIToAnthropic(openaiResp)
			if len(prefixBlocks) > 0 {
				merged := make([]any, 0, len(prefixBlocks)+len(final.Content))
				merged = append(merged, prefixBlocks...)
				merged = append(merged, final.Content...)
				final.Content = merged
			}
			return final, nil
		}

		// If non-web tool calls appear, pass through to client (don’t half-handle).
		for _, tc := range ch.Message.ToolCalls {
			n := strings.ToLower(strings.TrimSpace(tc.Function.Name))
			if n != "web_search" && n != "web_fetch" {
				out := convertOpenAIToAnthropic(openaiResp)
				if len(prefixBlocks) > 0 {
					merged := make([]any, 0, len(prefixBlocks)+len(out.Content))
					merged = append(merged, prefixBlocks...)
					merged = append(merged, out.Content...)
					out.Content = merged
				}
				return out, nil
			}
		}

		// Execute web tools and continue loop with tool results fed back to upstream.
		for _, tc := range ch.Message.ToolCalls {
			name := strings.ToLower(strings.TrimSpace(tc.Function.Name))
			toolID := strings.TrimSpace(tc.ID)
			if toolID == "" {
				toolID = fmt.Sprintf("srvtoolu_%d", time.Now().UnixNano())
			}

			args := map[string]any{}
			switch v := tc.Function.Arguments.(type) {
			case string:
				_ = json.Unmarshal([]byte(v), &args)
			case map[string]any:
				args = v
			}

			switch name {
			case "web_search":
				if sess.usesSearch >= sess.maxUsesSearch {
					prefixBlocks = append(prefixBlocks,
						map[string]any{
							"type": "server_tool_use",
							"id":   toolID,
							"name": "web_search",
							"input": map[string]any{
								"query": fmt.Sprintf("%v", args["query"]),
							},
						},
						map[string]any{
							"type":        "web_search_tool_result",
							"tool_use_id": toolID,
							"content": map[string]any{
								"type":           "web_search_tool_result_error",
								"error_code":     "max_uses_exceeded",
								"error_message":  "web_search max_uses exceeded",
								"retryable":      false,
								"details":        "",
							},
						},
					)
					openaiReq.Messages = append(openaiReq.Messages,
						openaiAssistantToolCallMessage(toolID, "web_search", mustJSON(args)),
						map[string]any{"role": "tool", "tool_call_id": toolID, "content": `{"error":"max_uses_exceeded"}`},
					)
					continue
				}
				sess.usesSearch++

				q := strings.TrimSpace(fmt.Sprintf("%v", args["query"]))
				if q == "" {
					q = strings.TrimSpace(fmt.Sprintf("%v", args["q"]))
				}

				prefixBlocks = append(prefixBlocks, map[string]any{
					"type": "server_tool_use",
					"id":   toolID,
					"name": "web_search",
					"input": map[string]any{
						"query": q,
					},
				})

				modelToolContent, clientResultBlock, discoveredURLs, err := tavilyWebSearch(ctx, cfg, q, sess.searchAllowedDomains, sess.searchBlockedDomains)
				if err != nil {
					prefixBlocks = append(prefixBlocks, map[string]any{
						"type":        "web_search_tool_result",
						"tool_use_id": toolID,
						"content": map[string]any{
							"type":          "web_search_tool_result_error",
							"error_code":    "search_failed",
							"error_message": err.Error(),
						},
					})
					openaiReq.Messages = append(openaiReq.Messages,
						openaiAssistantToolCallMessage(toolID, "web_search", mustJSON(args)),
						map[string]any{"role": "tool", "tool_call_id": toolID, "content": fmt.Sprintf(`{"error":%q}`, err.Error())},
					)
					continue
				}

				for u := range discoveredURLs {
					sess.allowedFetchURLs[u] = struct{}{}
				}

				// attach tool_use_id
				if m, ok := clientResultBlock.(map[string]any); ok {
					m["tool_use_id"] = toolID
				}
				prefixBlocks = append(prefixBlocks, clientResultBlock)

				openaiReq.Messages = append(openaiReq.Messages,
					openaiAssistantToolCallMessage(toolID, "web_search", mustJSON(args)),
					map[string]any{"role": "tool", "tool_call_id": toolID, "content": modelToolContent},
				)

			case "web_fetch":
				if sess.usesFetch >= sess.maxUsesFetch {
					prefixBlocks = append(prefixBlocks,
						map[string]any{
							"type": "server_tool_use",
							"id":   toolID,
							"name": "web_fetch",
							"input": map[string]any{
								"url": fmt.Sprintf("%v", args["url"]),
							},
						},
						map[string]any{
							"type":        "web_fetch_tool_result",
							"tool_use_id": toolID,
							"content": map[string]any{
								"type":          "web_fetch_tool_result_error",
								"error_code":    "max_uses_exceeded",
								"error_message": "web_fetch max_uses exceeded",
							},
						},
					)
					openaiReq.Messages = append(openaiReq.Messages,
						openaiAssistantToolCallMessage(toolID, "web_fetch", mustJSON(args)),
						map[string]any{"role": "tool", "tool_call_id": toolID, "content": `{"error":"max_uses_exceeded"}`},
					)
					continue
				}
				sess.usesFetch++

				targetURL := strings.TrimSpace(fmt.Sprintf("%v", args["url"]))
				prefixBlocks = append(prefixBlocks, map[string]any{
					"type": "server_tool_use",
					"id":   toolID,
					"name": "web_fetch",
					"input": map[string]any{
						"url": targetURL,
					},
				})

				modelToolContent, clientResultBlock, err := tavilyWebFetch(ctx, cfg, targetURL, sess)
				if err != nil {
					prefixBlocks = append(prefixBlocks, map[string]any{
						"type":        "web_fetch_tool_result",
						"tool_use_id": toolID,
						"content": map[string]any{
							"type":          "web_fetch_tool_result_error",
							"error_code":    "fetch_failed",
							"error_message": err.Error(),
						},
					})
					openaiReq.Messages = append(openaiReq.Messages,
						openaiAssistantToolCallMessage(toolID, "web_fetch", mustJSON(args)),
						map[string]any{"role": "tool", "tool_call_id": toolID, "content": fmt.Sprintf(`{"error":%q}`, err.Error())},
					)
					continue
				}

				if m, ok := clientResultBlock.(map[string]any); ok {
					m["tool_use_id"] = toolID
				}
				prefixBlocks = append(prefixBlocks, clientResultBlock)

				openaiReq.Messages = append(openaiReq.Messages,
					openaiAssistantToolCallMessage(toolID, "web_fetch", mustJSON(args)),
					map[string]any{"role": "tool", "tool_call_id": toolID, "content": modelToolContent},
				)
			}
		}
	}

	return anthropicMessageResponse{}, errors.New("web-tools loop exceeded max steps")
}

func runWebToolsLoopAndStream(w http.ResponseWriter, r *http.Request, cfg *serverConfig, reqID string, openaiReq openaiChatCompletionRequest, upstreamURL, apiKey string, allowedFetchURLs map[string]struct{}) error {
	// Not truly incremental streaming. We do the tool loop, then emit one Anthropic SSE message.
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)

	flusher, ok := w.(http.Flusher)
	if !ok {
		writeJSONError(w, http.StatusInternalServerError, "streaming_not_supported")
		return errors.New("http.Flusher not supported")
	}

	encoder := func(event string, payload any) error {
		b, err := json.Marshal(payload)
		if err != nil {
			return err
		}
		if _, err := fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, string(b)); err != nil {
			return err
		}
		flusher.Flush()
		return nil
	}

	// We do not have the original anthropic tools here; just run with defaults.
	respMsg, err := runWebToolsLoop(r.Context(), cfg, reqID, openaiReq, upstreamURL, apiKey, allowedFetchURLs, nil)
	if err != nil {
		_ = encoder("error", map[string]any{"type": "error", "message": err.Error()})
		return err
	}

	messageID := respMsg.ID
	if strings.TrimSpace(messageID) == "" {
		messageID = fmt.Sprintf("msg_%d", time.Now().UnixMilli())
	}

	_ = encoder("message_start", map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id":            messageID,
			"type":          "message",
			"role":          "assistant",
			"model":         respMsg.Model,
			"content":       []any{},
			"stop_reason":   nil,
			"stop_sequence": nil,
			"usage": map[string]any{"input_tokens": 0, "output_tokens": 0},
		},
	})

	idx := 0
	for _, blk := range respMsg.Content {
		// safest: stream as JSON text chunks (Claude Code UI still sees it)
		_ = encoder("content_block_start", map[string]any{
			"type":  "content_block_start",
			"index": idx,
			"content_block": map[string]any{
				"type": "text",
				"text": "",
			},
		})
		j, _ := json.Marshal(blk)
		_ = encoder("content_block_delta", map[string]any{
			"type":  "content_block_delta",
			"index": idx,
			"delta": map[string]any{
				"type": "text_delta",
				"text": string(j) + "\n",
			},
		})
		_ = encoder("content_block_stop", map[string]any{"type": "content_block_stop", "index": idx})
		idx++
	}

	_ = encoder("message_delta", map[string]any{
		"type": "message_delta",
		"delta": map[string]any{
			"stop_reason":   respMsg.StopReason,
			"stop_sequence": nil,
		},
		"usage": map[string]any{"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0},
	})
	_ = encoder("message_stop", map[string]any{"type": "message_stop"})
	return nil
}

func openaiAssistantToolCallMessage(toolID, name, argsJSON string) map[string]any {
	return map[string]any{
		"role":    "assistant",
		"content": nil,
		"tool_calls": []any{
			map[string]any{
				"id":   toolID,
				"type": "function",
				"function": map[string]any{
					"name":      name,
					"arguments": argsJSON,
				},
			},
		},
	}
}

func tavilyHTTPClient(cfg *serverConfig) (*http.Client, error) {
	tr := &http.Transport{
		Proxy: http.ProxyFromEnvironment,
	}
	if strings.TrimSpace(cfg.tavilyProxyURL) != "" {
		pu, err := url.Parse(cfg.tavilyProxyURL)
		if err != nil {
			return nil, fmt.Errorf("invalid tavily proxy url: %w", err)
		}
		tr.Proxy = http.ProxyURL(pu)
	}
	return &http.Client{
		Timeout:   cfg.timeout,
		Transport: tr,
	}, nil
}

type tavilySearchRequest struct {
	Query            string   `json:"query"`
	SearchDepth      string   `json:"search_depth,omitempty"`
	Topic            string   `json:"topic,omitempty"`
	MaxResults       int      `json:"max_results"`
	IncludeAnswer    bool     `json:"include_answer"`
	IncludeRawContent bool    `json:"include_raw_content"`
	IncludeDomains   []string `json:"include_domains,omitempty"`
	ExcludeDomains   []string `json:"exclude_domains,omitempty"`
}

type tavilySearchResponse struct {
	Query   string `json:"query"`
	Results []struct {
		Title   string  `json:"title"`
		URL     string  `json:"url"`
		Content string  `json:"content"`
		Score   float64 `json:"score"`
	} `json:"results"`
	RequestID string `json:"request_id"`
}

func tavilyWebSearch(ctx context.Context, cfg *serverConfig, query string, allowedDomains, blockedDomains []string) (modelToolContent string, clientResultBlock any, discoveredURLs map[string]struct{}, err error) {
	client, err := tavilyHTTPClient(cfg)
	if err != nil {
		return "", nil, nil, err
	}

	reqBody := tavilySearchRequest{
		Query:             query,
		SearchDepth:       cfg.tavilySearchDepth,
		Topic:             cfg.tavilyTopic,
		MaxResults:        cfg.tavilyMaxResults,
		IncludeAnswer:     false,
		IncludeRawContent: false,
	}
	if len(allowedDomains) > 0 {
		reqBody.IncludeDomains = allowedDomains
	}
	if len(blockedDomains) > 0 {
		reqBody.ExcludeDomains = blockedDomains
	}

	b, _ := json.Marshal(reqBody)
	endpoint := strings.TrimRight(cfg.tavilyBaseURL, "/") + "/search"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(b))
	if err != nil {
		return "", nil, nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+cfg.tavilyAPIKey)

	resp, err := client.Do(req)
	if err != nil {
		return "", nil, nil, err
	}
	defer resp.Body.Close()

	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", nil, nil, fmt.Errorf("tavily search status %d: %s", resp.StatusCode, takeFirstRunes(string(raw), 800))
	}

	var tr tavilySearchResponse
	if err := json.Unmarshal(raw, &tr); err != nil {
		return "", nil, nil, fmt.Errorf("tavily search decode failed: %w", err)
	}

	discoveredURLs = map[string]struct{}{}
	contentItems := make([]any, 0, len(tr.Results))

	var sb strings.Builder
	sb.WriteString(`{"type":"web_search","query":`)
	sb.WriteString(strconv.Quote(query))
	sb.WriteString(`,"results":[`)

	for i, it := range tr.Results {
		u := strings.TrimSpace(it.URL)
		if u != "" {
			discoveredURLs[u] = struct{}{}
		}

		enc := base64.StdEncoding.EncodeToString([]byte(strings.TrimSpace(it.Content)))
		contentItems = append(contentItems, map[string]any{
			"type":             "web_search_result",
			"url":              u,
			"title":            strings.TrimSpace(it.Title),
			"encrypted_content": enc,
		})

		if i > 0 {
			sb.WriteString(",")
		}
		sb.WriteString(`{"title":`)
		sb.WriteString(strconv.Quote(strings.TrimSpace(it.Title)))
		sb.WriteString(`,"url":`)
		sb.WriteString(strconv.Quote(u))
		sb.WriteString(`,"snippet":`)
		sb.WriteString(strconv.Quote(takeFirstRunes(strings.TrimSpace(it.Content), 1200)))
		sb.WriteString("}")
	}
	sb.WriteString("]}")

	clientResultBlock = map[string]any{
		"type":    "web_search_tool_result",
		"content": contentItems,
	}
	return sb.String(), clientResultBlock, discoveredURLs, nil
}

type tavilyExtractRequest struct {
	URLs   any    `json:"urls"` // string or []string
	Format string `json:"format,omitempty"`
}

type tavilyExtractResponse struct {
	Results []struct {
		URL        string `json:"url"`
		RawContent string `json:"raw_content"`
	} `json:"results"`
	FailedResults []struct {
		URL   string `json:"url"`
		Error string `json:"error"`
	} `json:"failed_results"`
	RequestID string `json:"request_id"`
}

func tavilyWebFetch(ctx context.Context, cfg *serverConfig, targetURL string, sess *webToolSession) (modelToolContent string, clientResultBlock any, err error) {
	targetURL = strings.TrimSpace(targetURL)
	if targetURL == "" {
		return "", nil, errors.New("missing url")
	}

	// Require URL to appear in conversation/search results first.
	if _, ok := sess.allowedFetchURLs[targetURL]; !ok {
		return "", nil, fmt.Errorf("web_fetch url not allowed (must appear in conversation/search results): %s", targetURL)
	}

	if err := validateFetchURL(targetURL); err != nil {
		return "", nil, err
	}
	if err := enforceDomainFilters(targetURL, sess.fetchAllowedDomains, sess.fetchBlockedDomains); err != nil {
		return "", nil, err
	}

	client, err := tavilyHTTPClient(cfg)
	if err != nil {
		return "", nil, err
	}

	reqBody := tavilyExtractRequest{
		URLs:   []string{targetURL},
		Format: "text",
	}
	b, _ := json.Marshal(reqBody)
	endpoint := strings.TrimRight(cfg.tavilyBaseURL, "/") + "/extract"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(b))
	if err != nil {
		return "", nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+cfg.tavilyAPIKey)

	resp, err := client.Do(req)
	if err != nil {
		return "", nil, err
	}
	defer resp.Body.Close()

	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", nil, fmt.Errorf("tavily extract status %d: %s", resp.StatusCode, takeFirstRunes(string(raw), 800))
	}

	var tr tavilyExtractResponse
	if err := json.Unmarshal(raw, &tr); err != nil {
		return "", nil, fmt.Errorf("tavily extract decode failed: %w", err)
	}
	if len(tr.Results) == 0 {
		if len(tr.FailedResults) > 0 {
			return "", nil, fmt.Errorf("tavily extract failed: %s", tr.FailedResults[0].Error)
		}
		return "", nil, errors.New("tavily extract returned no results")
	}

	text := tr.Results[0].RawContent
	if text == "" {
		text = "(empty content)"
	}
	text = takeFirstRunes(text, cfg.tavilyFetchMaxRunes)

	modelToolContent = fmt.Sprintf(`{"type":"web_fetch","url":%q,"content":%q}`, targetURL, text)

	clientResultBlock = map[string]any{
		"type": "web_fetch_tool_result",
		"content": map[string]any{
			"type": "web_fetch_result",
			"url":  targetURL,
			"content": map[string]any{
				"type": "document",
				"source": map[string]any{
					"type":       "text",
					"media_type": "text/plain",
					"data":       text,
				},
				"title":     "",
				"citations": map[string]any{"enabled": false},
			},
			"retrieved_at": time.Now().UTC().Format(time.RFC3339),
		},
	}

	return modelToolContent, clientResultBlock, nil
}

func validateFetchURL(raw string) error {
	u, err := url.Parse(raw)
	if err != nil {
		return fmt.Errorf("invalid url: %w", err)
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return fmt.Errorf("unsupported url scheme: %q", u.Scheme)
	}
	host := u.Hostname()
	if host == "" {
		return errors.New("missing url host")
	}
	lh := strings.ToLower(host)
	if lh == "localhost" || strings.HasSuffix(lh, ".local") {
		return fmt.Errorf("blocked host: %s", host)
	}
	if ip := net.ParseIP(host); ip != nil {
		if ip.IsLoopback() || ip.IsPrivate() || ip.IsLinkLocalUnicast() || ip.IsLinkLocalMulticast() {
			return fmt.Errorf("blocked ip host: %s", host)
		}
	}
	return nil
}

func enforceDomainFilters(raw string, allowed, blocked []string) error {
	u, err := url.Parse(raw)
	if err != nil {
		return fmt.Errorf("invalid url: %w", err)
	}
	host := strings.ToLower(u.Hostname())

	if len(blocked) > 0 {
		for _, d := range blocked {
			d = strings.ToLower(strings.TrimSpace(d))
			if d == "" {
				continue
			}
			if host == d || strings.HasSuffix(host, "."+d) {
				return fmt.Errorf("blocked domain by policy: %s", host)
			}
		}
	}
	if len(allowed) > 0 {
		ok := false
		for _, d := range allowed {
			d = strings.ToLower(strings.TrimSpace(d))
			if d == "" {
				continue
			}
			if host == d || strings.HasSuffix(host, "."+d) {
				ok = true
				break
			}
		}
		if !ok {
			return fmt.Errorf("domain not in allowlist: %s", host)
		}
	}
	return nil
}

// ----------------------
// Anthropic request types
// ----------------------

type anthropicMessageRequest struct {
	Model       string          `json:"model"`
	MaxTokens   int             `json:"max_tokens"`
	Temperature *float64        `json:"temperature,omitempty"`
	Stream      bool            `json:"stream,omitempty"`
	System      json.RawMessage `json:"system,omitempty"`
	Messages    []anthropicMsg  `json:"messages"`
	Tools       []anthropicTool `json:"tools,omitempty"`
	ToolChoice  any             `json:"tool_choice,omitempty"`
	Thinking    any             `json:"thinking,omitempty"`
}

type anthropicMsg struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

type anthropicTool struct {
	// Claude Code server tools often include "type":"web_search_YYYYMMDD" etc.
	Type string `json:"type,omitempty"`

	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"input_schema,omitempty"`

	// Optional policy knobs (Claude Code)
	MaxUses        int      `json:"max_uses,omitempty"`
	AllowedDomains []string `json:"allowed_domains,omitempty"`
	BlockedDomains []string `json:"blocked_domains,omitempty"`
}

type anthropicContentBlock struct {
	Type string `json:"type"`

	// text
	Text string `json:"text,omitempty"`

	// image
	Source *anthropicImageSource `json:"source,omitempty"`

	// tool_use
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`

	// tool_result
	ToolUseID string          `json:"tool_use_id,omitempty"`
	Content   json.RawMessage `json:"content,omitempty"`
}

type anthropicImageSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type"`
	Data      string `json:"data,omitempty"`
	URL       string `json:"url,omitempty"`
}

// ----------------------
// OpenAI request types
// ----------------------

type openaiChatCompletionRequest struct {
	Model       string `json:"model"`
	Messages    []any  `json:"messages"`
	MaxTokens   int    `json:"max_tokens,omitempty"`
	Temperature any    `json:"temperature,omitempty"`
	Stream      bool   `json:"stream,omitempty"`
	Tools       []any  `json:"tools,omitempty"`
	ToolChoice  any    `json:"tool_choice,omitempty"`
}

const defaultWebSearchSchemaJSON = `{
  "type": "object",
  "properties": {
    "query": { "type": "string", "description": "Search query string." }
  },
  "required": ["query"]
}`

const defaultWebFetchSchemaJSON = `{
  "type": "object",
  "properties": {
    "url": { "type": "string", "description": "URL to fetch (must already appear in the conversation or search results)." }
  },
  "required": ["url"]
}`

func convertAnthropicToOpenAI(req *anthropicMessageRequest) (openaiChatCompletionRequest, error) {
	var messages []any

	if sys := strings.TrimSpace(extractSystemText(req.System)); sys != "" {
		messages = append(messages, map[string]any{
			"role":    "system",
			"content": sys,
		})
	}

	for _, m := range req.Messages {
		role := strings.TrimSpace(m.Role)
		if role == "" {
			continue
		}

		var asString string
		if err := json.Unmarshal(m.Content, &asString); err == nil {
			messages = append(messages, map[string]any{
				"role":    role,
				"content": asString,
			})
			continue
		}

		var blocks []anthropicContentBlock
		if err := json.Unmarshal(m.Content, &blocks); err != nil {
			return openaiChatCompletionRequest{}, fmt.Errorf("invalid message content for role %q", role)
		}

		switch role {
		case "user":
			userMsgs, err := convertAnthropicUserBlocksToOpenAIMessages(blocks)
			if err != nil {
				return openaiChatCompletionRequest{}, err
			}
			messages = append(messages, userMsgs...)
		case "assistant":
			assistantMsg, err := convertAnthropicAssistantBlocksToOpenAIMessage(blocks)
			if err != nil {
				return openaiChatCompletionRequest{}, err
			}
			messages = append(messages, assistantMsg)
		default:
			text := joinTextBlocks(blocks)
			messages = append(messages, map[string]any{
				"role":    role,
				"content": text,
			})
		}
	}

	out := openaiChatCompletionRequest{
		Model:       req.Model,
		Messages:    messages,
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
		Stream:      req.Stream,
	}

	if len(req.Tools) > 0 {
		out.Tools = make([]any, 0, len(req.Tools))
		for _, t := range req.Tools {
			if isAnthropicWebTool(t) {
				var schema any
				if isWebFetchTool(t) {
					_ = json.Unmarshal([]byte(defaultWebFetchSchemaJSON), &schema)
					out.Tools = append(out.Tools, map[string]any{
						"type": "function",
						"function": map[string]any{
							"name":        "web_fetch",
							"description": "Fetch content from a URL (proxy-backed).",
							"parameters":  schema,
						},
					})
				} else {
					_ = json.Unmarshal([]byte(defaultWebSearchSchemaJSON), &schema)
					out.Tools = append(out.Tools, map[string]any{
						"type": "function",
						"function": map[string]any{
							"name":        "web_search",
							"description": "Search the web (proxy-backed).",
							"parameters":  schema,
						},
					})
				}
				continue
			}

			var params any
			if len(t.InputSchema) > 0 {
				_ = json.Unmarshal(t.InputSchema, &params)
			}
			out.Tools = append(out.Tools, map[string]any{
				"type": "function",
				"function": map[string]any{
					"name":        t.Name,
					"description": t.Description,
					"parameters":  params,
				},
			})
		}
	}

	if req.ToolChoice != nil {
		out.ToolChoice = convertToolChoice(req.ToolChoice)
	}

	return out, nil
}

func isAnthropicWebTool(t anthropicTool) bool {
	name := strings.ToLower(strings.TrimSpace(t.Name))
	typ := strings.ToLower(strings.TrimSpace(t.Type))
	if name == "web_search" || name == "web_fetch" {
		return true
	}
	if strings.HasPrefix(typ, "web_search_") || strings.HasPrefix(typ, "web_fetch_") {
		return true
	}
	return false
}

func isWebFetchTool(t anthropicTool) bool {
	name := strings.ToLower(strings.TrimSpace(t.Name))
	typ := strings.ToLower(strings.TrimSpace(t.Type))
	return name == "web_fetch" || strings.HasPrefix(typ, "web_fetch_")
}

func extractSystemText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}
	var blocks []anthropicContentBlock
	if err := json.Unmarshal(raw, &blocks); err == nil {
		return joinTextBlocks(blocks)
	}
	return ""
}

func joinTextBlocks(blocks []anthropicContentBlock) string {
	var b strings.Builder
	for _, blk := range blocks {
		if blk.Type == "text" && blk.Text != "" {
			if b.Len() > 0 {
				b.WriteString("\n")
			}
			b.WriteString(blk.Text)
		}
	}
	return b.String()
}

func convertAnthropicUserBlocksToOpenAIMessages(blocks []anthropicContentBlock) ([]any, error) {
	var out []any

	for _, blk := range blocks {
		if blk.Type != "tool_result" || strings.TrimSpace(blk.ToolUseID) == "" {
			continue
		}
		contentStr := ""
		if len(blk.Content) > 0 {
			var s string
			if err := json.Unmarshal(blk.Content, &s); err == nil {
				contentStr = s
			} else {
				contentStr = string(blk.Content)
			}
		}
		out = append(out, map[string]any{
			"role":         "tool",
			"tool_call_id": blk.ToolUseID,
			"content":      contentStr,
		})
	}

	var parts []any
	for _, blk := range blocks {
		switch blk.Type {
		case "text":
			if blk.Text != "" {
				parts = append(parts, map[string]any{"type": "text", "text": blk.Text})
			}
		case "image":
			if blk.Source == nil {
				continue
			}
			u := ""
			switch blk.Source.Type {
			case "base64":
				if blk.Source.MediaType == "" || blk.Source.Data == "" {
					continue
				}
				if _, err := base64.StdEncoding.DecodeString(blk.Source.Data); err != nil {
					continue
				}
				u = "data:" + blk.Source.MediaType + ";base64," + blk.Source.Data
			case "url":
				u = blk.Source.URL
			default:
				continue
			}
			if u != "" {
				parts = append(parts, map[string]any{
					"type": "image_url",
					"image_url": map[string]any{
						"url": u,
					},
				})
			}
		}
	}

	if len(parts) == 0 {
		out = append(out, map[string]any{"role": "user", "content": ""})
		return out, nil
	}
	if len(parts) == 1 {
		if p, ok := parts[0].(map[string]any); ok && p["type"] == "text" {
			if t, ok := p["text"].(string); ok {
				out = append(out, map[string]any{"role": "user", "content": t})
				return out, nil
			}
		}
	}

	out = append(out, map[string]any{
		"role":    "user",
		"content": parts,
	})
	return out, nil
}

func convertAnthropicAssistantBlocksToOpenAIMessage(blocks []anthropicContentBlock) (any, error) {
	text := joinTextBlocks(blocks)

	var toolCalls []any
	for _, blk := range blocks {
		if blk.Type != "tool_use" || strings.TrimSpace(blk.ID) == "" || strings.TrimSpace(blk.Name) == "" {
			continue
		}
		args := "{}"
		if len(blk.Input) > 0 {
			args = string(blk.Input)
		}
		toolCalls = append(toolCalls, map[string]any{
			"id":   blk.ID,
			"type": "function",
			"function": map[string]any{
				"name":      blk.Name,
				"arguments": args,
			},
		})
	}

	msg := map[string]any{"role": "assistant"}
	if text != "" {
		msg["content"] = text
	} else {
		msg["content"] = nil
	}
	if len(toolCalls) > 0 {
		msg["tool_calls"] = toolCalls
	}
	return msg, nil
}

func convertToolChoice(v any) any {
	m, ok := v.(map[string]any)
	if !ok {
		return v
	}
	typ, _ := m["type"].(string)
	switch typ {
	case "auto", "none", "required":
		return typ
	case "tool":
		name, _ := m["name"].(string)
		if name == "" {
			return "auto"
		}
		return map[string]any{
			"type": "function",
			"function": map[string]any{
				"name": name,
			},
		}
	default:
		return v
	}
}

// ----------------------
// OpenAI response types
// ----------------------

type openaiChatCompletionResponse struct {
	ID      string `json:"id"`
	Model   string `json:"model"`
	Choices []struct {
		Message struct {
			Role    string  `json:"role"`
			Content *string `json:"content"`
			ToolCalls []struct {
				ID       string `json:"id"`
				Type     string `json:"type"`
				Function struct {
					Name      string `json:"name"`
					Arguments any    `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls,omitempty"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		PromptTokensDetails *struct {
			CachedTokens int `json:"cached_tokens"`
		} `json:"prompt_tokens_details,omitempty"`
	} `json:"usage,omitempty"`
}

type anthropicMessageResponse struct {
	ID           string `json:"id"`
	Type         string `json:"type"`
	Role         string `json:"role"`
	Model        string `json:"model"`
	Content      []any  `json:"content"`
	StopReason   string `json:"stop_reason"`
	StopSequence any    `json:"stop_sequence"`
	Usage        any    `json:"usage"`
}

func convertOpenAIToAnthropic(resp openaiChatCompletionResponse) anthropicMessageResponse {
	content := make([]any, 0, 4)

	var finishReason string
	if len(resp.Choices) > 0 {
		ch := resp.Choices[0]
		finishReason = ch.FinishReason
		if ch.Message.Content != nil && *ch.Message.Content != "" {
			content = append(content, map[string]any{
				"type": "text",
				"text": *ch.Message.Content,
			})
		}
		if len(ch.Message.ToolCalls) > 0 {
			for _, tc := range ch.Message.ToolCalls {
				input := map[string]any{}
				switch v := tc.Function.Arguments.(type) {
				case string:
					_ = json.Unmarshal([]byte(v), &input)
				case map[string]any:
					input = v
				default:
					input = map[string]any{"text": fmt.Sprintf("%v", v)}
				}
				content = append(content, map[string]any{
					"type":  "tool_use",
					"id":    tc.ID,
					"name":  tc.Function.Name,
					"input": input,
				})
			}
		}
	}

	inputTokens := 0
	outputTokens := 0
	cacheRead := 0
	if resp.Usage != nil {
		if resp.Usage.PromptTokensDetails != nil {
			cacheRead = resp.Usage.PromptTokensDetails.CachedTokens
		}
		inputTokens = resp.Usage.PromptTokens - cacheRead
		outputTokens = resp.Usage.CompletionTokens
	}

	return anthropicMessageResponse{
		ID:           resp.ID,
		Type:         "message",
		Role:         "assistant",
		Model:        resp.Model,
		Content:      content,
		StopReason:   mapFinishReason(finishReason),
		StopSequence: nil,
		Usage: map[string]any{
			"input_tokens":            inputTokens,
			"output_tokens":           outputTokens,
			"cache_read_input_tokens": cacheRead,
		},
	}
}

func mapFinishReason(finish string) string {
	switch finish {
	case "stop":
		return "end_turn"
	case "length":
		return "max_tokens"
	case "tool_calls":
		return "tool_use"
	case "content_filter":
		return "stop_sequence"
	default:
		return "end_turn"
	}
}

// ----------------------
// Streaming chunk types
// ----------------------

type openaiChatCompletionChunk struct {
	Model   string `json:"model,omitempty"`
	Choices []struct {
		Delta struct {
			Content *string `json:"content,omitempty"`
			ToolCalls []struct {
				Index int    `json:"index,omitempty"`
				ID    string `json:"id,omitempty"`
				Type  string `json:"type,omitempty"`
				Function struct {
					Name      string `json:"name,omitempty"`
					Arguments string `json:"arguments,omitempty"`
				} `json:"function,omitempty"`
			} `json:"tool_calls,omitempty"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason,omitempty"`
	} `json:"choices"`
}

func logForwardedRequest(reqID string, cfg *serverConfig, anthropicReq anthropicMessageRequest, openaiReq openaiChatCompletionRequest, upstreamURL string) {
	inSummary := map[string]any{
		"model":      anthropicReq.Model,
		"max_tokens": anthropicReq.MaxTokens,
		"stream":     anthropicReq.Stream,
		"messages":   len(anthropicReq.Messages),
		"tools":      len(anthropicReq.Tools),
	}
	log.Printf("[%s] inbound summary=%s", reqID, mustJSONTrunc(inSummary, cfg.logBodyMax))

	out := sanitizeOpenAIRequest(openaiReq)
	log.Printf("[%s] forward url=%s", reqID, upstreamURL)
	log.Printf("[%s] forward headers=%s", reqID, mustJSONTrunc(map[string]any{
		"Content-Type":  "application/json",
		"Authorization": "Bearer <redacted>",
	}, cfg.logBodyMax))
	log.Printf("[%s] forward body=%s", reqID, mustJSONTrunc(out, cfg.logBodyMax))
}

func logForwardedUpstreamBody(reqID string, cfg *serverConfig, body []byte) {
	if cfg.logBodyMax == 0 {
		return
	}
	s := string(body)
	if len([]rune(s)) > cfg.logBodyMax {
		s = string([]rune(s)[:cfg.logBodyMax]) + "...(truncated)"
	}
	log.Printf("[%s] upstream body=%s", reqID, s)
}

func mustJSON(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		return `{"_error":"json_marshal_failed"}`
	}
	return string(b)
}

func mustJSONTrunc(v any, maxChars int) string {
	b, err := json.Marshal(v)
	if err != nil {
		return fmt.Sprintf(`{"_error":"json_marshal_failed","detail":%q}`, err.Error())
	}
	s := string(b)
	if maxChars <= 0 {
		return s
	}
	if len([]rune(s)) > maxChars {
		return string([]rune(s)[:maxChars]) + "...(truncated)"
	}
	return s
}

func sanitizeOpenAIRequest(req openaiChatCompletionRequest) openaiChatCompletionRequest {
	out := req
	out.Messages = sanitizeOpenAIMessages(req.Messages)
	out.Tools = sanitizeAnySlice(req.Tools)
	return out
}

func sanitizeOpenAIMessages(msgs []any) []any {
	if len(msgs) == 0 {
		return nil
	}
	out := make([]any, 0, len(msgs))
	for _, m := range msgs {
		mm, ok := m.(map[string]any)
		if !ok {
			out = append(out, m)
			continue
		}
		cp := map[string]any{}
		for k, v := range mm {
			cp[k] = v
		}
		if content, ok := cp["content"]; ok {
			cp["content"] = sanitizeMessageContent(content)
		}
		if tc, ok := cp["tool_calls"]; ok {
			cp["tool_calls"] = sanitizeAny(tc)
		}
		out = append(out, cp)
	}
	return out
}

func sanitizeMessageContent(content any) any {
	switch v := content.(type) {
	case string:
		return v
	case []any:
		parts := make([]any, 0, len(v))
		for _, p := range v {
			pm, ok := p.(map[string]any)
			if !ok {
				parts = append(parts, p)
				continue
			}
			cp := map[string]any{}
			for k, vv := range pm {
				cp[k] = vv
			}
			if cp["type"] == "image_url" {
				if iu, ok := cp["image_url"].(map[string]any); ok {
					if u, ok := iu["url"].(string); ok && strings.HasPrefix(u, "data:") {
						iu2 := map[string]any{}
						for k, vv := range iu {
							iu2[k] = vv
						}
						iu2["url"] = "data:<redacted>"
						cp["image_url"] = iu2
					}
				}
			}
			parts = append(parts, cp)
		}
		return parts
	default:
		return sanitizeAny(v)
	}
}

func sanitizeAnySlice(v []any) []any {
	if len(v) == 0 {
		return nil
	}
	out := make([]any, 0, len(v))
	for _, it := range v {
		out = append(out, sanitizeAny(it))
	}
	return out
}

func sanitizeAny(v any) any {
	switch t := v.(type) {
	case map[string]any:
		cp := map[string]any{}
		for k, vv := range t {
			cp[k] = sanitizeAny(vv)
		}
		return cp
	case []any:
		return sanitizeAnySlice(t)
	case string:
		return t
	default:
		return v
	}
}

func takeFirstRunes(s string, max int) string {
	if max <= 0 || s == "" {
		return ""
	}
	r := []rune(s)
	if len(r) <= max {
		return s
	}
	return string(r[:max])
}
