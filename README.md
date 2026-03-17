# flock-routing

An OpenAI-compatible proxy that routes chat completion requests between a **local model** (fast, private, weaker) and a **remote model** (slower, paid, stronger) based on prompt complexity. Simple prompts stay local; hard ones are escalated.

## How it works

Every request is scored by a principled complexity algorithm before it reaches a model:

| Component | Weight | Signal |
|-----------|--------|--------|
| Perplexity (distilgpt2) | 55% | Language model assigns higher loss to specialized, technical, or domain-specific text that a weak local model is unlikely to handle well |
| Linguistic features | 45% | Structural properties: Flesch-Kincaid readability, vocabulary richness (TTR, hapax ratio), content-word density, sentence length, conditional clauses, discourse connectives |

The score is a float in `[0.0, 1.0]`. Requests at or above `complexity_threshold` go to the remote model; everything below stays local.

**Calibration guide:**

| Score range | Typical content |
|-------------|-----------------|
| 0.05–0.10 | Simple greetings and short conversational text |
| 0.20–0.35 | Short factual questions |
| 0.55–0.75 | Multi-step technical or analytical requests |
| > 0.80 | Dense domain-specific or highly specialized text |

> **Note on latency:** The perplexity component runs distilgpt2 locally (~30–60ms on Apple Silicon MPS, ~200–400ms on CPU). The model is loaded at server startup to avoid first-request latency.

## Quickstart

```bash
git clone git@github.com:limberc/flock-routing.git
cd flock-routing

python -m venv .venv && source .venv/bin/activate
pip install -e .

cp config.example.yaml config.yaml
# edit config.yaml — set your remote API key and model names
export OPENAI_API_KEY=sk-...

privacy-serving          # starts on http://0.0.0.0:8080
```

Point any OpenAI-compatible client at `http://localhost:8080` and it will transparently route through the proxy.

## Configuration

```yaml
local_model:
  base_url: "http://localhost:11434/v1"   # Ollama or any OpenAI-compatible server
  model: "qwen2.5:9b"
  api_key: "none"                          # omit Authorization header

remote_model:
  base_url: "https://api.openai.com/v1"
  model: "gpt-4o"
  api_key: "${OPENAI_API_KEY}"             # expanded from environment

routing:
  complexity_threshold: 0.5   # 0.0 = always local, 1.0 = always remote

server:
  host: "0.0.0.0"
  port: 8080
```

The config file path defaults to `config.yaml` in the working directory. Override with:

```bash
PRIVACY_SERVING_CONFIG=/path/to/config.yaml privacy-serving
```

### Threshold tuning

| Threshold | Behaviour |
|-----------|-----------|
| `0.3` | Aggressive local usage — most requests stay on-device |
| `0.5` | Balanced (default) |
| `0.7` | Conservative — only clearly simple requests go local |

## API

The proxy is a drop-in replacement for any OpenAI-compatible endpoint.

### `POST /v1/chat/completions`

Standard OpenAI request body. Response adds three headers:

| Header | Example | Meaning |
|--------|---------|---------|
| `X-Routed-To` | `local` | Which tier handled the request |
| `X-Complexity-Score` | `0.2314` | Raw complexity score (0–1) |
| `X-Model-Used` | `qwen2.5:9b` | Exact model name forwarded downstream |

> Streaming (`"stream": true`) is not supported and returns `501`.

### `GET /v1/models`

Returns both configured models in OpenAI list format.

### `GET /stats`

Returns routing counters since startup:

```json
{
  "local": 142,
  "remote": 38,
  "total": 180,
  "local_rate": 0.7889
}
```

`local_rate` is the fraction of requests handled locally — the **replacement rate**.

## Development

```bash
pip install -e ".[dev]"
pytest                   # 38 tests
```

## Requirements

- Python 3.10+
- A local model endpoint (e.g. [Ollama](https://ollama.com))
- A remote OpenAI-compatible API key
