# flock-routing

An OpenAI-compatible proxy that routes chat completion requests between a **local model** (fast, private, weaker) and a **remote model** (slower, paid, stronger) based on prompt complexity. Simple prompts stay local; hard ones are escalated.

## How it works

Every request is scored by a multi-factor complexity algorithm before it reaches a model:

| Factor | Weight | Signal |
|--------|--------|--------|
| Token length | 25% | Longer prompts carry more information to process |
| Reasoning markers | 30% | Keywords like *analyze*, *prove*, *step by step* |
| Code / math content | 25% | Fenced code blocks, formal notation (∫, O(n), …) |
| Question multiplicity | 10% | Number of `?` marks in the conversation |
| Conversation depth | 10% | Number of turns in the multi-turn context |

The score is a float in `[0.0, 1.0]`. Requests at or above `complexity_threshold` go to the remote model; everything below stays local.

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
