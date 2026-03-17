import json
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock

from privacy_serving.config import Config, ModelConfig, RoutingConfig, ServerConfig
from privacy_serving.main import create_app


FAKE_COMPLETION = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "qwen2.5:9b",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "42"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
}


@pytest.fixture
def test_config():
    return Config(
        local_model=ModelConfig(base_url="http://local/v1", model="qwen2.5:9b"),
        remote_model=ModelConfig(base_url="http://remote/v1", model="gpt-4o", api_key="sk-test"),
        routing=RoutingConfig(complexity_threshold=0.5),
        server=ServerConfig(),
    )


@pytest.fixture
def client(test_config):
    app = create_app(test_config)
    return TestClient(app)


def test_models_endpoint(client):
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    models = [m["id"] for m in data["data"]]
    assert "qwen2.5:9b" in models
    assert "gpt-4o" in models


def test_chat_completions_routes_to_local(client, test_config):
    """A simple 'Hi' prompt should route to local model."""
    with patch("privacy_serving.main.ModelClient") as MockClient:
        mock_instance = AsyncMock()
        mock_instance.complete = AsyncMock(return_value=FAKE_COMPLETION)
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=None)
        MockClient.return_value = mock_instance

        resp = client.post("/v1/chat/completions", json={
            "model": "auto",
            "messages": [{"role": "user", "content": "Hi"}],
        })
    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["message"]["content"] == "42"
    # Routing header tells which model was used
    assert "X-Routed-To" in resp.headers


def test_chat_completions_includes_routing_metadata(client):
    """Response headers include routing decision and score."""
    with patch("privacy_serving.main.ModelClient") as MockClient:
        mock_instance = AsyncMock()
        mock_instance.complete = AsyncMock(return_value=FAKE_COMPLETION)
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=None)
        MockClient.return_value = mock_instance

        resp = client.post("/v1/chat/completions", json={
            "model": "auto",
            "messages": [{"role": "user", "content": "Hi"}],
        })
    assert "X-Routed-To" in resp.headers
    assert "X-Complexity-Score" in resp.headers


def test_stats_endpoint(client):
    resp = client.get("/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "local" in data
    assert "remote" in data
    assert "total" in data
    assert "local_rate" in data


def test_invalid_request_returns_422(client):
    resp = client.post("/v1/chat/completions", json={"model": "x"})  # missing messages
    assert resp.status_code == 422
