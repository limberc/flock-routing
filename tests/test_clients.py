import json
import pytest
import httpx
from pytest_httpx import HTTPXMock
from privacy_serving.clients import ModelClient
from privacy_serving.config import ModelConfig
from privacy_serving.models import ChatCompletionRequest, Message


FAKE_RESPONSE = {
    "id": "chatcmpl-abc",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "qwen2.5:9b",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
}


@pytest.fixture
def local_config() -> ModelConfig:
    return ModelConfig(base_url="http://localhost:11434/v1", model="qwen2.5:9b", api_key="none")


@pytest.fixture
def request_payload() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="qwen2.5:9b",
        messages=[Message(role="user", content="Hi")],
    )


@pytest.mark.asyncio
async def test_complete_returns_parsed_response(httpx_mock: HTTPXMock, local_config, request_payload):
    httpx_mock.add_response(
        method="POST",
        url="http://localhost:11434/v1/chat/completions",
        json=FAKE_RESPONSE,
        status_code=200,
    )
    async with ModelClient(local_config) as client:
        result = await client.complete(request_payload)
    assert result["choices"][0]["message"]["content"] == "Hello!"


@pytest.mark.asyncio
async def test_complete_overrides_model(httpx_mock: HTTPXMock, local_config, request_payload):
    """Client should replace the request model name with the configured model."""
    captured = {}

    def capture(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=FAKE_RESPONSE)

    httpx_mock.add_callback(capture)
    async with ModelClient(local_config) as client:
        await client.complete(request_payload)
    assert captured["body"]["model"] == "qwen2.5:9b"


@pytest.mark.asyncio
async def test_http_error_propagates(httpx_mock: HTTPXMock, local_config, request_payload):
    httpx_mock.add_response(status_code=500, text="Internal Server Error")
    async with ModelClient(local_config) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await client.complete(request_payload)
