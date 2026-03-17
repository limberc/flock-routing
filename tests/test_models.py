# tests/test_models.py
import json
import pytest
from privacy_serving.models import ChatCompletionRequest, ChatCompletionResponse, Message


def test_request_parses_basic():
    data = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello!"}],
    }
    req = ChatCompletionRequest.model_validate(data)
    assert req.model == "gpt-4o"
    assert len(req.messages) == 1
    assert req.messages[0].role == "user"
    assert req.messages[0].content == "Hello!"
    assert req.stream is False


def test_request_with_system_message():
    data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Prove Fermat's Last Theorem."},
        ],
        "temperature": 0.7,
        "max_tokens": 2048,
    }
    req = ChatCompletionRequest.model_validate(data)
    assert req.messages[0].role == "system"
    assert req.temperature == 0.7


def test_response_serializes():
    resp = ChatCompletionResponse(
        id="chatcmpl-test",
        model="gpt-4o",
        choices=[{
            "index": 0,
            "message": {"role": "assistant", "content": "42"},
            "finish_reason": "stop",
        }],
        usage={"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
    )
    data = json.loads(resp.model_dump_json())
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["content"] == "42"


def test_request_extra_fields_pass_through():
    """Extra fields must be preserved — this proxy forwards unknown params to backends."""
    data = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}],
        "custom_param": "some_value",
        "another_extra": 42,
    }
    req = ChatCompletionRequest.model_validate(data)
    dumped = req.model_dump()
    assert dumped["custom_param"] == "some_value"
    assert dumped["another_extra"] == 42
