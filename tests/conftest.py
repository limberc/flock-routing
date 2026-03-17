# tests/conftest.py
import pytest
from privacy_serving.config import Config, ModelConfig, RoutingConfig, ServerConfig


@pytest.fixture
def default_config() -> Config:
    return Config(
        local_model=ModelConfig(base_url="http://localhost:11434/v1", model="qwen2.5:9b", api_key="none"),
        remote_model=ModelConfig(base_url="https://api.openai.com/v1", model="gpt-4o", api_key="sk-test"),
        routing=RoutingConfig(complexity_threshold=0.5),
        server=ServerConfig(host="127.0.0.1", port=8080),
    )
