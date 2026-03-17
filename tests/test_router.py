import pytest
from privacy_serving.config import Config, ModelConfig, RoutingConfig, ServerConfig
from privacy_serving.models import Message, ChatCompletionRequest
from privacy_serving.router import Router


@pytest.fixture
def config_threshold_05():
    return Config(
        local_model=ModelConfig(base_url="http://local/v1", model="local-model"),
        remote_model=ModelConfig(base_url="http://remote/v1", model="remote-model"),
        routing=RoutingConfig(complexity_threshold=0.5),
        server=ServerConfig(),
    )


def make_request(content: str) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="any",
        messages=[Message(role="user", content=content)],
    )


def test_simple_routes_to_local(config_threshold_05):
    router = Router(config_threshold_05)
    destination, score = router.route(make_request("Hi"))
    assert destination == "local"
    assert score < 0.5


def test_complex_routes_to_remote(config_threshold_05):
    router = Router(config_threshold_05)
    complex_content = (
        "Analyze the trade-offs between quicksort and mergesort. "
        "Prove that mergesort is O(n log n) step by step. "
        "Compare their space complexities. Why is this important? "
        "What are the implications for real-world systems? "
        "```python\nclass Sort:\n    pass\n```"
    )
    destination, score = router.route(make_request(complex_content))
    assert destination == "remote"
    assert score >= 0.5


def test_at_threshold_routes_to_remote(config_threshold_05):
    """Score exactly at threshold goes to remote (>=)."""
    router = Router(config_threshold_05)
    # patch complexity to return exactly 0.5
    router._compute = lambda _: 0.5
    destination, score = router.route(make_request("anything"))
    assert destination == "remote"
    assert score == 0.5


def test_returns_model_config_for_local(config_threshold_05):
    router = Router(config_threshold_05)
    router._compute = lambda _: 0.1
    destination, _ = router.route(make_request("Hi"))
    assert destination == "local"
    model_cfg = router.model_config_for(destination)
    assert model_cfg.model == "local-model"


def test_returns_model_config_for_remote(config_threshold_05):
    router = Router(config_threshold_05)
    router._compute = lambda _: 0.9
    destination, _ = router.route(make_request("Hi"))
    assert destination == "remote"
    model_cfg = router.model_config_for(destination)
    assert model_cfg.model == "remote-model"
