import textwrap
import pytest
from pydantic import ValidationError
from privacy_serving.config import Config, RoutingConfig, load_config


def _write_config(tmp_path, content: str) -> str:
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent(content))
    return str(p)


_MINIMAL = """\
    local_model:
      base_url: "http://localhost:11434/v1"
      model: "qwen2.5:9b"
      api_key: "none"
    remote_model:
      base_url: "https://api.openai.com/v1"
      model: "gpt-4o"
      api_key: "sk-secret"
"""


def test_load_config_from_yaml(tmp_path):
    cfg = load_config(_write_config(tmp_path, _MINIMAL + "    routing:\n      complexity_threshold: 0.6\n    server:\n      host: '0.0.0.0'\n      port: 9000\n"))
    assert cfg.local_model.base_url == "http://localhost:11434/v1"
    assert cfg.local_model.model == "qwen2.5:9b"
    assert cfg.remote_model.model == "gpt-4o"
    assert cfg.routing.complexity_threshold == 0.6
    assert cfg.server.port == 9000


def test_env_var_expansion(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "sk-from-env")
    content = _MINIMAL.replace("sk-secret", "${TEST_API_KEY}")
    cfg = load_config(_write_config(tmp_path, content))
    assert cfg.remote_model.api_key == "sk-from-env"


def test_missing_env_var_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("UNDEFINED_VAR", raising=False)
    content = _MINIMAL.replace("sk-secret", "${UNDEFINED_VAR}")
    with pytest.raises(ValueError, match="UNDEFINED_VAR"):
        load_config(_write_config(tmp_path, content))


def test_threshold_out_of_range_raises():
    with pytest.raises(ValidationError):
        RoutingConfig(complexity_threshold=1.5)


def test_threshold_lower_bound_raises():
    with pytest.raises(ValidationError):
        RoutingConfig(complexity_threshold=-0.1)
