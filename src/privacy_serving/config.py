from __future__ import annotations

import os
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    base_url: str
    model: str
    api_key: str = "none"


class RoutingConfig(BaseModel):
    complexity_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080


class Config(BaseModel):
    local_model: ModelConfig
    remote_model: ModelConfig
    routing: RoutingConfig = RoutingConfig()
    server: ServerConfig = ServerConfig()


_ENV_VAR_RE = re.compile(r"\$\{([^}]+)\}")


def _expand_env_vars(text: str) -> str:
    """Replace ${VAR} patterns with environment variable values."""
    def replace(match: re.Match) -> str:
        var_name = match.group(1)
        value = os.environ.get(var_name)
        if value is None:
            raise ValueError(f"Environment variable '{var_name}' is not set")
        return value

    return _ENV_VAR_RE.sub(replace, text)


def load_config(path: str) -> Config:
    """Load configuration from a YAML file, expanding ${ENV_VAR} references."""
    raw = Path(path).read_text()
    expanded = _expand_env_vars(raw)
    data = yaml.safe_load(expanded)
    return Config.model_validate(data)
