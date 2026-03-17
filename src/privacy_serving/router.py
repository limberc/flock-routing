from __future__ import annotations

from typing import Callable, Literal

from privacy_serving.complexity import compute_complexity
from privacy_serving.config import Config, ModelConfig
from privacy_serving.models import ChatCompletionRequest

Destination = Literal["local", "remote"]


class Router:
    """Decides whether a request should go to the local or remote model."""

    def __init__(self, config: Config) -> None:
        self._config = config
        # Allows tests to swap in a stub complexity function
        self._compute: Callable[[ChatCompletionRequest], float] = (
            lambda req: compute_complexity(req.messages)
        )

    def route(self, request: ChatCompletionRequest) -> tuple[Destination, float]:
        """
        Return (destination, complexity_score).

        Requests with score >= threshold go to "remote"; below go to "local".
        """
        score = self._compute(request)
        destination: Destination = (
            "remote" if score >= self._config.routing.complexity_threshold else "local"
        )
        return destination, score

    def model_config_for(self, destination: Destination) -> ModelConfig:
        if destination == "local":
            return self._config.local_model
        return self._config.remote_model
