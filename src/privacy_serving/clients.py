from __future__ import annotations

from types import TracebackType
from typing import Any

import httpx

from privacy_serving.config import ModelConfig
from privacy_serving.models import ChatCompletionRequest


class ModelClient:
    """Async HTTP client for an OpenAI-compatible chat completions endpoint."""

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._http: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "ModelClient":
        headers = {"Content-Type": "application/json"}
        if self._config.api_key and self._config.api_key != "none":
            headers["Authorization"] = f"Bearer {self._config.api_key}"
        self._http = httpx.AsyncClient(
            base_url=self._config.base_url,
            headers=headers,
            timeout=120.0,
        )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._http:
            await self._http.aclose()

    async def complete(self, request: ChatCompletionRequest) -> dict[str, Any]:
        """Forward a chat completion request, replacing model name with config value."""
        if self._http is None:
            raise RuntimeError("ModelClient must be used as an async context manager")

        payload = request.model_dump(exclude_none=True)
        payload["model"] = self._config.model  # always use the configured model name

        response = await self._http.post("/chat/completions", json=payload)
        response.raise_for_status()
        return response.json()
