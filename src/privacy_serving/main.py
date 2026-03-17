from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse

from privacy_serving.clients import ModelClient
from privacy_serving.config import Config, load_config
from privacy_serving.models import ChatCompletionRequest
from privacy_serving.router import Router
from privacy_serving.stats import StatsTracker


def create_app(config: Config) -> FastAPI:
    router = Router(config)
    stats = StatsTracker()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.router = router
        app.state.stats = stats
        yield

    app = FastAPI(title="Privacy Serving Proxy", lifespan=lifespan)

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {"id": config.local_model.model, "object": "model", "owned_by": "local"},
                {"id": config.remote_model.model, "object": "model", "owned_by": "remote"},
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest) -> Response:
        destination, score = router.route(request)
        model_config = router.model_config_for(destination)

        async with ModelClient(model_config) as client:
            result = await client.complete(request)

        stats.record(destination)

        headers = {
            "X-Routed-To": destination,
            "X-Complexity-Score": f"{score:.4f}",
            "X-Model-Used": model_config.model,
        }
        return JSONResponse(content=result, headers=headers)

    @app.get("/stats")
    async def get_stats() -> dict[str, Any]:
        return stats.snapshot()

    return app


def main() -> None:
    import uvicorn

    config_path = os.environ.get("PRIVACY_SERVING_CONFIG", "config.yaml")
    config = load_config(config_path)
    app = create_app(config)
    uvicorn.run(app, host=config.server.host, port=config.server.port)


if __name__ == "__main__":
    main()
