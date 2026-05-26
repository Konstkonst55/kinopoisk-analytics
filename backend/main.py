import json
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.coder import Coder
from redis import asyncio as aioredis

from backend.config import REDIS_URL, setup_logger
from backend.routes import router

logger = setup_logger("FastAPIServer")


class PydanticCoder(Coder):
    @classmethod
    def encode(cls, value: Any) -> bytes:
        if hasattr(value, "model_dump"):
            return json.dumps(value.model_dump(), default=str).encode("utf-8")
        elif hasattr(value, "dict"):
            return json.dumps(value.dict(), default=str).encode("utf-8")

        return json.dumps(value, default=str).encode("utf-8")

    @classmethod
    def decode(cls, value: Any) -> Any:
        if isinstance(value, bytes):
            value = value.decode("utf-8")

        return json.loads(value)

    @classmethod
    def decode_as_type(cls, value: Any, type_: Any = None) -> Any:
        result = cls.decode(value)

        if type_ is not None:
            try:
                if hasattr(type_, "model_validate"):
                    return type_.model_validate(result)
                elif hasattr(type_, "parse_obj"):
                    return type_.parse_obj(result)
            except Exception:
                pass

        return result


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting FastAPI server, connecting to Redis")
    redis = aioredis.from_url(REDIS_URL, encoding="utf8", decode_responses=True)

    FastAPICache.init(RedisBackend(redis), prefix="kinopoisk-cache", coder=PydanticCoder)

    logger.info("Redis cache initialized successfully")

    yield

    logger.info("Shutting down FastAPI server")


app = FastAPI(title="Kinopoisk ABSA & Summary API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    logger.info(
        f"{request.method} {request.url.path} - " f"Status: {response.status_code} - Time: {process_time:.4f}s"
    )

    return response


app.include_router(router)
