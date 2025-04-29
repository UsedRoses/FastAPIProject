from asyncio import Lock
from urllib.parse import quote
from redis.asyncio import Redis
from typing import Optional
import functools

from common.public_configuration.public_settings import settings


def get_redis_url() -> str:
    password = quote(settings.REDIS_PASSWORD)
    return f'redis://{settings.REDIS_USERNAME}:{password}@{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}'


class RedisConfig:
    _redis: Optional[Redis] = None
    _max_connections: int = 100
    _lock = Lock()  # 线程安全锁

    @classmethod
    async def initialize(cls):
        """初始化 Redis 连接池"""
        if cls._redis is None:
            async with cls._lock:
                if cls._redis is None:
                    try:
                        cls._redis = Redis(
                            host=settings.REDIS_HOST,
                            port=settings.REDIS_PORT,
                            username=settings.REDIS_USERNAME,
                            password=settings.REDIS_PASSWORD,
                            db=settings.REDIS_DB,
                            decode_responses=True,
                            max_connections=cls._max_connections,
                            auto_close_connection_pool=False
                        )

                        # 测试连接
                        await cls._redis.ping()
                    except Exception as e:
                        raise e

    @classmethod
    async def get_redis(cls) -> Redis:
        """获取 Redis 实例"""
        if cls._redis is None:
            await cls.initialize()
        print("Redis connection established.")
        return cls._redis

    @classmethod
    async def close(cls):
        """关闭 Redis 连接池"""
        if cls._redis:
            await cls._redis.close()
            print("Redis connection closed.")


async def get_redis_connection() -> Redis:
    return await RedisConfig.get_redis()


def get_singleton_redis_connection() -> Redis:
    return Redis.from_url(url=get_redis_url(), encoding="utf-8", decode_responses=True)


def with_redis_client(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        redis_client = await get_redis_connection()
        return await func(redis_client, *args, **kwargs)
    return wrapper


async def redis_startup():
    await RedisConfig.initialize()


async def redis_shutdown():
    await RedisConfig.close()
