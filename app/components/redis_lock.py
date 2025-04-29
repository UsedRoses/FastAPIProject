from contextlib import asynccontextmanager

from redis.asyncio import Redis
from redis.exceptions import LockError


class RedisLock:

    def __init__(self, redis: Redis, lock_key: str, lock_expiry: int = 10, blocking_timeout: int = 0):
        self.lock_key = f'rlock:{lock_key}'
        self.lock_expiry = lock_expiry
        self.blocking_timeout = blocking_timeout
        self.redis = redis
        self.lock = self.redis.lock(name=self.lock_key, timeout=self.lock_expiry)

    async def acquire(self):
        block = self.blocking_timeout != 0
        if not await self.lock.acquire(blocking=block, blocking_timeout=self.blocking_timeout):
            return False
        return True

    async def release(self):
        try:
            await self.lock.release()
        except LockError as e:
            # 捕获 LockError 时，做日志记录或其他处理
            if e.message == 'Cannot release an unlocked lock':
                print(f"Attempted to release an unlocked lock for {self.lock_key}.")
            else:
                raise e


    @asynccontextmanager
    async def try_lock(self):
        await self.acquire()
        try:
            yield
        finally:
            await self.release()
