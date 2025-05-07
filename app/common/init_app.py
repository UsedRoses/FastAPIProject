import os
import importlib.util
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from common.public_configuration.mysql_configuration import mysql_startup, mysql_shutdown
from common.public_configuration.redis_configuration import redis_startup, redis_shutdown
from middleware.custom_exception_handler import custom_exception_handler
from middleware.response_middleware import ResponseMiddleware
from middleware.user_info_middleware import UserInfoContextMiddleware
from utils.aiohttp_client_util import close_client


def init_middlewares(app: FastAPI):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(UserInfoContextMiddleware)
    app.add_middleware(ResponseMiddleware)


def register_exceptions(app: FastAPI):
    app.add_exception_handler(Exception, custom_exception_handler)


def register_routers(app: FastAPI, routers_dir: str):
    print(f"扫描路由: {routers_dir}")
    # 遍历目录，自动导入所有路由模块
    for root, dirs, files in os.walk(routers_dir):
        for filename in files:
            if filename.endswith(".py"):
                module_name = filename[:-3]  # 去掉 .py 后缀
                module_path = os.path.join(root, filename)

                # 动态导入模块
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # 自动注册路由
                if hasattr(module, "router"):
                    print(module.router)
                    app.include_router(module.router)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # await mysql_startup()
    # await redis_startup()

    yield

    # await mysql_shutdown()
    # await redis_shutdown()
    await close_client()


app = FastAPI(title="your_fastapi_project", lifespan=lifespan)