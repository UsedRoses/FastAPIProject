import os
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

# export APP_ENV=prod 来切换环境
APP_ENV = os.environ.get("APP_ENV", "dev")


class Settings(BaseSettings):
    # --- 配置项定义 ---
    project_name: str = "Default Project Name"
    api_prefix: str = ""
    debug: bool = False
    database_url: str = ""

    # --- Pydantic V2 配置 ---
    model_config = SettingsConfigDict(
        # 支持多个 env 文件级联读取。
        # 会先读取 .env，然后读取 .env.{APP_ENV}，后者会覆盖前者的同名配置。
        # 最后，系统真实的系统环境变量优先级最高，会覆盖文件中的配置。
        env_file=(".env", f".env.{APP_ENV}"),
        env_file_encoding="utf-8",
        # 优雅点 2: 忽略 env 文件中未在类中定义的额外字段，防止报错
        extra="ignore"
    )

    ROUTER_DIR: str = '/controller'


# 这样在项目的生命周期内，不管调用多少次 get_settings()，都只会读取一次文件
@lru_cache
def get_settings() -> Settings:
    return Settings()


# 全局可用实例（适用于非路由请求的普通模块调用）
settings = get_settings()