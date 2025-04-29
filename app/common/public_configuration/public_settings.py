import os

from pydantic_settings import BaseSettings

class BaseConfig(BaseSettings):
    # 当前环境配置
    ENVIRONMENT: str

    APP_NAME: str = "your app name"
    # 数据库
    DB_HOST: str
    DB_USER: str
    DB_PASSWORD: str
    DB_PORT: str
    DB_DATABASE: str
    # 中台用户数据库
    USER_DB_HOST: str
    USER_DB_USER: str
    USER_DB_PASSWORD: str
    USER_DB_PORT: str
    USER_DB_DATABASE: str

    # redis
    REDIS_HOST: str
    REDIS_PORT: int
    REDIS_USERNAME: str
    REDIS_PASSWORD: str
    REDIS_DB: int

    SQL_DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    DEBUG: bool = False

    ROUTER_DIR: str = "controller"


def get_settings():
    return BaseConfig()


settings: BaseConfig = get_settings()


os.environ['REPLICATE_API_TOKEN'] = '*************'