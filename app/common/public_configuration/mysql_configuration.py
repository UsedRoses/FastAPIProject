from tortoise import Tortoise
from tortoise.router import ConnectionRouter

from common.public_configuration.public_settings import settings
from models.table import user


async def mysql_startup():
    config = {
        'connections': {
            'default': {
                'engine': 'tortoise.backends.mysql',
                'credentials': {
                    'host': settings.DB_HOST,
                    'port': settings.DB_PORT,
                    'user': settings.DB_USER,
                    'password': settings.DB_PASSWORD,
                    'database': settings.DB_DATABASE,
                    'minsize': 5,  # 最小连接池大小
                    'maxsize': 100,  # 最大连接池大小
                    'pool_recycle': 3600,  # 连接池回收时间（秒）
                }
            },
            'zuser': {
                'engine': 'tortoise.backends.mysql',
                'credentials': {
                    'host': settings.USER_DB_HOST,
                    'port': settings.USER_DB_PORT,
                    'user': settings.USER_DB_USER,
                    'password': settings.USER_DB_PASSWORD,
                    'database': settings.USER_DB_DATABASE,
                    'minsize': 5,  # 最小连接池大小
                    'maxsize': 50,  # 最大连接池大小
                    'pool_recycle': 3600,  # 连接池回收时间（秒）
                }
            },
        },
        'apps': {
            'default': {
                'models': [user],
                # If no default_connection specified, defaults to 'default'
                'default_connection': 'default',
            },
        },
        'routers': [CustomRouter],
        'use_tz': False,
        'timezone': 'UTC'
    }
    await Tortoise.init(config=config)
    print("Tortoise init start")
    # await Tortoise.generate_schemas()


async def mysql_shutdown():
    await Tortoise.close_connections()
    print("Tortoise close")


class CustomRouter(ConnectionRouter):
    def db_for_read(self, model):
        print(model)
        if model._model.table in ["z_user", "social_account"]:
            return "zuser"
        return "default"

    def db_for_write(self, model):
        return "default"


