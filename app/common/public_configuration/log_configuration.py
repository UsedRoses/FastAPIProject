import logging
from logging.config import dictConfig
from uuid import uuid4

from aliyun.log import QueuedLogHandler

from common.constant import get_user_info_context
from common.public_configuration.public_settings import settings
from components.aliyun import get_aliyun_log_access


class AliQueuedLogHandler(QueuedLogHandler):
    def __init__(self, get_credentials_func, endpoint, project, logstore, *args, **kwargs):
        """
        :param get_credentials_func: 用于动态获取密钥的函数
        :param endpoint: 阿里云日志服务的终端节点
        :param logstore: 日志存储的名称
        """
        self.get_credentials_func = get_credentials_func

        # 初次获取密钥
        credentials = get_credentials_func()
        required_params = ['access_key_id', 'access_key']
        for param in required_params:
            if param not in credentials:
                raise ValueError(f"Missing required credential parameter: {param}")

        # 更新 kwargs，并包含 endpoint 和 logstore
        kwargs.update({
            'access_key_id': credentials['access_key_id'],
            'access_key': credentials['access_key'],
            'end_point': endpoint,
            'project': project,
            'log_store': logstore,
            'extract_json': True,
            'extract_json_drop_message': True
        })

        super().__init__(*args, **kwargs)

        self.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))

    def emit(self, record: logging.LogRecord):
        """
        在每次 emit 时检查并动态更新密钥。
        """
        new_credentials = self.get_credentials_func()
        if new_credentials['access_key_id'] != self.access_key_id or new_credentials['access_key'] != self.access_key:
            # 动态更新密钥
            self.access_key_id = new_credentials['access_key_id']
            self.access_key = new_credentials['access_key']
        super().emit(record)


def setup_logging():
    # 读取环境变量，根据环境切换日志配置
    env = settings.ENVIRONMENT
    # 配置日志
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'verbose': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            },
            'simple': {
                'format': '%(levelname)s %(message)s',
            },
        },
        'handlers': {
            'console': {
                'level': 'DEBUG',
                'class': 'logging.StreamHandler',
                'formatter': 'verbose',
            },
            'aliyun': {
                'level': 'INFO',  # 生产环境推送INFO及以上日志
                'class': 'logging.StreamHandler',  # 暂时使用 StreamHandler 作为占位符
                'formatter': 'verbose',
            },
        },
        'loggers': {
            'api_log': {
                'handlers': ['aliyun'] if env in ['production', 'development'] else ['console'],
                'level': 'INFO',
                'propagate': False,
            },
        },
    }

    dictConfig(logging_config)

    if env in ['production', 'development']:
        aliyun_handler = AliQueuedLogHandler(
            get_credentials_func=get_aliyun_log_access,
            endpoint="us-west-1.log.aliyuncs.com",
            project="kodecrypto-us",
            logstore="kodecrypto",
        )
        aliyun_handler.setLevel(logging.INFO)

        # 手动将 aliyun 处理器加入到 'api_log' logger 中
        logger = logging.getLogger('api_log')
        logger.handlers = [aliyun_handler]  # 替换控制台处理器为 aliyun 处理器


setup_logging()

logger = logging.getLogger('api_log')


class LoggerData:

    def __init__(self, message: str = None):
        self.log_id = str(uuid4())
        self._message = message
        self.user_info = None
        self._data = {}
        self._env_name = settings.ENVIRONMENT

    def set_message(self, message: str) -> "LoggerData":
        self._message = message
        return self

    def append_data(self, **kwargs) -> "LoggerData":
        self._data.update(kwargs)
        return self

    def del_data(self, key: str) -> "LoggerData":
        self._data.pop(key)
        return self

    def clear_data(self):
        self._data = {}
        return self

    def _cleaned_data(self) -> dict:
        # 获取实例的属性字典，排除父类的属性
        obj_dict = {k: v for k, v in self.__dict__.items() if not k.startswith("__")}

        # 清理为空的元素
        cleaned_dict = {k: v for k, v in obj_dict.items() if v not in [None, "", {}, [], set()]}

        # 特别处理 data 字段，清除空元素并提取到顶级
        if "_data" in cleaned_dict:
            # 把 data 字典中的每个键值对提取到顶级
            cleaned_dict.update(cleaned_dict.pop("_data"))  # 从 cleaned_dict 中移除 'data' 并将其内容提取出来

        return cleaned_dict

    def info(self):
        user_info = get_user_info_context()
        if user_info:
            self.user_info = user_info.dict()
        logger.info(self._cleaned_data())

    def warning(self):
        user_info = get_user_info_context()
        if user_info:
            self.user_info = user_info.dict()
        logger.warning(self._cleaned_data())

    def error(self):
        user_info = get_user_info_context()
        if user_info:
            self.user_info = user_info.dict()
        logger.error(self._cleaned_data())
