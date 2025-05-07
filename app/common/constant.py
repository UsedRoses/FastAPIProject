import contextvars
from typing import Optional

from pydantic import BaseModel

from common.public_configuration.public_settings import settings

ALIYUN_STS_TOKEN_KEY = f'{settings.APP_NAME}:aliyun:sts:token'


class UserInfoContext(BaseModel):
    user_id: int = None
    username: str = None
    email: str = None

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    def to_dict(self) -> dict:
        return self.model_dump()


# 创建一个上下文变量，用于存储每个协程的独立数据
db_context: contextvars.ContextVar[Optional[UserInfoContext]] = contextvars.ContextVar('user_info_context', default=None)

# 设置上下文变量的值
def set_user_info_context(user_info_context: UserInfoContext):
    db_context.set(user_info_context)

# 获取上下文变量的值
def get_user_info_context():
    context = db_context.get()
    if context is None:
        return UserInfoContext()
    return context
